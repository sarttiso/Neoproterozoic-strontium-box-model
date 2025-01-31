# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:23:08 2024

@author: Adrian
"""

from numba import njit
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from helper import *
import numpy as np

###
### FORWARD MODEL
###

@njit
def sigmoid(t, scale=1e3):
    return 1/(1+np.exp(-t/scale))

# flux functions with smoothing (could just use one, but let's have them separate)
@njit
def F_in_fun(t, F_in_params):
    """
    t: time(s) at which to evaluate flux
    F_in_params: Flux values at (in order) t0, t1_st(-), t1_st(+)=t2_st(-), t2_st(+), t1
        len=5.
    """
    # tonian contribution
    F_ton = ( (F_in_params[1] - F_in_params[0])/dt_ton * (t-t0) + F_in_params[0] ) * sigmoid(t1_st-t)
    # sturtian contribution
    F_stu = F_in_params[2] * sigmoid(t-t1_st) * sigmoid(t2_st-t)
    # middle cryogenian contribution
    F_mcy = ( (F_in_params[4] - F_in_params[3])/dt_mcy * (t-t2_st) + F_in_params[3] ) * sigmoid(t-t2_st)
    return F_ton + F_stu + F_mcy

@njit
def pack(F_ri, F_di, F_ht, F_out, N, alpha):
    """
    pack params into array
    """
    params = np.concatenate((np.asarray(F_ri), 
                             np.asarray(F_di), 
                             np.asarray(F_ht), 
                             np.asarray(F_out),
                             np.asarray([N, alpha])))
    return params

@njit
def unpack(params):
    """
    unpack params
    """
    F_ri = params[0:5]
    F_di = params[5:10]
    F_ht = params[10:15]
    F_out = params[15:20]
    N = params[20]
    alpha = params[21]
    return F_ri, F_di, F_ht, F_out, N, alpha

@njit
def F_out_fun(t, F_out_params):
    """
    Complement to F_in_fun.
    t: time(s) at which to evaluate flux
    F_out_params: Flux values at (in order) t0, t1_st(-), t1_st(+)=t2_st(-), t2_st(+), t1
        len=5.
    """
    # tonian contribution
    F_ton = ( (F_out_params[1] - F_out_params[0])/dt_ton * (t-t0) + F_out_params[0] ) * sigmoid(t1_st-t)
    # sturtian contribution
    F_stu = F_out_params[2] * sigmoid(t-t1_st) * sigmoid(t2_st-t)
    # middle cryogenian contribution
    F_mcy = ( (F_out_params[4] - F_out_params[3])/dt_mcy * (t-t2_st) + F_out_params[3] ) * sigmoid(t-t2_st)
    return F_ton + F_stu + F_mcy

# factors for each parameter to make all similar magnitude for optimization
F_ri_norm_fact = 1e10
F_di_norm_fact = 1e9
F_ht_norm_fact = 1e10
N_norm_fact = 1e16
rSr_norm_fact = 1e-3
param_norm_fact = np.hstack([np.tile(np.array([F_ri_norm_fact, F_di_norm_fact, F_ht_norm_fact]), 4), # flux normalizations
                             np.array([N_norm_fact, # reservoir
                                       rSr_norm_fact] # Sr ratio 
                                     )])

@njit
def rhs_ivp(t, X, params):
    """
    X: [N, alpha]
        N: reservoir size
        alpha: seawater Sr isotopic ratio
    t: time (no time term on rhs)
    params: (normalized) stepwise fluxes for each of the 3 time windows as well as
        initial reservoir size and initial seawater Sr ratio
    """
    # unpack parameters
    N, alpha = X

    # un-normalize parameters
    # params = params * param_norm_fact

    F_ri_ton, F_di_ton, F_ht_ton, \
    F_ri_stu, F_di_stu, F_ht_stu, \
    F_ri_mcy, F_di_mcy, F_ht_mcy, \
    F_out_ton, F_out_stu, F_out_mcy, \
    N0, rSr0 = params
    
    # equations
    dN_dt = F_ri_fun(t, F_ri_ton, F_ri_stu, F_ri_mcy) + \
            F_di_fun(t, F_di_ton, F_di_stu, F_di_mcy) + \
            F_ht_fun(t, F_ht_ton, F_ht_stu, F_ht_mcy) - \
            F_out_fun(t, F_out_ton, F_out_stu, F_out_mcy)
    dalpha_dt = 1/N * (F_ri_fun(t, F_ri_ton, F_ri_stu, F_ri_mcy)*(rSr_ri - alpha) + \
                       F_di_fun(t, F_di_ton, F_di_stu, F_di_mcy)*(rSr_di - alpha) + \
                       F_ht_fun(t, F_ht_ton, F_ht_stu, F_ht_mcy)*(rSr_ht - alpha))
    
    return dN_dt, dalpha_dt

@njit
def rhs_ode(X, t, params):
    """
    X: [N, alpha]
        N: reservoir size
        alpha: seawater Sr isotopic ratio
    t: time (no time term on rhs)
    params: (normalized) stepwise fluxes for each of the 3 time windows as well as
        initial reservoir size and initial seawater Sr ratio
    """
    # unpack parameters
    N, alpha = X

    # un-normalize parameters
    # params = params * param_norm_fact
    F_ri, F_di, F_ht, F_out, N0, rSr0 = unpack(params)
    
    # equations
    dN_dt = F_in_fun(t, F_ri) + F_in_fun(t, F_di) + F_in_fun(t, F_ht) - F_out_fun(t, F_out)
    dalpha_dt = 1/N * (F_in_fun(t, F_ri)*(rSr_ri - alpha) + \
                       F_in_fun(t, F_di)*(rSr_di - alpha) + \
                       F_in_fun(t, F_ht)*(rSr_ht - alpha))
    
    return dN_dt, dalpha_dt

# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def pytensor_forward_model_matrix_ode(params):
    # print(params)
    cur_forward = odeint(func=rhs_ode,  
                         y0=params[-2:], # check
                         t=sr_data_ocn_binned.index.values, 
                         args=(params,))
    return cur_forward[:, 1]

# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def pytensor_forward_model_matrix_ivp(params):
    cur_forward = solve_ivp(rhs_ivp,  
                      [t0, t1], 
                      params[-2:], # check
                      t_eval=sr_data_ocn_binned.index.values, 
                      args=(params,), 
                      method='RK45', rtol=1e-10, atol=1e-15)
    if not cur_forward.success:
        print(f'ivp failure: {params}')
        return np.inf * np.ones(len(sr_data_avg))
    return cur_forward.y[1, :]

@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dvector])
def print_mcy1_upper(F_out_mcy1_upper_avg, F_out_mcy1_upper_min, F_out_mcy1_upper):
    upper = pt.math.min([F_out_mcy1_upper_avg, F_out_mcy1_upper_min])
    if upper < F_out_mcy1_upper:
        print(f'upper (avg): {F_out_mcy1_upper_avg}\nupper (min): {F_out_mcy1_upper_min}\nupper: {F_out_mcy1_upper}')

@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def print_F_in(F_in):
    print(f'F_in[0]: {F_in[0]}\nF_in[1]: {F_in[1]}')

@as_op(itypes=[pt.dscalar], otypes=[pt.dvector])
def print_F_out_fact_ton0_upper(F_out_fact_ton0_upper):
    print(f'F_out_fact_ton0_upper: {F_out_fact_ton0_upper}')