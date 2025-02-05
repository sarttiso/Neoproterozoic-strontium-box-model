import numpy as np
import pandas as pd

from scipy.optimize import least_squares, minimize, basinhopping
from scipy.integrate import odeint, solve_ivp
from scipy import stats
from scipy.stats import iqr



###
### USEFUL VALUES
###

# sturtian
t1_st = -717e6
t2_st = -660.99e6
# marinoan
t1_ma = -639e6
t2_ma = -635e6

N_modern = 1.25e17       # mols, initial Sr reservoir, modern
rSr_ton = 0.70625    # Tonian seawater isotopic ratio
rSr0 = 0.707  # initial
rSr_ri = 0.7092     # isotopic ratio for riverine input
rSr_ht = 0.703      # isotopic ratio for hydrothermal input
rSr_mantle = 0.7023 # mantle
rSr_di = rSr_ton    # isotopic ratio for diagenetic inputs

F_ri_modern = 4e10    # mol/yr
F_di_modern = 5.5e9   # mol/yr
F_ht_modern = 1.4e10  # mol/yr

# tonian
t0 = -750e6

# middle cryogenian
t1 = -650e6

# durations
dt_ton = t1_st - t0
dt_stu = t2_st - t1_st
dt_mcy = t1 - t2_st

# time 
dt = 1e5
t = np.arange(t0, t1+dt, dt)
nt = len(t)

# number of variables for each epoch
n_ton = np.sum(t < t1_st)
n_stu = np.sum((t >= t1_st) & (t < t2_st))
n_mcy = np.sum(t >= t2_st)

###
### DATA
###

sr_data = pd.read_excel('Sr-data.xlsx')
sr_data['Age (Ma)'] = sr_data['Age (Ma)'] * -1e6 # work in years with correct younging

# sort time values
sr_data.sort_values('Age (Ma)', inplace=True)

# separate measurements interpreted to reflect mixing of stratified ocean 
sr_data_mix = sr_data[sr_data['mixing'] == True].copy()
sr_data_ocn = sr_data[sr_data['mixing'] == False].copy()

# binned average (every 100 kyr)
t_bounds = np.arange(t0 - 5e4, t1+1e5, 1e5)
t_cen = np.arange(t0, t1+5e4, 1e5)

# bin
sr_data_ocn['t_cen'] = pd.cut(sr_data_ocn['Age (Ma)'], t_bounds, labels=t_cen)
# dataframe for binned data
sr_data_ocn_binned = pd.DataFrame(index=t_cen, columns=['mean', 'median', 'std'])
# grouping on bins
sr_data_ocn_binned['mean'] = sr_data_ocn.groupby('t_cen', observed=False)['87Sr/86Sr'].mean()
sr_data_ocn_binned['median'] = sr_data_ocn.groupby('t_cen', observed=False)['87Sr/86Sr'].median()
sr_data_ocn_binned['std'] = sr_data_ocn.groupby('t_cen', observed=False)['87Sr/86Sr'].std()
sr_data_ocn_binned['iqr'] = sr_data_ocn.groupby('t_cen', observed=False)['87Sr/86Sr'].agg(iqr)
# keep only bins with data
sr_data_ocn_binned = sr_data_ocn_binned[sr_data_ocn_binned['mean'].notna()]
# set std to some value if nan or 0
sr_std = 5e-5
sr_data_ocn_binned.loc[sr_data_ocn_binned['std'].isna(), 'std'] = sr_std
sr_data_ocn_binned.loc[sr_data_ocn_binned['std']==0, 'std'] = sr_std
sr_data_ocn_binned.loc[sr_data_ocn_binned['iqr'].isna(), 'iqr'] = sr_std
sr_data_ocn_binned.loc[sr_data_ocn_binned['iqr']==0, 'iqr'] = sr_std

sr_data_ocn_binned['unc'] = 2e-4
idx = sr_data_ocn_binned.index.isin(np.arange(-660.9e6, -660e6, 1e5))
sr_data_ocn_binned.loc[idx, 'unc'] = 1.2e-4
idx = sr_data_ocn_binned.index == -661e6
sr_data_ocn_binned.loc[idx, 'unc'] = 1.2e-4

# also look at just data in t_cen
sr_data_ocn = sr_data_ocn.dropna(subset='t_cen')
# set uncertainty on original data to be variability of Sr within its bin
sr_data_ocn['std'] = sr_data_ocn_binned.loc[sr_data_ocn['t_cen'], 'std'].values

###
### FUNCTIONS
###

ocnvol_t = 1.332e9 # km3

def ocnvol_sample(t, n_samp=1):
    """
    model Snowball ocean volume as uniformly sampled between 685-995 x 10e6 km3 based on uncertainties in sea glacier and terrestrial ice volumes.
    non-Snowball ice volume is taken to be same as present.
    """
    t = np.atleast_1d(t)
    vols = np.zeros((len(t), n_samp))
    snow_idx = (t < -t2_st) & (t > -t1_st)
    vols[snow_idx, :] = stats.uniform.rvs(685e6, 995e6-685e6, size=(1, n_samp))
    vols[~snow_idx, :] = ocnvol_t
    return vols.squeeze()

def sr_inv2conc(inv, ocnvol):
    """
    given an inventory of moles in the ocean as well as ocean volume, compute Sr concentration
    inventory in moles
    ocnvol in cubic km
    """
    ocnmass = ocnvol * 1020e9
    return inv / ocnmass * 1e6