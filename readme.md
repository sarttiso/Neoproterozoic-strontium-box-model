# Sr Box Model

This repository contains the data and code for reproducing Bayesian box modeling of the Neoproterozoic Sr reservoir and isotopic composition.

Use the `environment.yml` to set up a conda environment with the necessary packages:

```bash
conda env create -f environment.yml
```

All modeling, analysis, and generation of figures is performed in the [`strontium_box_model`](strontium_box_model.ipynb) file Jupyter notebook.

The Markov Chain Monte Carlo output used to produce the figures in the manuscript is saved in the `models\2025-02-05_smc_12-chains_10k.nc` file, which can be loaded in python via Arviz as with `trace = az.from_netcdf('models/2025-02-05_smc_12-chains_10k.nc')`. 