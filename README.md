# 3D Droplet Dynamics on Heterogeneous surfaces with mass transfer

This repository contains an implementation of the asymptotic model in Python (v.3) derived in

* [Droplet motion on chemically heterogeneous substrates with mass transfer. II. Three-dimensional dynamics](https://arxiv.org/abs/2007.07008)

In this work, we develop a reduced model based on matched asymptotic expansions that allows us to simulate the dynamics of droplets on heterogeneous surfaces as they undergo changes in their mass. Extended comparisons are offered with simulations to the long-wave governing equation, highlight the generally excellent agreement with the governing long-wave equation and, importantly, at a small fraction of the computing cost. This work expands upon the previous work

* [Savva, Groves & Kalliadasis (2019) Droplet dynamics on chemically heterogeneous substrates J. Fluid Mech. 859 321--361](https://doi.org/10.1017/jfm.2018.758)

by including additional terms in the asymptotic expansions and incorporating prescribed mass transfer effects.

## Files

The provided python scripts reproduce the figures in the above mentioned paper, namely Figures 2 to 8, namedly accordingly and within their corresponding folders. Generating the figures relies on the data and codes in the directory `main`, and, whenever appropriate on PDE data, which was obtained using MATLAB scripts (not currently made openly available). The `main` folder includes

* `ODEdrop.py`, which implements the solver of the asymptotic model
* `parameters.mat` and `hypergeom.mat` are matlab files for storing the parameters of the asymptotic model, as well as the Gaussian hypergeometric functions which are necessary for computing localized fluxes. 

In addition, the following are provided:

* the code and PDE data for generating Figure 9 in the JFM paper above, as a means to compare the how well the different approaches compare
* A jupyter notebook which explains the functionality of the solver (pending).
