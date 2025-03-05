import numpy as np
import polars as pl
import bambi as bmb
import arviz as az

# Import foxes data.
foxes = pl.read_csv('data/foxes.csv')

# Use Bambi to estimate the direct causal effect of avgfood on weight.
bambi_model_01 = bmb.Model('weight ~ avgfood + groupsize', foxes.to_pandas())
bambi_model_01

# Calls pm.sample().
bambi_fit_01 = bambi_model_01.fit()
az.plot_trace(bambi_fit_01, compact = False)

# Visualize marginal posteriors.
az.plot_forest(bambi_fit_01, var_names = ['avgfood', 'groupsize'], combined = True, hdi_prob = 0.95)

# Use Bambi to estimate the direct causal effect of avgfood on weight by group.
bambi_model_02 = bmb.Model('weight ~ (avgfood|group) + (groupsize|group)', foxes.to_pandas(), noncentered = True)
bambi_model_02

# Calls pm.sample().
bambi_fit_02 = bambi_model_02.fit()
az.plot_trace(bambi_fit_02, compact = False)

# Visualize marginal posteriors.
az.plot_forest(bambi_fit_02, var_names = ['avgfood|group', 'groupsize|group'], combined = True, hdi_prob = 0.95)

