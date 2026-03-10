#!/usr/bin/env python
"""
Trading Cost Estimation - script version of the notebook.
Saves all plots to scripts/output/ for offline inspection.

Usage (inside the container):
    python scripts/run_trading_cost_estimation.py
"""

import os
import typing as tp

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjags

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

figsize = (12, 8)
plt.rcParams["figure.figsize"] = figsize


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved {path}')


# ── Read Data ──────────────────────────────────────────────────────────────

print('Reading data...')
data_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'data')

def read_yahoo_log_return_df(filepath, dropna=True):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Log Return'] = np.log(df['Adj Close']).diff().dropna()
    if dropna:
        df = df.dropna().copy()
    return df

log_return_df = read_yahoo_log_return_df(os.path.join(data_dir, 'F.csv'), dropna=False)
spy_log_return_df = read_yahoo_log_return_df(os.path.join(data_dir, 'SPY.csv'), dropna=False)
spy_log_return_df = (
    spy_log_return_df
    .loc[:, ['Date', 'Log Return']]
    .rename(columns={'Log Return': 'Market Log Return'})
    .copy()
)

merged_df = pd.merge(left=log_return_df, right=spy_log_return_df, on='Date')
merged_df = merged_df.loc[merged_df['Date'].between('2018-01-01', '2018-12-31'), :].copy()

log_price = np.log(merged_df['Adj Close'].values)
volume = merged_df['Volume'].values
market_log_return = merged_df['Market Log Return']
datetime = merged_df['Date'].values
N = len(log_price)
print(f'  N = {N} observations')

# ── Model 1: Roll's Bid-Ask Bounce ─────────────────────────────────────────

print('\nConstructing JAGS model (no market factor)...')
prior_sigma_c = 0.05
prior_a = 1e-12
prior_b = 1e-12

def construct_hasbrouck_model(log_price, volume, prior_sigma_c, prior_a, prior_b,
                              chains=4, threads=None):
    if not threads:
        threads = chains
    code = '''
    model {
        for (t in 1:length(volume)) {
            o[t] = ifelse(volume[t] == 0.0, 0, 1)
            l[t] ~ dbern(0.5)
            q[t] = o[t] * (2.0 * l[t] - 1.0)
        }
        for (t in 2:length(volume)) {
            p[t] ~ dnorm(p[t-1] + c * (q[t] - q[t-1]), tau)
        }
        c ~ dnorm(0.0, prior_sigma_c^(-2)) I(0, )
        tau ~ dgamma(prior_a, prior_b)
        sigma = 1.0/sqrt(tau)
    }
    '''
    jags_data = {
        'p': log_price, 'volume': volume,
        'prior_sigma_c': prior_sigma_c, 'prior_a': prior_a, 'prior_b': prior_b,
    }
    return pyjags.Model(code=code, data=jags_data, chains=chains, threads=threads)

jags_model = construct_hasbrouck_model(log_price, volume, prior_sigma_c, prior_a, prior_b)

parameter_names = ['c', 'sigma']
state_variable_names = ['q']
variables = parameter_names + state_variable_names

print('Burn-in (1000 iterations)...')
jags_model.sample(1000, vars=[])

print('Sampling (2000 iterations)...')
samples = jags_model.sample(iterations=2000, vars=variables)

print(f"  Posterior mean of c = {samples['c'].flatten().mean():.6f}")

print('Converting to ArviZ...')
idata = pyjags.from_pyjags(samples)
print(f'  idata type: {type(idata)}')
print(f'  idata: {idata}')

# ── Model 1: Diagnostics & Plots ──────────────────────────────────────────

print('\nPlotting trace (model 1)...')
try:
    az.plot_trace(idata, var_names=parameter_names,
                  figure_kwargs={"figsize": figsize})
    save_fig('m1_trace')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

print('Computing rhat (model 1)...')
try:
    rhat = az.rhat(idata, var_names=parameter_names)
    print(f'  rhat: {rhat}')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')

print('Computing ess (model 1)...')
try:
    ess = az.ess(idata, var_names=parameter_names)
    print(f'  ess: {ess}')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')

print('Plotting posterior (model 1)...')
try:
    az.plot_forest(idata, var_names=parameter_names,
                     figure_kwargs={"figsize": figsize})
    save_fig('m1_posterior')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

print('Plotting posterior mean of trade direction (model 1)...')
try:
    data = idata.posterior['q']
    data = data.mean(axis=1).T
    data_mean = data.mean(axis=1)
    plt.figure(figsize=figsize)
    plt.plot(datetime, data_mean, marker='.', linestyle='None')
    plt.title('Posterior Mean of Trade Direction')
    save_fig('m1_trade_direction')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

# ── Model 2: With Market Factor ───────────────────────────────────────────

print('\nConstructing JAGS model (with market factor)...')
prior_sigma_beta_m = 1.0

def construct_hasbrouck_model_with_market_factor(
        log_price, volume, market_log_return, prior_sigma_c, prior_a, prior_b):
    code = '''
    model {
        for (t in 1:length(volume)) {
            o[t] = ifelse(volume[t] == 0.0, 0, 1)
            l[t] ~ dbern(0.5)
            q[t] = o[t] * (2.0 * l[t] - 1.0)
        }
        for (t in 2:length(volume)) {
            p[t] ~ dnorm(p[t-1] + c * (q[t] - q[t-1]) + beta_m * market_log_return[t], tau)
        }
        beta_m ~ dnorm(1.0, prior_sigma_beta_m^(-2))
        c ~ dnorm(0.0, prior_sigma_c^(-2)) I(0, )
        tau ~ dgamma(prior_a, prior_b)
        sigma = 1.0/sqrt(tau)
    }
    '''
    jags_data = {
        'p': log_price, 'volume': volume, 'market_log_return': market_log_return,
        'prior_sigma_c': prior_sigma_c, 'prior_a': prior_a, 'prior_b': prior_b,
        'prior_sigma_beta_m': prior_sigma_beta_m,
    }
    return pyjags.Model(code=code, data=jags_data, chains=4, threads=4)

jags_model_mf = construct_hasbrouck_model_with_market_factor(
    log_price, volume, market_log_return, prior_sigma_c, prior_a, prior_b)

parameter_names_mf = ['c', 'sigma', 'beta_m']
variables_mf = parameter_names_mf + state_variable_names

print('Burn-in (1000 iterations)...')
jags_model_mf.sample(1000, vars=[])

print('Sampling (2000 iterations)...')
samples_mf = jags_model_mf.sample(iterations=2000, vars=variables_mf)

print(f"  Posterior mean of c = {samples_mf['c'].flatten().mean():.6f}")
print(f"  Posterior mean of beta_m = {samples_mf['beta_m'].flatten().mean():.6f}")

print('Converting to ArviZ...')
idata_mf = pyjags.from_pyjags(samples_mf)

# ── Model 2: Diagnostics & Plots ──────────────────────────────────────────

print('\nPlotting trace (model 2)...')
try:
    az.plot_trace(idata_mf, var_names=parameter_names_mf,
                  figure_kwargs={"figsize": figsize})
    save_fig('m2_trace')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

print('Computing rhat (model 2)...')
try:
    rhat_mf = az.rhat(idata_mf, var_names=parameter_names_mf)
    print(f'  rhat: {rhat_mf}')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')

print('Computing ess (model 2)...')
try:
    ess_mf = az.ess(idata_mf, var_names=parameter_names_mf)
    print(f'  ess: {ess_mf}')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')

print('Plotting posterior (model 2)...')
try:
    az.plot_forest(idata_mf, var_names=parameter_names_mf,
                     figure_kwargs={"figsize": figsize})
    save_fig('m2_posterior')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

print('Plotting posterior mean of trade direction (model 2)...')
try:
    data = idata_mf.posterior['q']
    data = data.mean(axis=1).T
    data_mean = data.mean(axis=1)
    plt.figure(figsize=figsize)
    plt.plot(datetime, data_mean, marker='.', linestyle='None')
    plt.title('Posterior Mean of Trade Direction (with Market Factor)')
    save_fig('m2_trade_direction')
except Exception as e:
    print(f'  FAILED: {type(e).__name__}: {e}')
    plt.close('all')

print('\nDone. Check scripts/output/ for plots.')