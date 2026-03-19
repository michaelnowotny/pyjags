# Quick Start

## Your First Bayesian Model

```python
import pyjags
import numpy as np
import arviz as az

# Some data
np.random.seed(42)
y = np.random.normal(loc=5.0, scale=2.0, size=30)

# Write the model in the BUGS language
model_code = """
model {
    mu ~ dnorm(0, 0.001)
    sigma ~ dunif(0, 100)
    tau <- pow(sigma, -2)
    for (i in 1:N) {
        y[i] ~ dnorm(mu, tau)
    }
}
"""

# Compile, adapt, and sample
model = pyjags.Model(
    code=model_code,
    data=dict(y=y, N=len(y)),
    chains=4,
    adapt=1000,
    seed=42,
)

model.sample(1000, vars=[])                       # burn-in
samples = model.sample(5000, vars=["mu", "sigma"]) # production

# Analyze with ArviZ
idata = pyjags.from_pyjags(samples)
az.summary(idata)
```

## What Just Happened?

1. **Model specification**: We wrote a probabilistic model in the BUGS
   language -- priors on `mu` and `sigma`, and a normal likelihood for
   each data point.

2. **Compilation**: `pyjags.Model()` compiled the model, loaded the data,
   and ran 1,000 adaptation steps where JAGS tunes its samplers.

3. **Burn-in**: We ran 1,000 iterations without monitoring any variables,
   letting the chains move to the high-probability region.

4. **Production**: We drew 5,000 samples from each of 4 chains (20,000
   total), monitoring `mu` and `sigma`.

5. **Analysis**: We converted to ArviZ and computed summary statistics
   including posterior means, credible intervals, ESS, and R-hat.

## Next Steps

- Work through [The Reverend's Question](../notebooks/Getting Started.ipynb)
  for a narrative introduction
- Explore the [API Reference](../api/index.md) for the complete API
- Install `pip install pyjags[diagnostics]` for information-theoretic
  diagnostics via [Divergence](https://github.com/michaelnowotny/divergence)
