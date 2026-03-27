<p align="center">
  <img src="https://raw.githubusercontent.com/michaelnowotny/pyjags/master/logo.jpg" alt="PyJAGS" width="400">
</p>

<h3 align="center"><em>From prior to posterior</em></h3>

<p align="center">
  <a href="https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml"><img src="https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/pyjags/"><img src="https://img.shields.io/pypi/v/pyjags" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyjags/"><img src="https://img.shields.io/pypi/pyversions/pyjags" alt="Python"></a>
  <a href="https://github.com/michaelnowotny/pyjags/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/pyjags" alt="License"></a>
  <a href="https://codecov.io/gh/michaelnowotny/pyjags"><img src="https://codecov.io/gh/michaelnowotny/pyjags/graph/badge.svg" alt="codecov"></a>
</p>

<p align="center">
PyJAGS brings <a href="https://mcmc-jags.sourceforge.io/">JAGS</a>
(Just Another Gibbs Sampler) to Python -- a mature, battle-tested engine
for Bayesian inference via Markov Chain Monte Carlo, trusted by
researchers in statistics, ecology, epidemiology, and finance for over
two decades.
</p>

## Why PyJAGS?

**Instant models, no compilation.** JAGS interprets the BUGS model
language at runtime. Change a prior, re-run immediately. No C++ compiler,
no minutes-long compilation step. From idea to posterior in seconds.

**Models that gradient-based samplers cannot touch.** Discrete parameters,
mixture components, change-point models, latent class indicators -- JAGS
samples them directly through Gibbs sampling. No marginalization tricks,
no relaxation hacks, no divergence warnings.

**Incremental sampling.** JAGS retains chain state between `sample()`
calls. Extending a run is a single line of code. Save your samples to
HDF5, shut down, come back tomorrow, pick up exactly where you left off.
No other Python Bayesian package makes this so natural.

**The BUGS language.** If you've used JAGS, WinBUGS, or OpenBUGS in R,
your model files work unchanged. Decades of published BUGS models from
textbooks, papers, and the vast BUGS/JAGS ecosystem are immediately
available.

**Information-theoretic diagnostics.** With the optional
[Divergence](https://github.com/michaelnowotny/divergence) package,
PyJAGS offers diagnostics that go beyond R-hat and ESS: *information gain*
(how much did the data teach us about each parameter?), *chain divergence*
(are my chains truly sampling the same distribution?), *Bayesian surprise*
(which observations are outliers?), and *prior sensitivity analysis* --
all computed directly from ArviZ inference data.

**Lightweight and modern.** Three core dependencies (numpy, arviz, h5py).
Pre-built wheels for Linux and macOS. PEP 561 type stubs. 100% docstring
coverage. 250 tests. CI on Python 3.12, 3.13, and 3.14.

## Quick Start

```python
import pyjags
import arviz as az

model_code = """
model {
    mu ~ dnorm(0, 0.001)
    sigma ~ dunif(0, 100)
    tau <- pow(sigma, -2)
    for (i in 1:N) { y[i] ~ dnorm(mu, tau) }
}
"""

model = pyjags.Model(code=model_code, data=dict(y=y, N=len(y)),
                     chains=4, adapt=1000, seed=42)
model.sample(1000, vars=[])                         # burn-in
samples = model.sample(5000, vars=["mu", "sigma"])   # production

idata = pyjags.from_pyjags(samples)
az.summary(idata)
```

## Tutorials

The `notebooks/` directory contains a series of tutorials that teach
Bayesian inference through compelling real-world stories. Each notebook
weaves historical context and narrative alongside working PyJAGS code.

| Notebook | The Story |
|----------|-----------|
| [**The Reverend's Question**](notebooks/Getting%20Started.ipynb) | Begin where it all began: Thomas Bayes' posthumous theorem (1763). Estimate a star's position, watch beliefs sharpen with evidence, and discover why Bayesian inference gives you not just answers but *honest uncertainty*. |
| [**The Paradox of Shrinkage**](notebooks/Eight%20Schools.ipynb) | In 1955, Charles Stein proved something nobody believed: estimating things separately is *always* worse than borrowing strength across groups. See the paradox come alive in the classic Eight Schools dataset. |
| [**The Wells of Bangladesh**](notebooks/Logistic%20Regression.ipynb) | Three thousand families discovered their drinking water was poisoned. Their decision to switch wells -- driven by distance, arsenic levels, and education -- becomes a lesson in logistic regression with real human stakes. |
| [**The Hidden Cost of Every Trade**](notebooks/Trading%20Cost%20Estimation.ipynb) | Every stock trade carries a hidden cost that most investors never see. Use a latent variable model to reveal the bid-ask spread from daily prices -- and discover why JAGS's ability to sample discrete parameters is a structural advantage. |
| [**When It Rains, It Pours**](notebooks/Advanced%20Functionality.ipynb) | The 2008 financial crisis through the lens of stochastic volatility. Build a model with 1,260 latent variables, learn parallel sampling, HDF5 persistence, chain merging, and automatic convergence -- the full practitioner's toolkit. |

## Features

### Core Sampling
* Parallel chains across CPU cores (3-4x speedup)
* Reproducible sampling via `seed` (`numpy.random.SeedSequence`)
* Generator-based sampling with `iter_sample()` for live monitoring
* Incremental sampling with `sample_more()` and `sample_until()`
* Model syntax validation with `check_model()`

### ArviZ Integration
* `from_pyjags()` converts to ArviZ `InferenceData` with prior,
  observed data, constant data, log-likelihood, and warmup groups
* `loo()` and `compare()` for model comparison via PSIS-LOO
* `summary()` and `dic_samples()` for quick diagnostics

### Advanced Diagnostics (via [Divergence](https://github.com/michaelnowotny/divergence))

Install with `pip install pyjags[diagnostics]`:

```python
from pyjags.diagnostics import information_gain, convergence_report

idata = pyjags.from_pyjags(posterior, prior=prior_samples)

ig = information_gain(idata)        # KL divergence: prior -> posterior
report = convergence_report(idata)  # R-hat + ESS + chain energy distance
```

Also available: `bayesian_surprise()`, `model_divergence()`,
`prior_sensitivity()`, `uncertainty_decomposition()`.

### Persistence
* Save and load MCMC samples via HDF5
* Merge samples along iterations or across chains
* PEP 561 `py.typed` and `console.pyi` type stubs

## Who Is PyJAGS For?

**Coming from R?** Your BUGS model files work unchanged. You get the
same JAGS engine with Python's data science ecosystem.

**New to Bayesian statistics?** Start with
[The Reverend's Question](notebooks/Getting%20Started.ipynb) -- your
first posterior in minutes, with historical context that makes the
ideas unforgettable.

**Bayesian veteran?** PyJAGS complements HMC-based tools for models
with discrete parameters, mixture components, and change-points. The
Divergence integration gives you information-theoretic diagnostics
that no other package provides.

## Compatibility

| Component | Supported |
|-----------|-----------|
| Python | 3.12+ |
| NumPy | 1.x and 2.x |
| ArviZ | 1.0+ |
| macOS | Intel and Apple Silicon |
| Linux | Debian/Ubuntu (tested), others (untested) |

## Getting Started

### Option 1: Docker (fastest, any platform)

If you have Docker installed, PyJAGS includes a ready-to-run development
environment with JAGS, Python, and Jupyter Lab pre-configured. No native
JAGS installation needed:

```bash
git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags
cp .env.example .env
./scripts/jagslab build   # build Docker image with JAGS + Python
./scripts/jagslab start   # launch Jupyter Lab at http://localhost:8888
```

The `jagslab` CLI manages everything:

| Command | What it does |
|---------|-------------|
| `./scripts/jagslab start` | Start Jupyter Lab (Docker) |
| `./scripts/jagslab test` | Run the full test suite (250 tests) |
| `./scripts/jagslab shell` | Open a bash shell in the container |
| `./scripts/jagslab lab` | Start Jupyter Lab natively (no Docker) |

### Option 2: Native Installation (macOS / Linux)

For native installation, you need JAGS (the C++ engine) installed on
your system and a properly configured Python environment.

We recommend [uv](https://docs.astral.sh/uv/) for managing Python
installations and virtual environments:

**macOS (Apple Silicon):**

```bash
# 1. Install JAGS via Homebrew
brew install jags

# 2. Set PKG_CONFIG_PATH (add to ~/.zprofile or ~/.zshrc)
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

# 3. Set up Python environment with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 4. Install PyJAGS
uv pip install pyjags

# 5. Optional: advanced diagnostics via Divergence
uv pip install pyjags[diagnostics]
```

**macOS (Intel):**

```bash
brew install jags
pip install pyjags
```

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install jags pkg-config
pip install pyjags
```

**Windows:** Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
with Ubuntu, then follow the Linux instructions.

### From Source (development)

```bash
git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags
pip install -e ".[dev]"
pre-commit install
```

## Ecosystem

PyJAGS is part of a growing ecosystem for Bayesian analysis in Python:

* **[ArviZ](https://www.arviz.org/)** -- the standard library for
  Bayesian diagnostics and visualization. PyJAGS integrates natively
  via `from_pyjags()`.

* **[Divergence](https://github.com/michaelnowotny/divergence)** --
  a comprehensive information-theoretic toolkit providing 40+ measures
  of entropy, divergence, and distance (KL, Jensen-Shannon, Hellinger,
  total variation, chi-squared, Renyi, MMD, Wasserstein, energy distance,
  transfer entropy, Fisher divergence, kernel Stein discrepancy, and
  more). Through the `pyjags.diagnostics` module, Divergence provides
  Bayesian-specific analyses that are **unique in the Python ecosystem**:

  - **Information gain**: how many nats of information did the data
    contribute to each parameter? No other package answers this question.
  - **Chain divergence**: pairwise distributional comparison between
    MCMC chains using energy distance or MMD -- catches problems that
    R-hat misses because it compares entire distributions, not just
    moments.
  - **Bayesian surprise**: per-observation surprise scores that identify
    influential data points and outliers.
  - **Prior sensitivity**: quantify how much your conclusions depend on
    your prior choice, using any of 7 divergence measures.
  - **Uncertainty decomposition**: separate aleatoric (irreducible noise)
    from epistemic (parameter uncertainty) in your predictions.

  Install with `pip install pyjags[diagnostics]`.

  **GPU acceleration**: Divergence's pairwise distance computations
  (energy distance, MMD, kernel Stein discrepancy) and permutation
  tests can run on NVIDIA GPUs via JAX for orders-of-magnitude speedup
  on large MCMC outputs. To enable GPU support:

  ```bash
  pip install pyjags[diagnostics] jax[cuda12]
  ```

  When a CUDA-capable GPU is available, Divergence automatically
  dispatches to GPU kernels. No code changes needed -- the same
  `chain_two_sample_test()` call that runs on CPU will use the GPU
  when available.

## Troubleshooting

<details>
<summary>macOS: symbol not found '_JAGS_NA'</summary>

Verify `pkg-config --libs --cflags jags` succeeds. If not, set
`PKG_CONFIG_PATH` as shown in the installation instructions.
</details>

<details>
<summary>macOS: architecture mismatch (x86_64 vs arm64)</summary>

Install JAGS via the ARM Homebrew at `/opt/homebrew`:
```bash
/opt/homebrew/bin/brew install jags
```
</details>

<details>
<summary>CMake errors during build</summary>

```bash
brew install cmake    # macOS
sudo apt install cmake  # Linux
pip install cmake     # anywhere
```
</details>

## Links

* [PyPI](https://pypi.org/project/pyjags/)
* [GitHub](https://github.com/michaelnowotny/pyjags)
* [JAGS manual](https://sourceforge.net/projects/mcmc-jags/files/)
* [Divergence](https://github.com/michaelnowotny/divergence)
* [ArviZ](https://www.arviz.org/)

## Acknowledgements

* **JAGS** was created by [Martyn Plummer](https://mcmc-jags.sourceforge.io/)
* **PyJAGS** was originally created by [Tomasz Miasko](https://github.com/tmiasko)
* As of 2020, PyJAGS is developed by [Michael Nowotny](https://github.com/michaelnowotny)
* [Max Schulist](https://github.com/mschulist) and
  [Scout Jarman](https://github.com/scoutiii) contributed ideas that
  inspired the packaging modernization

License: GPLv2