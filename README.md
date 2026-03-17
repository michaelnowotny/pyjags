<p align="center">
  <img src="https://raw.githubusercontent.com/michaelnowotny/pyjags/master/logo.jpg" alt="PyJAGS" width="400">
</p>

<p align="center">
  <a href="https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml"><img src="https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/pyjags/"><img src="https://img.shields.io/pypi/v/pyjags" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyjags/"><img src="https://img.shields.io/pypi/pyversions/pyjags" alt="Python"></a>
  <a href="https://github.com/michaelnowotny/pyjags/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/pyjags" alt="License"></a>
  <a href="https://codecov.io/gh/michaelnowotny/pyjags"><img src="https://codecov.io/gh/michaelnowotny/pyjags/graph/badge.svg" alt="codecov"></a>
</p>

PyJAGS provides a Python interface to [JAGS](https://mcmc-jags.sourceforge.io/)
(Just Another Gibbs Sampler), a mature engine for Bayesian analysis via Markov
Chain Monte Carlo (MCMC) simulation.

## Why PyJAGS?

**No compilation step.** JAGS interprets models at runtime. Change your model
code and re-run immediately, with no C++ compiler toolchain required. Even
state-of-the-art HMC packages can take 30-60+ seconds to compile each model;
JAGS models are ready in milliseconds.

**Handles models that gradient-based samplers cannot.** JAGS works with
non-differentiable likelihoods, discrete parameters, mixture models, and
change-point models that challenge HMC/NUTS. No divergence diagnostics to fight.

**Incremental sampling is built in.** JAGS retains chain state between
`sample()` calls, so extending a run is a single line of code -- a capability
that is unique among Python Bayesian packages.

**Familiar model syntax.** JAGS uses the BUGS language, the lingua franca of
Bayesian modeling in the R ecosystem (WinBUGS, OpenBUGS, nimble). R users
migrating to Python can use their existing models unchanged.

**Lightweight.** PyJAGS depends only on numpy, arviz, and h5py. Install with
`pip install pyjags` and pre-built wheels for Linux and macOS.

**Information-theoretic diagnostics.** With the optional
[Divergence](https://github.com/michaelnowotny/divergence) package, PyJAGS
offers diagnostics that go beyond R-hat: information gain (how much did the
data teach us?), chain divergence (are my chains truly sampling the same
distribution?), Bayesian surprise (which observations are outliers?), and
prior sensitivity analysis -- all from a single `InferenceData` object.

## Quick Start

```python
import pyjags
import arviz as az

# Define a model in the BUGS language
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

# Fit the model
model = pyjags.Model(code=model_code, data=dict(y=y, N=len(y)),
                     chains=4, adapt=1000, seed=42)
model.sample(1000, vars=[])                         # burn-in
samples = model.sample(5000, vars=["mu", "sigma"])   # production

# Analyze with ArviZ
idata = pyjags.from_pyjags(samples)
az.summary(idata)
az.plot_trace(idata)
```

## Features

### Core Sampling
* Multicore support for parallel simulation of multiple Markov chains
* Reproducible sampling via a single `seed` parameter (`numpy.random.SeedSequence`)
* Generator-based sampling with `iter_sample()` for live convergence monitoring
* Incremental sampling with `sample_more()` and automatic convergence detection
* Standalone model syntax validation with `check_model()`
* `pathlib.Path` support for model files

### ArviZ Integration
* `pyjags.from_pyjags()` converts samples to ArviZ `InferenceData` with
  observed data, constant data, prior samples, and log-likelihood groups
* `pyjags.loo()` and `pyjags.compare()` for model comparison via PSIS-LOO
* `pyjags.summary()` for quick posterior summaries
* `pyjags.dic_samples()` for Deviance Information Criterion

### Advanced Diagnostics (optional, via [Divergence](https://github.com/michaelnowotny/divergence))

Install with `pip install pyjags[diagnostics]` to unlock:

* `pyjags.diagnostics.information_gain()` -- KL divergence from prior to posterior:
  how much did each parameter learn from the data?
* `pyjags.diagnostics.convergence_report()` -- R-hat + ESS + pairwise chain
  energy distance in one call
* `pyjags.diagnostics.bayesian_surprise()` -- per-observation surprise scores
  for outlier and influence detection
* `pyjags.diagnostics.model_divergence()` -- distributional comparison between
  two models' posterior predictives
* `pyjags.diagnostics.prior_sensitivity()` -- quantify how much your conclusions
  depend on the prior
* `pyjags.diagnostics.uncertainty_decomposition()` -- separate aleatoric
  (irreducible noise) from epistemic (parameter uncertainty)

```python
from pyjags.diagnostics import information_gain, convergence_report

idata = pyjags.from_pyjags(posterior_samples, prior=prior_samples)

# How much did the data teach us?
ig = information_gain(idata)
# {'mu': 3.98, 'sigma': 4.59}  -- substantial learning on both parameters

# Comprehensive convergence check
report = convergence_report(idata)
print(f"Converged: {report['converged']}, max R-hat: {report['max_rhat']:.4f}")
```

### Persistence and Utilities
* Saving and restoring MCMC sample chains to/from HDF5 files
* Merging samples along iterations or across chains
* PEP 561 `py.typed` marker for IDE and type checker support

License: GPLv2

## Who Is PyJAGS For?

**Coming from R?** If you have used JAGS, WinBUGS, or OpenBUGS in R, your
BUGS model files work unchanged in PyJAGS. You get the same sampler with
Python's data science ecosystem (numpy, pandas, matplotlib, ArviZ).

**New to Bayesian statistics?** The [Getting Started](notebooks/Getting%20Started.ipynb)
notebook walks you through your first Bayesian model in minutes. PyJAGS's
model-in-a-string approach is the gentlest on-ramp to MCMC: write the
probability model, pass your data, sample.

**Bayesian veteran?** PyJAGS complements HMC-based tools. Use it for models
with discrete parameters, mixture components, and change-points where JAGS's
Gibbs sampler has a structural advantage. The Divergence integration gives you
information-theoretic diagnostics that no other package provides.

**Educator?** We are building a comprehensive Bayesian statistics curriculum
as Jupyter notebooks, powered by PyJAGS and enriched with historical narrative
and information-theoretic analysis via Divergence. See the roadmap below.

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.12+ |
| NumPy | 1.x and 2.x |
| ArviZ | 1.0+ |
| macOS | Intel and Apple Silicon (M1/M2/M3/M4) |
| Linux | Debian/Ubuntu (tested), other distributions (untested) |

> **Note:** Python 3.10 and 3.11 were supported in earlier releases but are no longer
> supported because ArviZ 1.0 -- a core dependency -- requires Python 3.12+.

## Installation

### Prerequisites

A working JAGS installation, a C++ compiler, and CMake are required. PyJAGS uses
CMake with `pkg-config` to locate the JAGS headers and libraries at build time.

### macOS

#### 1. Install JAGS via Homebrew

**Apple Silicon (M1/M2/M3/M4):**

Homebrew installs to `/opt/homebrew` on Apple Silicon Macs:

```bash
# Install Homebrew if needed (https://brew.sh)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Follow Homebrew's instructions to add it to your PATH, then:
brew install jags
```

Make sure `pkg-config` can find the JAGS installation by adding this to your
shell profile (`~/.zprofile` or `~/.zshrc`):

```bash
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
```

**Intel Mac:**

Homebrew installs to `/usr/local` on Intel Macs and `pkg-config` typically
finds JAGS automatically:

```bash
brew install jags
```

#### 2. Set up Python and install PyJAGS

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python
installations and virtual environments:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 and create a virtual environment
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install PyJAGS
uv pip install pyjags

# Optional: install advanced diagnostics (Divergence integration)
uv pip install pyjags[diagnostics]
```

To install from source (for development):

```bash
source .venv/bin/activate

git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags

uv pip install -e .
```

### Linux

Install JAGS using your distribution's package manager:

```bash
# Debian/Ubuntu
sudo apt-get install jags pkg-config

# Fedora/RHEL (untested -- package names may differ)
sudo dnf install jags jags-devel pkgconf
```

Then install PyJAGS:

```bash
pip install pyjags
```

Or from source:

```bash
git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags

pip install -e .
```

### Windows

PyJAGS is not natively supported on Windows. Windows users should use
[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows
Subsystem for Linux) with an Ubuntu distribution, then follow the Linux
installation instructions above.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating PyJAGS features:

| Notebook | Description |
|----------|-------------|
| [Getting Started](notebooks/Getting%20Started.ipynb) | Beginner-friendly introduction to Bayesian inference with PyJAGS |
| [New in v2.3.0](notebooks/New%20in%20v2.3.0.ipynb) | Showcase of all v2.3.0 features: `iter_sample()`, `seed=`, `check_model()`, and more |
| [Trading Cost Estimation](notebooks/Trading%20Cost%20Estimation.ipynb) | Bayesian estimation of bid-ask spreads using Hasbrouck's model (with and without a market factor) |
| [Logistic Regression](notebooks/Logistic%20Regression.ipynb) | Bayesian logistic regression with MCMC diagnostics |
| [Eight Schools](notebooks/Eight%20Schools.ipynb) | Classic hierarchical model with prior/posterior analysis, warmup splitting, and LOO |
| [Advanced Functionality](notebooks/Advanced%20Functionality.ipynb) | Parallel chains, HDF5 persistence, chain merging, and incremental sampling until convergence |

If you have a native installation (macOS or Linux with JAGS and `.venv` set up as
described above), you can run Jupyter Lab directly without Docker:

```bash
./scripts/jagslab lab
```

On first run, this installs notebook dependencies (JupyterLab, seaborn, scikit-learn)
into your `.venv` and registers a **"Python 3.12 (pyjags)"** Jupyter kernel that
points to the correct Python environment. The notebooks are pre-configured to use
this kernel.

## ArviZ Integration

PyJAGS includes a built-in converter for [ArviZ](https://www.arviz.org/) 1.0+.
Use `pyjags.from_pyjags()` to convert sample dictionaries returned by
`Model.sample()` into ArviZ `InferenceData` objects for diagnostics and visualization:

```python
import pyjags
import arviz as az

model = pyjags.Model(code=model_code, data=data, chains=4)
model.sample(1000, vars=[])                     # burn-in
samples = model.sample(5000, vars=['mu', 'sigma'])

idata = pyjags.from_pyjags(
    samples,
    prior=prior_samples,                        # optional
    observed_data={"y": y},                     # optional
    constant_data={"N": np.array(len(y))},      # optional
)

az.summary(idata)
az.plot_trace(idata)
```

The converter supports prior samples, log-likelihood extraction, observed and
constant data groups, warmup splitting, and automatic metadata attributes.
See the [Eight Schools](notebooks/Eight%20Schools.ipynb) notebook for a
complete example.

## Ecosystem

PyJAGS is part of a growing ecosystem of tools for Bayesian analysis in Python:

* **[ArviZ](https://www.arviz.org/)** -- diagnostics and visualization (built in)
* **[Divergence](https://github.com/michaelnowotny/divergence)** -- information-theoretic
  measures: KL divergence, Jensen-Shannon, Hellinger, total variation, MMD,
  Wasserstein, Renyi, and more. The `pyjags.diagnostics` module provides a seamless
  bridge between PyJAGS and Divergence for Bayesian-specific analyses.

## Development Environment (jagslab)

PyJAGS also includes a Docker-based development environment managed by the `jagslab`
CLI script. This provides a reproducible setup with JAGS, Python 3.12, and Jupyter Lab
without requiring a local JAGS installation.

### Quick Start (Docker)

```bash
# 1. Copy the environment configuration
cp .env.example .env          # Edit .env to customize (e.g., Jupyter port)

# 2. Build the Docker image
./scripts/jagslab build

# 3. Run the test suite
./scripts/jagslab test

# 4. Start Jupyter Lab
./scripts/jagslab start       # Opens at http://localhost:8888
```

### jagslab Commands

| Command | Description |
|---------|-------------|
| `start` | Start Jupyter Lab via Docker (auto-starts container and installs pyjags) |
| `lab` | Start Jupyter Lab natively from `.venv` (no Docker required) |
| `stop` | Stop the Docker container |
| `test [args]` | Run the test suite |
| `install` | Install/reinstall pyjags (recompiles C++ extension) |
| `shell` | Open a bash shell in the container |
| `python [args]` | Start Python REPL or run a script in the container |
| `build` | Build the Docker image |
| `rebuild` | Rebuild image from scratch (no Docker cache) |
| `status` | Show container status |
| `logs` | Tail container logs |
| `clean` | Remove build artifacts |
| `version` | Show pyjags, JAGS, Python, and numpy versions |

### Running Tests

```bash
./scripts/jagslab test                                           # All tests
./scripts/jagslab test test.test_model                           # One module
./scripts/jagslab test test.test_model.TestModel.test_samples_shape  # Single test
```

### Pre-commit Hooks

PyJAGS uses [pre-commit](https://pre-commit.com/) to run ruff linting and formatting
checks before each commit, preventing CI failures:

```bash
pip install pre-commit
pre-commit install
```

After installation, ruff will automatically check and fix files in `src/` and `test/`
on every `git commit`. To run manually against all files:

```bash
pre-commit run --all-files
```

### Working with the C++ Extension

PyJAGS includes a C++ extension (`src/pyjags/console.cc`) that wraps the JAGS library
using pybind11. When editing Python files under `src/pyjags/`, changes take effect
immediately thanks to the editable install. However, **after editing
`src/pyjags/console.cc`, you must recompile**:

```bash
./scripts/jagslab install
```

### Configuration

The `.env` file (created from `.env.example`) controls environment settings:

- `JAGSLAB_PORT` -- Host port for Jupyter Lab (default: `8888`)

If `.env` does not exist when you run a `jagslab` command, it is automatically
created from `.env.example`.

## Troubleshooting

### macOS: `symbol not found '_JAGS_NA'` or missing JAGS symbols at runtime

This usually means the compiled extension was not linked against `libjags`. Verify
that `pkg-config` finds your JAGS installation:

```bash
pkg-config --libs --cflags jags
```

If this fails or points to the wrong location, set `PKG_CONFIG_PATH` as described
in the installation instructions above, then reinstall PyJAGS.

### macOS: `found architecture 'x86_64', required architecture 'arm64'`

This occurs on Apple Silicon Macs when JAGS was installed via an Intel (Rosetta)
Homebrew at `/usr/local` but Python is running natively as ARM64. The fix is to
install JAGS using the native ARM Homebrew at `/opt/homebrew`:

```bash
# Check which Homebrew you are using
which brew
# /opt/homebrew/bin/brew  -> ARM (correct for Apple Silicon)
# /usr/local/bin/brew     -> Intel/Rosetta (will cause architecture mismatch)

# If needed, install ARM Homebrew and then:
/opt/homebrew/bin/brew install jags
```

Make sure `PKG_CONFIG_PATH` points to the ARM Homebrew pkgconfig directory
(`/opt/homebrew/lib/pkgconfig`) so that the build picks up the correct library.

### CMake errors during build

PyJAGS uses CMake (via scikit-build-core) to find and link against JAGS. If
the build fails with CMake errors, ensure CMake is installed:

```bash
# macOS
brew install cmake

# Debian/Ubuntu
sudo apt-get install cmake

# pip
pip install cmake
```

## Useful Links
* Package on the Python Package Index <https://pypi.python.org/pypi/pyjags>
* Project page on GitHub <https://github.com/michaelnowotny/pyjags>
* JAGS manual and examples <http://sourceforge.net/projects/mcmc-jags/files/>
* Divergence package <https://github.com/michaelnowotny/divergence>


## Acknowledgements

* JAGS was created by Martyn Plummer
* PyJAGS was originally created by Tomasz Miasko
* As of May 2020, PyJAGS is developed by Michael Nowotny
* [Max Schulist](https://github.com/mschulist) ([PR #1](https://github.com/michaelnowotny/pyjags/pull/1))
  proposed migrating to scikit-build-core, pyproject.toml, and pytest, ideas that
  inspired the packaging modernization in version 2.1
* [Scout Jarman](https://github.com/scoutiii) ([PR #2](https://github.com/michaelnowotny/pyjags/pull/2))
  independently proposed CI/CD workflows, CMake builds, and JAGS install scripts,
  reinforcing the direction of the modernization effort
