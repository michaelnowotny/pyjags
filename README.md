# PyJAGS: The Python Interface to JAGS

[![CI](https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml/badge.svg)](https://github.com/michaelnowotny/pyjags/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pyjags)](https://pypi.org/project/pyjags/)
[![Python](https://img.shields.io/pypi/pyversions/pyjags)](https://pypi.org/project/pyjags/)
[![License](https://img.shields.io/pypi/l/pyjags)](https://github.com/michaelnowotny/pyjags/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/michaelnowotny/pyjags/graph/badge.svg)](https://codecov.io/gh/michaelnowotny/pyjags)

PyJAGS provides a Python interface to JAGS, a program for analysis of Bayesian
hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

PyJAGS adds the following features on top of JAGS:

* Multicore support for parallel simulation of multiple Markov chains
* Built-in ArviZ integration via `pyjags.from_pyjags()` for diagnostics and visualization
* Incremental sampling with automatic convergence detection (ESS and R-hat criteria)
* Saving and restoring MCMC sample chains to/from HDF5 files
* Merging samples along iterations or across chains for resumed sampling

License: GPLv2

## Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.12+ |
| NumPy | 1.x and 2.x |
| ArviZ | 1.0+ |
| macOS | Intel and Apple Silicon (M1/M2/M3/M4) |
| Linux | Debian/Ubuntu (tested), other distributions (untested) |

> **Note:** Python 3.10 and 3.11 were supported in earlier releases but are no longer
> supported because ArviZ 1.0 — a core dependency — requires Python 3.12+.

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

# Fedora/RHEL (untested — package names may differ)
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
`Model.sample()` into ArviZ `DataTree` objects for diagnostics and visualization:

```python
import pyjags
import arviz as az

model = pyjags.Model(code=model_code, data=data, chains=4)
model.sample(1000, vars=[])                     # burn-in
samples = model.sample(5000, vars=['mu', 'sigma'])

idata = pyjags.from_pyjags(samples)             # -> xarray.DataTree
az.summary(idata)
az.plot_trace(idata)
```

The converter also supports prior samples, log-likelihood extraction, and warmup
splitting. See the [Eight Schools](notebooks/Eight%20Schools.ipynb) notebook for a
complete example.

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

- `JAGSLAB_PORT` — Host port for Jupyter Lab (default: `8888`)

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
* Project page on github <https://github.com/michaelnowotny/pyjags>
* JAGS manual and examples <http://sourceforge.net/projects/mcmc-jags/files/>


## Acknowledgements

* JAGS was created by Martyn Plummer
* PyJAGS was originally created by Tomasz Miasko
* As of May 2020, PyJAGS is developed by Michael Nowotny
* [Max Schulist](https://github.com/mschulist) ([PR #1](https://github.com/michaelnowotny/pyjags/pull/1))
  proposed migrating to scikit-build-core, pyproject.toml, and pytest — ideas that
  inspired the packaging modernization in version 2.1