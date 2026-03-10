# PyJAGS: The Python Interface to JAGS
PyJAGS provides a Python interface to JAGS, a program for analysis of Bayesian
hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

PyJAGS adds the following features on top of JAGS:

* Multicore support for parallel simulation of multiple Markov chains (See Jupyter Notebook [Advanced Functionality](notebooks/Advanced%20Functionality.ipynb)
* Saving sample MCMC chains to and restoring from HDF5 files
* Functionality to merge samples along iterations or across chains so that sampling can be resumed in consecutive chunks until convergence criteria are satisfied
* Connectivity to the Bayesian analysis and visualization package Arviz

License: GPLv2

## Supported Platforms
PyJAGS works on MacOS and Linux. Windows is not currently supported.

## Installation
A working JAGS installation is required.

<pre>
    pip install pyjags
</pre>

## Development Environment (jagslab)

PyJAGS includes a Docker-based development environment managed by the `jagslab` CLI
script. This provides a reproducible setup with JAGS, Python 3.12, and Jupyter Lab
without requiring a local JAGS installation.

### Quick Start

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
| `start` | Start Jupyter Lab (auto-starts container and installs pyjags) |
| `stop` | Stop the container |
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

### Working with the C++ Extension

PyJAGS includes a C++ extension (`pyjags/console.cc`) that wraps the JAGS library
using pybind11. When editing Python files under `pyjags/`, changes take effect
immediately thanks to the editable install. However, **after editing
`pyjags/console.cc`, you must recompile**:

```bash
./scripts/jagslab install
```

### Configuration

The `.env` file (created from `.env.example`) controls environment settings:

- `JAGSLAB_PORT` — Host port for Jupyter Lab (default: `8888`)

If `.env` does not exist when you run a `jagslab` command, it is automatically
created from `.env.example`.

## Useful Links
* Package on the Python Package Index <https://pypi.python.org/pypi/pyjags>
* Project page on github <https://github.com/michaelnowotny/pyjags>
* JAGS manual and examples <http://sourceforge.net/projects/mcmc-jags/files/>


## Acknowledgements


* JAGS was created by Martyn Plummer
* PyJAGS was originally created by Tomasz Miasko
* As of May 2020, PyJAGS is developed by Michael Nowotny