# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyJAGS is a Python interface to JAGS (Just Another Gibbs Sampler) for Bayesian hierarchical model analysis via MCMC. It uses pybind11 to wrap the JAGS C++ Console class, providing a Pythonic API for model specification, compilation, and sampling.

## Development Environment (Docker)

Two-tier Docker architecture: `Dockerfile` (base — JAGS + build tools + core deps) and `Dockerfile.lab` (extends base — adds JupyterLab + notebook deps). The `jagslab` CLI manages both.

```bash
./scripts/jagslab build          # Build base + lab images
./scripts/jagslab test           # Run all tests
./scripts/jagslab start          # Start Jupyter Lab (port from .env, default 8889)
./scripts/jagslab install        # Recompile C++ extension after editing console.cc
./scripts/jagslab shell          # Bash shell in the container
```

## Build & Install

Requires JAGS library installed on the system. On Linux/macOS, `pkg-config` is used to locate JAGS. Inside the Docker container, `--no-build-isolation` is required because numpy must be importable at build time.

```bash
# Inside Docker container
pip install --no-build-isolation -e .

# Outside Docker (system JAGS required)
pip install -e .
```

The build compiles a single C++ extension (`pyjags/console.cc`) using pybind11 and C++14.

## Testing

```bash
# Via jagslab (recommended)
./scripts/jagslab test
./scripts/jagslab test test.test_model
./scripts/jagslab test test.test_model.TestModel.test_samples_shape

# Directly (inside container or with local JAGS)
python -m unittest discover test/
```

Tests use Python's built-in `unittest` framework. Test files are in `test/`.

## Architecture

### Core Components

- **`pyjags/console.cc`** — pybind11 C++ bindings wrapping the JAGS Console class. Handles data conversion between Python (numpy arrays) and JAGS (SArray). Exposes `checkModel`, `compile`, `initialize`, `update`, `sample`, monitor management, and RNG control.

- **`pyjags/model.py`** — High-level `Model` class (main public API). Manages model compilation, initialization, adaptation, and sampling. Contains `MultiConsole` which wraps multiple JAGS Console instances for parallel chain execution across threads.

- **`pyjags/chain_utilities.py`** — Utilities for MCMC chain manipulation: burn-in discarding, merging parallel/consecutive chains, extracting final iterations for re-initialization.

- **`pyjags/incremental_sampling.py`** — Convergence criteria (`EffectiveSampleSizeCriterion`, `RHatDeviationCriterion`) and iterative `sample_until()` for sampling until convergence.

- **`pyjags/modules.py`** — JAGS module discovery and loading from filesystem paths.

- **`pyjags/io.py`** — HDF5 persistence for sample dictionaries via `h5py`.

- **`pyjags/dic.py`** — Deviance Information Criterion calculation.

### Data Flow

User data (Python dicts with numpy arrays/scalars) → `dict_to_jags()` (ensures 1D+ numpy arrays) → C++ `to_jags()` (converts to JAGS SArray) → JAGS engine → C++ `to_python()` (numpy arrays) → `dict_from_jags()` (MaskedArrays for NA values)

### Sample Array Convention

Sample arrays returned by `Model.sample()` have shape `(*variable_dims, iterations, chains)`. For example, a 3x5 matrix variable sampled for 17 iterations across 7 chains yields shape `(3, 5, 17, 7)`.

### Threading Model

`MultiConsole` distributes chains across threads. Each thread gets its own JAGS Console instance(s). The `threads` and `chains_per_thread` parameters control parallelism. Chain numbering maps outer chain indices to per-console internal indices.

## Dependencies

- **Python 3.12+** — required (arviz 1.0 requires Python 3.12+)
- **numpy** — array handling and data conversion
- **arviz >= 1.0** — Bayesian analysis/visualization; used in `incremental_sampling.py` for ESS/Rhat and in `pyjags/arviz.py` for `from_pyjags()` converter
- **h5py** — HDF5 file I/O for sample persistence
- **JAGS** — external system library (must be installed separately)
- **pybind11** — included as git submodule in `pybind11/`

## Version Management

Uses versioneer with git tags (no prefix) for automatic version inference.