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
./scripts/jagslab test test/test_model.py
./scripts/jagslab test test/test_model.py::TestModel::test_samples_shape
./scripts/jagslab test -k "test_scalar"

# Directly (inside container or with local JAGS)
python -m pytest test/ -v

# Run a single test file
python -m pytest test/test_chain_utilities.py -v

# Multi-Python version matrix
./scripts/test-all-pythons
```

Tests use pytest with Hypothesis for property-based testing. Dev dependencies: `pip install pytest hypothesis`. Test files are in `test/`. Hypothesis is configured with `dev` (10 examples) and `ci` (50 examples) profiles in `test/conftest.py`.

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

## Copyright and Licensing

PyJAGS is released under the GNU General Public License version 2 (GPLv2). The following rules are **mandatory** and must never be violated:

### Copyright Headers

Every source file (Python, C++, shell scripts, Dockerfiles) **must** carry a GPLv2 copyright header:

```
# Copyright (C) <year> <author>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
```

### Rules

1. **Never remove existing copyright notices.** The original author Tomasz Miasko (2015-2016) created PyJAGS. Michael Nowotny took over as maintainer in 2020. All prior copyright attributions must be preserved.
2. **When modifying an existing file**, add `Michael Nowotny` with the year of modification to the copyright header if not already present. Use the multi-line format:
   ```
   # Copyright (C) 2015-2016 Tomasz Miasko
   #               2020 Michael Nowotny
   ```
3. **New files** get `Copyright (C) <current_year> Michael Nowotny` with the full GPLv2 header.
4. **Use the correct year**: the year the code was first contributed, not the current year (unless the file is being created now). Check `git log --diff-filter=A -- <file>` when uncertain.
5. **GPLv2 Section 2a compliance**: modified files must carry prominent notices stating that you changed the files and the date of any change. The copyright header update satisfies this requirement.

### Copyright Holders

- **Tomasz Miasko** — original creator (2015-2016)
- **Michael Nowotny** — maintainer (2020-present)
- **Martyn Plummer** — author of JAGS (the upstream dependency, not part of this codebase)

## Version Management

Uses versioneer with git tags (no prefix) for automatic version inference.