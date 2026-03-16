# Changelog

All notable changes to PyJAGS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.1] - 2026-03-16

### Added
- PyJAGS logo in README.
- Conda-forge CI workflow testing JAGS from conda-forge on Linux and macOS.
- Conda-forge recipe template (`conda-forge/meta.yaml`).
- Conda-aware JAGS module directory fallback (checks `$CONDA_PREFIX`).

### Changed
- Softened competitive language in README "Why PyJAGS?" section.
- Codecov upload now uses `CODECOV_TOKEN` secret.

## [2.3.0] - 2026-03-15

### Added
- `Model(seed=42)` parameter for reproducible sampling via
  `numpy.random.SeedSequence`, deriving independent per-chain RNG seeds.
- `model.iter_sample()` generator for chunked sampling with live
  convergence monitoring, yielding `SamplingState` objects with lazy ESS
  and R-hat diagnostics.
- `model.sample_more()` convenience method for extending existing samples
  by drawing additional iterations and concatenating.
- `Model.__repr__` showing chains, variables, iteration count, and
  adaptation status.
- `Model.samplers`, `Model.is_adapted`, and `Model.iteration` properties
  exposing JAGS sampler diagnostics.
- `pyjags.check_model()` standalone function for validating JAGS model
  syntax without compiling.
- `pyjags.loo()`, `pyjags.waic()`, and `pyjags.compare()` convenience
  wrappers around ArviZ functions accepting PyJAGS sample dictionaries.
- `observed_data` and `constant_data` parameters on `from_pyjags()` for
  attaching data to the ArviZ DataTree.
- Metadata attributes (`inference_library`, `inference_library_version`)
  automatically set on ArviZ DataTree objects.
- `warn_convergence` parameter on `Model.sample()` for opt-in R-hat/ESS
  convergence warnings after sampling.
- `pathlib.Path` support for `Model(file=...)` and `check_model(file=...)`.
- PEP 561 `py.typed` marker file for type checker and IDE support.
- `console.pyi` type stub for the C++ extension module.
- Comprehensive numpy-style docstrings on all public modules, classes,
  methods, and properties for auto-documentation readiness.
- "Getting Started" Jupyter notebook: beginner-friendly introduction.
- "New in v2.3.0" Jupyter notebook: showcase of all new features.

### Changed
- `SamplingState` and `check_model` are now exported from the top-level
  `pyjags` namespace.
- README updated with new features and notebook table.

## [2.2.0] - 2026-03-13

### Added
- `pyjags.summary()` convenience function for MCMC diagnostics (mean, sd, HDI,
  ESS, Rhat) without manual ArviZ conversion.
- `pyjags.version_info()` function reporting versions of pyjags, JAGS, numpy,
  arviz, h5py, and Python for debugging installation issues.
- Enhanced JAGS error messages: syntax errors now display the offending model
  code lines with an arrow pointing to the error location.
- Ruff linting and formatting configured in `pyproject.toml`.
- Ruff lint check added to CI workflow.
- Python 3.14 added to CI test matrix and wheel build matrix.
- `twine check` validation step added to the release workflow.

### Changed
- Codebase formatted with `ruff format` for consistent style.
- Minor lint fixes across the codebase (import sorting, modern type annotations,
  explicit re-exports).

## [2.1.0] - 2026-03-12

### Added
- Modern build system: `pyproject.toml` + scikit-build-core + CMake, replacing
  `setup.py`, `setup.cfg`, and `MANIFEST.in`.
- `setuptools-scm` for automatic version inference from git tags, replacing
  versioneer (removed 2,159 lines of generated code).
- `src/` layout for proper package isolation in non-editable installs.
- `CMakeLists.txt` with pkg-config primary + fallback path search for JAGS,
  including macOS architecture detection (arm64 vs x86_64).
- GitHub Actions CI: test workflow on Linux (Python 3.12, 3.13) and macOS.
- GitHub Actions release workflow: tag-triggered builds of sdist + wheels for
  Linux (x86_64, aarch64) and macOS (arm64, x86_64) via cibuildwheel, with
  automated PyPI publishing via trusted publishing.
- `scripts/install-jags-manylinux.sh` for compiling JAGS from source inside
  manylinux containers.
- `RELEASING.md` documenting the automated release process.
- Windows/WSL2 guidance in README.
- Acknowledgement of Max Schulist's PR #1 in README.

### Removed
- `setup.py`, `setup.cfg`, `MANIFEST.in`, `pytest.ini` (replaced by
  `pyproject.toml`).
- `versioneer.py` and `pyjags/_version.py` (replaced by setuptools-scm).
- pybind11 git submodule (now a pip build dependency).
- `.gitmodules` file.
- Dead Windows build code from `setup.py`.
- `publishing_on_pypi.txt` (replaced by `RELEASING.md`).

### Fixed
- Python 3.12 `SafeConfigParser` error caused by versioneer (reported by
  Pierre Gérenton).

## [2.0.0] - 2026-03-10

### Added
- ArviZ 1.0+ integration with `pyjags.from_pyjags()` converter supporting
  posterior, prior, log-likelihood, warmup splitting, coordinates, and
  dimensions.
- Comprehensive test suite with pytest and Hypothesis property-based testing
  (166 tests covering all public API).
- Docker-based development environment with `jagslab` CLI.

### Changed
- Minimum Python version raised to 3.12 (required by ArviZ 1.0).
- Minimum ArviZ version raised to 1.0.

[2.3.1]: https://github.com/michaelnowotny/pyjags/compare/2.3.0...2.3.1
[2.3.0]: https://github.com/michaelnowotny/pyjags/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/michaelnowotny/pyjags/compare/2.1.0...2.2.0
[2.1.0]: https://github.com/michaelnowotny/pyjags/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/michaelnowotny/pyjags/releases/tag/2.0.0
