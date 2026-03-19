# API Reference

PyJAGS provides a clean, well-documented API with 100% docstring coverage.

| Module | Description |
|--------|-------------|
| [Model](model.md) | `Model`, `SamplingState`, `check_model` -- the core sampling API |
| [ArviZ](arviz.md) | `from_pyjags`, `loo`, `compare`, `summary` -- ArviZ conversion |
| [Chain Utilities](chain_utilities.md) | Merge, discard burn-in, extract final state |
| [Convergence](convergence.md) | ESS/R-hat criteria, `sample_until` |
| [DIC](dic.md) | Deviance Information Criterion |
| [I/O](io.md) | HDF5 save/load |
| [Diagnostics](diagnostics.md) | Divergence integration (optional) |
| [Modules](modules.md) | JAGS module loading |
