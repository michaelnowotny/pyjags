# Copyright (C) 2020 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Convert PyJAGS sample dictionaries to ArviZ inference data objects.

PyJAGS sample arrays have shape ``(*variable_dims, iterations, chains)``.
ArviZ expects arrays with shape ``(chains, draws, *variable_dims)``.
This module handles the conversion and delegates to ``arviz.from_dict``.
"""

import typing as tp

import numpy as np


def _convert_pyjags_samples_to_arviz(
    pyjags_samples: tp.Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reshape a PyJAGS sample dictionary for ArviZ.

    Parameters
    ----------
    pyjags_samples
        Mapping of variable names to arrays with shape
        ``(*variable_dims, iterations, chains)``.

    Returns
    -------
    Dictionary of variable names to arrays with shape
    ``(chains, draws, *variable_dims)``.
    """
    result = {}
    for name, arr in pyjags_samples.items():
        arr = np.asarray(arr)
        if arr.ndim < 2:
            raise ValueError(
                f"Expected at least 2 dimensions for variable '{name}', "
                f"got shape {arr.shape}"
            )
        # Last two axes are (iterations, chains).  Move chains to front,
        # iterations to second position, keep variable dims after that.
        # (*var_dims, iterations, chains) -> (chains, iterations, *var_dims)
        n_dims = arr.ndim
        if n_dims == 2:
            # Shape (iterations, chains) — scalar variable with no param dim.
            result[name] = arr.T  # -> (chains, draws)
        elif n_dims == 3 and arr.shape[0] == 1:
            # Shape (1, iterations, chains) — scalar variable.
            # Squeeze out the singleton param dim to get (chains, draws).
            result[name] = arr[0, :, :].T
        else:
            # Shape (*var_dims, iterations, chains) — vector/matrix variable.
            # -> (chains, draws, *var_dims)
            new_axes = [n_dims - 1, n_dims - 2, *list(range(n_dims - 2))]
            result[name] = np.transpose(arr, axes=new_axes)
    return result


def from_pyjags(
    posterior: tp.Mapping[str, np.ndarray] | None = None,
    *,
    prior: tp.Mapping[str, np.ndarray] | None = None,
    log_likelihood: str
    | list[str]
    | tuple[str, ...]
    | tp.Mapping[str, str]
    | None = None,
    observed_data: tp.Mapping[str, np.ndarray] | None = None,
    constant_data: tp.Mapping[str, np.ndarray] | None = None,
    coords: tp.Mapping | None = None,
    dims: tp.Mapping | None = None,
    save_warmup: bool | None = None,
    warmup_iterations: int = 0,
):
    """Convert PyJAGS posterior samples to an ArviZ ``DataTree``.

    Parameters
    ----------
    posterior
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, iterations, chains)`` as returned by
        ``Model.sample()``.
    prior
        Prior samples in the same format as *posterior*.
    log_likelihood
        Variables to place in the ``log_likelihood`` group.  Can be:

        - a single variable name (str),
        - a list/tuple of variable names (extracted from *posterior*),
        - a mapping ``{obs_name: posterior_var_name}``.
    observed_data
        Dictionary of observed data arrays.  Stored in the
        ``observed_data`` group of the returned ``DataTree``.
    constant_data
        Dictionary of constant (non-random) data arrays.  Stored in the
        ``constant_data`` group of the returned ``DataTree``.
    coords
        Mapping of dimension names to coordinate values, forwarded to
        ``arviz.from_dict``.
    dims
        Mapping of variable names to dimension names, forwarded to
        ``arviz.from_dict``.
    save_warmup
        Whether to save warmup draws in a separate group.
    warmup_iterations
        Number of initial iterations to treat as warmup.  Only used when
        *save_warmup* is ``True``.

    Returns
    -------
    xarray.DataTree
        ArviZ inference data object.
    """
    import arviz as az

    data: dict[str, dict[str, np.ndarray]] = {}

    if posterior is not None:
        posterior = dict(posterior)

        # Extract log-likelihood variables from posterior if requested.
        ll_samples = None
        if log_likelihood is not None:
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            if isinstance(log_likelihood, (list, tuple)):
                log_likelihood = {name: name for name in log_likelihood}
            ll_samples = {
                obs_name: posterior.pop(ll_name)
                for obs_name, ll_name in log_likelihood.items()
            }

        # Split warmup if requested.
        if warmup_iterations > 0 and save_warmup:
            warmup_post, posterior = _split_warmup(posterior, warmup_iterations)
            data["warmup_posterior"] = _convert_pyjags_samples_to_arviz(warmup_post)
            if ll_samples is not None:
                warmup_ll, ll_samples = _split_warmup(ll_samples, warmup_iterations)
                data["warmup_log_likelihood"] = _convert_pyjags_samples_to_arviz(
                    warmup_ll
                )
        elif warmup_iterations > 0:
            # Discard warmup without saving.
            _, posterior = _split_warmup(posterior, warmup_iterations)
            if ll_samples is not None:
                _, ll_samples = _split_warmup(ll_samples, warmup_iterations)

        data["posterior"] = _convert_pyjags_samples_to_arviz(posterior)
        if ll_samples is not None:
            data["log_likelihood"] = _convert_pyjags_samples_to_arviz(ll_samples)

    if prior is not None:
        data["prior"] = _convert_pyjags_samples_to_arviz(prior)

    if observed_data is not None:
        data["observed_data"] = dict(observed_data)

    if constant_data is not None:
        data["constant_data"] = dict(constant_data)

    result = az.from_dict(
        data,
        sample_dims=["chain", "draw"],
        save_warmup=save_warmup if save_warmup and warmup_iterations > 0 else None,
        coords=coords,
        dims=dims,
    )

    # Set metadata attributes following ArviZ conventions.
    import contextlib
    from importlib.metadata import version as _metadata_version

    result.attrs["inference_library"] = "pyjags"
    with contextlib.suppress(Exception):
        result.attrs["inference_library_version"] = _metadata_version("pyjags")

    return result


def summary(
    samples: tp.Mapping[str, np.ndarray],
    var_names: list[str] | None = None,
    **kwargs,
):
    """Compute summary statistics for PyJAGS samples.

    Convenience wrapper around ``arviz.summary()`` that converts PyJAGS
    sample dictionaries automatically.

    Parameters
    ----------
    samples
        Dictionary mapping variable names to numpy arrays with shape
        ``(*variable_dims, iterations, chains)`` as returned by
        ``Model.sample()``.
    var_names
        Variables to include in the summary.  If ``None``, all variables
        are included.
    **kwargs
        Additional keyword arguments passed to ``arviz.summary()``.

    Returns
    -------
    pandas.DataFrame
        Summary table with mean, sd, HDI, ESS, and Rhat for each
        variable.
    """
    import arviz as az

    idata = from_pyjags(samples)
    return az.summary(idata, var_names=var_names, **kwargs)


def _split_warmup(
    samples: dict[str, np.ndarray],
    warmup_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split sample arrays at the warmup boundary.

    The iteration axis is the second-to-last axis (``axis=-2``) in the
    PyJAGS convention ``(*variable_dims, iterations, chains)``.
    """
    warmup = {}
    actual = {}
    for name, arr in samples.items():
        # iterations are along axis -2
        warmup[name] = arr[..., :warmup_iterations, :]
        actual[name] = arr[..., warmup_iterations:, :]
    return warmup, actual


def loo(
    posterior: tp.Mapping[str, np.ndarray],
    *,
    log_likelihood: str | list[str] | tuple[str, ...] | tp.Mapping[str, str],
    **kwargs,
):
    """Compute Pareto-smoothed importance-sampling leave-one-out cross-validation.

    Convenience wrapper around ``arviz.loo()`` that accepts PyJAGS sample
    dictionaries directly.

    Parameters
    ----------
    posterior
        Sample dictionary as returned by ``Model.sample()``.  Must
        contain the log-likelihood variable(s) specified in
        *log_likelihood*.
    log_likelihood
        Variable(s) to use as pointwise log-likelihood.  Accepts the
        same forms as ``from_pyjags(log_likelihood=...)``.
    **kwargs
        Additional keyword arguments forwarded to ``arviz.loo()``.

    Returns
    -------
    arviz.ELPDData
        LOO-CV results including ``elpd_loo``, ``p_loo``, and
        ``pareto_k`` diagnostics.
    """
    import arviz as az

    idata = from_pyjags(posterior, log_likelihood=log_likelihood)
    return az.loo(idata, **kwargs)


def waic(
    posterior: tp.Mapping[str, np.ndarray],
    *,
    log_likelihood: str | list[str] | tuple[str, ...] | tp.Mapping[str, str],
    **kwargs,
):
    """Compute the Widely Applicable Information Criterion (WAIC).

    Convenience wrapper around ``arviz.waic()`` that accepts PyJAGS sample
    dictionaries directly.

    Parameters
    ----------
    posterior
        Sample dictionary as returned by ``Model.sample()``.  Must
        contain the log-likelihood variable(s) specified in
        *log_likelihood*.
    log_likelihood
        Variable(s) to use as pointwise log-likelihood.  Accepts the
        same forms as ``from_pyjags(log_likelihood=...)``.
    **kwargs
        Additional keyword arguments forwarded to ``arviz.waic()``.

    Returns
    -------
    arviz.ELPDData
        WAIC results including ``elpd_waic``, ``p_waic``, and
        pointwise values.
    """
    import arviz as az

    idata = from_pyjags(posterior, log_likelihood=log_likelihood)
    return az.waic(idata, **kwargs)


def compare(
    model_dict: tp.Mapping[str, tp.Mapping[str, np.ndarray]],
    *,
    log_likelihood: str | list[str] | tuple[str, ...] | tp.Mapping[str, str],
    ic: str = "loo",
    **kwargs,
):
    """Compare multiple models using LOO-CV or WAIC.

    Convenience wrapper around ``arviz.compare()`` that accepts PyJAGS
    sample dictionaries directly.

    Parameters
    ----------
    model_dict
        Mapping of model names to sample dictionaries.  Each sample
        dictionary must contain the log-likelihood variable(s)
        specified in *log_likelihood*.
    log_likelihood
        Variable(s) to use as pointwise log-likelihood.  Applied to
        all models.
    ic : str
        Information criterion: ``"loo"`` (default) or ``"waic"``.
    **kwargs
        Additional keyword arguments forwarded to ``arviz.compare()``.

    Returns
    -------
    pandas.DataFrame
        Comparison table ranked by the chosen information criterion,
        with columns for ``elpd``, ``p``, ``weight``, and more.
    """
    import arviz as az

    idata_dict = {
        name: from_pyjags(samples, log_likelihood=log_likelihood)
        for name, samples in model_dict.items()
    }
    return az.compare(idata_dict, ic=ic, **kwargs)
