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
) -> tp.Dict[str, np.ndarray]:
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
            new_axes = [n_dims - 1, n_dims - 2] + list(range(n_dims - 2))
            result[name] = np.transpose(arr, axes=new_axes)
    return result


def from_pyjags(
    posterior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
    *,
    prior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
    log_likelihood: tp.Optional[
        tp.Union[str, tp.List[str], tp.Tuple[str, ...], tp.Mapping[str, str]]
    ] = None,
    coords: tp.Optional[tp.Mapping] = None,
    dims: tp.Optional[tp.Mapping] = None,
    save_warmup: tp.Optional[bool] = None,
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

    data: tp.Dict[str, tp.Dict[str, np.ndarray]] = {}

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
                data["warmup_log_likelihood"] = _convert_pyjags_samples_to_arviz(warmup_ll)
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

    return az.from_dict(
        data,
        sample_dims=["chain", "draw"],
        save_warmup=save_warmup if save_warmup and warmup_iterations > 0 else None,
        coords=coords,
        dims=dims,
    )


def _split_warmup(
    samples: tp.Dict[str, np.ndarray],
    warmup_iterations: int,
) -> tp.Tuple[tp.Dict[str, np.ndarray], tp.Dict[str, np.ndarray]]:
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