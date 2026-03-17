# Copyright (C) 2026 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Optional advanced diagnostics powered by the Divergence package.

These functions require the ``divergence`` package to be installed::

    pip install divergence[bayesian]

All functions accept ArviZ ``InferenceData`` objects as produced by
:func:`pyjags.from_pyjags`.  When ``divergence`` is not installed,
importing this module succeeds but calling any function raises
:class:`ImportError` with installation instructions.
"""

import typing as tp

import numpy as np


def _import_divergence():
    """Import divergence, raising a helpful error if not installed."""
    try:
        import divergence
    except ImportError:
        raise ImportError(
            "The 'divergence' package is required for advanced diagnostics. "
            "Install with: pip install divergence[bayesian]"
        ) from None
    return divergence


def convergence_report(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    chain_method: str = "energy",
) -> dict[str, tp.Any]:
    """Comprehensive convergence diagnostic combining ArviZ and Divergence.

    Computes R-hat and ESS (from ArviZ) alongside pairwise chain
    divergence (from Divergence) in a single call.

    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data with a ``posterior`` group containing multiple
        chains.
    var_names : list[str], optional
        Parameters to diagnose.  ``None`` means all.
    chain_method : str
        Divergence measure for chain comparison.  Default ``"energy"``
        (parameter-free, works in any dimension).  Also accepts
        ``"mmd"``, ``"js"``, ``"kl"``, ``"tv"``, ``"wasserstein"``.

    Returns
    -------
    dict
        Dictionary with keys:

        ``"rhat"``
            dict mapping parameter names to R-hat values.
        ``"ess_bulk"``
            dict mapping parameter names to bulk ESS values.
        ``"ess_tail"``
            dict mapping parameter names to tail ESS values.
        ``"chain_divergence"``
            dict mapping parameter names to pairwise chain divergence
            matrices (numpy arrays of shape ``(n_chains, n_chains)``).
        ``"max_rhat"``
            Maximum R-hat across all parameters.
        ``"max_chain_divergence"``
            Maximum off-diagonal chain divergence across all parameters.
        ``"converged"``
            ``True`` if all R-hat values are below 1.01.
    """
    import arviz as az

    div = _import_divergence()

    rhat = az.rhat(idata, var_names=var_names)
    ess_bulk = az.ess(idata, var_names=var_names, method="bulk")
    ess_tail = az.ess(idata, var_names=var_names, method="tail")
    chain_div = div.chain_divergence(idata, var_names=var_names, method=chain_method)

    rhat_dict = {str(v): float(rhat[v]) for v in rhat.data_vars}
    ess_bulk_dict = {str(v): float(ess_bulk[v]) for v in ess_bulk.data_vars}
    ess_tail_dict = {str(v): float(ess_tail[v]) for v in ess_tail.data_vars}

    max_rhat = max(rhat_dict.values()) if rhat_dict else 0.0
    max_chain_div = 0.0
    if chain_div:
        for m in chain_div.values():
            offdiag = m[np.triu_indices_from(m, k=1)]
            if len(offdiag) > 0:
                max_chain_div = max(max_chain_div, float(offdiag.max()))

    return {
        "rhat": rhat_dict,
        "ess_bulk": ess_bulk_dict,
        "ess_tail": ess_tail_dict,
        "chain_divergence": chain_div,
        "max_rhat": max_rhat,
        "max_chain_divergence": max_chain_div,
        "converged": max_rhat < 1.01,
    }


def information_gain(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "kl",
) -> dict[str, float]:
    """Compute prior-to-posterior information gain per parameter.

    Measures how much the data updated beliefs about each parameter,
    expressed as a divergence from the prior to the posterior distribution.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain both ``posterior`` and ``prior`` groups.
    var_names : list[str], optional
        Parameters to analyze.  ``None`` means all.
    method : str
        Divergence measure: ``"kl"`` (default), ``"js"``, ``"hellinger"``,
        ``"tv"``, ``"wasserstein"``, ``"mmd"``, ``"energy"``.

    Returns
    -------
    dict[str, float]
        Divergence from prior to posterior for each parameter.
        High values mean the data was informative; near-zero values
        mean the parameter is prior-dominated.
    """
    div = _import_divergence()
    return div.information_gain(idata, var_names=var_names, method=method)


def bayesian_surprise(
    idata: tp.Any,
    *,
    var_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-observation Bayesian surprise.

    Surprise is the negative log posterior predictive probability of each
    observation.  High values indicate influential or outlying data points.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain a ``log_likelihood`` group.
    var_name : str, optional
        Log-likelihood variable to use.  If the group contains exactly
        one variable, it is selected automatically.

    Returns
    -------
    dict[str, numpy.ndarray]
        Surprise score per observation.
    """
    div = _import_divergence()
    return div.bayesian_surprise(idata, var_name=var_name)


def model_divergence(
    idata1: tp.Any,
    idata2: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "energy",
    group: str = "posterior_predictive",
) -> dict[str, float]:
    """Compare predictive distributions from two models.

    Complements LOO-CV (a scalar score) with a distributional comparison
    that shows *how differently* two models see the data.

    Parameters
    ----------
    idata1, idata2 : arviz.InferenceData
        Inference data from two models, each containing the specified
        group.
    var_names : list[str], optional
        Variables to compare.
    method : str
        Divergence measure (default ``"energy"``).
    group : str
        InferenceData group to compare (default
        ``"posterior_predictive"``).

    Returns
    -------
    dict[str, float]
        Divergence between the two models' distributions per variable.
    """
    div = _import_divergence()
    return div.model_divergence(
        idata1, idata2, var_names=var_names, method=method, group=group
    )


def prior_sensitivity(
    idata: tp.Any,
    reference_prior_samples: dict[str, np.ndarray],
    *,
    var_names: list[str] | None = None,
    method: str = "kl",
) -> dict[str, dict[str, float]]:
    """Quantify sensitivity of posteriors to prior choice.

    Compares the information gain under the actual prior to the
    information gain under a reference (e.g., vague) prior.  High
    sensitivity means conclusions depend on the prior.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain ``posterior`` and ``prior`` groups.
    reference_prior_samples : dict[str, numpy.ndarray]
        Samples from an alternative prior, as a dictionary mapping
        parameter names to 1-D numpy arrays.
    var_names : list[str], optional
        Parameters to analyze.
    method : str
        Divergence measure (default ``"kl"``).

    Returns
    -------
    dict[str, dict[str, float]]
        Per parameter: ``{"actual": ..., "reference": ...,
        "sensitivity": ...}``.
    """
    div = _import_divergence()
    return div.prior_sensitivity(
        idata, reference_prior_samples, var_names=var_names, method=method
    )


def uncertainty_decomposition(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Decompose predictive uncertainty into aleatoric and epistemic.

    Uses the entropy identity:

        Total = Aleatoric + Epistemic

    where aleatoric uncertainty is irreducible data noise and epistemic
    uncertainty is reducible parameter uncertainty.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain a ``posterior_predictive`` group with an
        observation dimension.
    var_names : list[str], optional
        Variables to decompose.

    Returns
    -------
    dict[str, dict[str, float]]
        Per variable: ``{"total": ..., "aleatoric": ...,
        "epistemic": ...}``.
    """
    div = _import_divergence()
    return div.uncertainty_decomposition(idata, var_names=var_names)
