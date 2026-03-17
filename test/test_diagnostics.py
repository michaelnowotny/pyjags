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

"""Tests for pyjags.diagnostics (Divergence integration)."""

import numpy as np
import pytest

import pyjags

# Skip entire module if divergence is not installed
divergence = pytest.importorskip("divergence")

MODEL_CODE = """
model {
    mu ~ dnorm(0, 0.001)
    sigma ~ dunif(0, 100)
    tau <- pow(sigma, -2)
    for (i in 1:N) { y[i] ~ dnorm(mu, tau) }
}
"""

PRIOR_CODE = """
model {
    mu ~ dnorm(0, 0.001)
    sigma ~ dunif(0, 100)
}
"""


@pytest.fixture(scope="module")
def idata_with_prior():
    """Create InferenceData with both posterior and prior groups."""
    np.random.seed(42)
    y = np.random.normal(5, 2, 30)
    data = dict(y=y, N=len(y))

    # Prior samples (no likelihood)
    prior_model = pyjags.Model(
        code=PRIOR_CODE,
        data={},
        chains=4,
        adapt=500,
        progress_bar=False,
        seed=99,
    )
    prior_samples = prior_model.sample(2000, vars=["mu", "sigma"])

    # Posterior samples
    model = pyjags.Model(
        code=MODEL_CODE,
        data=data,
        chains=4,
        adapt=1000,
        progress_bar=False,
        seed=42,
    )
    model.sample(1000, vars=[])
    posterior_samples = model.sample(3000, vars=["mu", "sigma"])

    return pyjags.from_pyjags(posterior_samples, prior=prior_samples)


@pytest.fixture(scope="module")
def idata_posterior_only():
    """Create InferenceData with posterior only (no prior)."""
    np.random.seed(42)
    y = np.random.normal(5, 2, 30)
    data = dict(y=y, N=len(y))

    model = pyjags.Model(
        code=MODEL_CODE,
        data=data,
        chains=4,
        adapt=1000,
        progress_bar=False,
        seed=42,
    )
    model.sample(1000, vars=[])
    posterior_samples = model.sample(3000, vars=["mu", "sigma"])

    return pyjags.from_pyjags(posterior_samples)


class TestConvergenceReport:
    """Test convergence_report with real JAGS model."""

    def test_returns_expected_keys(self, idata_posterior_only):
        from pyjags.diagnostics import convergence_report

        report = convergence_report(idata_posterior_only)
        assert "rhat" in report
        assert "ess_bulk" in report
        assert "ess_tail" in report
        assert "chain_divergence" in report
        assert "max_rhat" in report
        assert "max_chain_divergence" in report
        assert "converged" in report

    def test_well_mixed_chains_converge(self, idata_posterior_only):
        from pyjags.diagnostics import convergence_report

        report = convergence_report(idata_posterior_only)
        assert report["converged"]
        assert report["max_rhat"] < 1.01

    def test_chain_divergence_is_small(self, idata_posterior_only):
        from pyjags.diagnostics import convergence_report

        report = convergence_report(idata_posterior_only)
        assert report["max_chain_divergence"] < 0.1

    def test_ess_is_positive(self, idata_posterior_only):
        from pyjags.diagnostics import convergence_report

        report = convergence_report(idata_posterior_only)
        for param, ess in report["ess_bulk"].items():
            assert ess > 0, f"ESS for {param} should be positive"

    def test_var_names_filter(self, idata_posterior_only):
        from pyjags.diagnostics import convergence_report

        report = convergence_report(idata_posterior_only, var_names=["mu"])
        assert "mu" in report["rhat"]
        assert "sigma" not in report["rhat"]


class TestInformationGain:
    """Test information_gain wrapper."""

    def test_positive_with_informative_data(self, idata_with_prior):
        from pyjags.diagnostics import information_gain

        ig = information_gain(idata_with_prior)
        assert "mu" in ig
        assert "sigma" in ig
        # With 30 data points from N(5,2), both should be substantially informed
        assert ig["mu"] > 0.5, "mu should have substantial information gain"

    def test_multiple_methods(self, idata_with_prior):
        from pyjags.diagnostics import information_gain

        for method in ["kl", "js", "hellinger", "energy"]:
            ig = information_gain(idata_with_prior, method=method)
            assert "mu" in ig

    def test_missing_prior_raises(self, idata_posterior_only):
        from pyjags.diagnostics import information_gain

        with pytest.raises(ValueError, match="prior"):
            information_gain(idata_posterior_only)

    def test_var_names_filter(self, idata_with_prior):
        from pyjags.diagnostics import information_gain

        ig = information_gain(idata_with_prior, var_names=["mu"])
        assert "mu" in ig
        assert "sigma" not in ig


class TestSamplingStateChainDivergence:
    """Test SamplingState.chain_divergence lazy property."""

    def test_returns_dict_with_divergence_installed(self):
        np.random.seed(42)
        y = np.random.normal(5, 2, 30)
        data = dict(y=y, N=len(y))

        model = pyjags.Model(
            code=MODEL_CODE,
            data=data,
            chains=4,
            adapt=500,
            progress_bar=False,
            seed=42,
        )
        model.sample(500, vars=[])

        for state in model.iter_sample(
            iterations=1000, chunk_size=1000, vars=["mu", "sigma"]
        ):
            cd = state.chain_divergence
            assert cd is not None
            assert "mu" in cd
            assert cd["mu"].shape == (4, 4)
            # Diagonal should be near zero
            assert cd["mu"][0, 0] < 1e-10
            break  # one chunk is enough


class TestPublicAPI:
    """Test that diagnostics are importable from the expected paths."""

    def test_diagnostics_importable(self):
        from pyjags.diagnostics import (  # noqa: F401
            bayesian_surprise,
            convergence_report,
            information_gain,
            model_divergence,
            prior_sensitivity,
            uncertainty_decomposition,
        )
