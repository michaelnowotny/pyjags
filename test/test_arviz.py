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

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st

from pyjags.arviz import _convert_pyjags_samples_to_arviz, from_pyjags, _split_warmup


# ---------------------------------------------------------------------------
# _convert_pyjags_samples_to_arviz
# ---------------------------------------------------------------------------

class TestConvertPyjagsSamplesToArviz:

    def test_scalar_variable(self):
        """Scalar variable: shape (1, iterations, chains) -> (chains, draws)."""
        iterations, chains = 100, 4
        samples = {"mu": np.random.randn(1, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)
        assert result["mu"].shape == (chains, iterations)

    def test_vector_variable(self):
        """Vector variable: shape (3, iterations, chains) -> (chains, draws, 3)."""
        param_dim, iterations, chains = 3, 100, 4
        samples = {"beta": np.random.randn(param_dim, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)
        assert result["beta"].shape == (chains, iterations, param_dim)

    def test_matrix_variable(self):
        """Matrix variable: shape (3, 5, iterations, chains) -> (chains, draws, 3, 5)."""
        d1, d2, iterations, chains = 3, 5, 100, 4
        samples = {"Sigma": np.random.randn(d1, d2, iterations, chains)}
        result = _convert_pyjags_samples_to_arviz(samples)
        assert result["Sigma"].shape == (chains, iterations, d1, d2)

    def test_values_preserved(self):
        """Verify that actual data values are correctly transposed."""
        arr = np.arange(12).reshape(2, 3, 2).astype(float)
        samples = {"x": arr}
        result = _convert_pyjags_samples_to_arviz(samples)
        # result[chain, draw, param] == arr[param, draw, chain]
        for chain in range(2):
            for draw in range(3):
                for param in range(2):
                    assert result["x"][chain, draw, param] == arr[param, draw, chain]

    def test_rejects_1d_array(self):
        """Arrays with fewer than 2 dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            _convert_pyjags_samples_to_arviz({"x": np.array([1.0, 2.0])})

    def test_multiple_variables(self):
        """Multiple variables with different shapes."""
        iterations, chains = 50, 3
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "beta": np.random.randn(5, iterations, chains),
            "Sigma": np.random.randn(2, 2, iterations, chains),
        }
        result = _convert_pyjags_samples_to_arviz(samples)
        assert result["mu"].shape == (chains, iterations)
        assert result["beta"].shape == (chains, iterations, 5)
        assert result["Sigma"].shape == (chains, iterations, 2, 2)

    def test_2d_array_scalar(self):
        """2D array (iterations, chains) — scalar with no param dim."""
        iterations, chains = 100, 4
        arr = np.random.randn(iterations, chains)
        result = _convert_pyjags_samples_to_arviz({"x": arr})
        assert result["x"].shape == (chains, iterations)

    @given(
        param_dim=st.integers(min_value=2, max_value=5),
        iterations=st.integers(min_value=2, max_value=30),
        chains=st.integers(min_value=1, max_value=4),
    )
    def test_shape_invariant(self, param_dim, iterations, chains):
        """Output shape is always (chains, iterations, *param_dims)."""
        arr = np.random.randn(param_dim, iterations, chains)
        result = _convert_pyjags_samples_to_arviz({"x": arr})
        assert result["x"].shape == (chains, iterations, param_dim)

    def test_empty_sample_dictionary(self):
        """Empty dictionary should return empty dictionary."""
        result = _convert_pyjags_samples_to_arviz({})
        assert result == {}

    def test_high_dimensional_array(self):
        """5D array: shape (2, 3, 4, iterations, chains) -> (chains, draws, 2, 3, 4)."""
        d1, d2, d3, iterations, chains = 2, 3, 4, 20, 2
        arr = np.random.randn(d1, d2, d3, iterations, chains)
        result = _convert_pyjags_samples_to_arviz({"tensor": arr})
        assert result["tensor"].shape == (chains, iterations, d1, d2, d3)

    def test_rejects_0d_array(self):
        """0D scalar array should raise ValueError."""
        with pytest.raises(ValueError):
            _convert_pyjags_samples_to_arviz({"x": np.float64(1.0)})


# ---------------------------------------------------------------------------
# from_pyjags
# ---------------------------------------------------------------------------

class TestFromPyjags:

    def _make_samples(self, iterations=100, chains=4):
        return {
            "mu": np.random.randn(1, iterations, chains),
            "sigma": np.random.randn(1, iterations, chains),
        }

    def test_basic_posterior(self):
        """Basic posterior conversion returns a DataTree with correct dims."""
        samples = self._make_samples()
        idata = from_pyjags(samples)
        posterior = idata["posterior"]
        assert "mu" in posterior.data_vars
        assert "sigma" in posterior.data_vars
        assert dict(posterior.sizes)["chain"] == 4
        assert dict(posterior.sizes)["draw"] == 100

    def test_ess_and_rhat_work(self):
        """ESS and Rhat should work on the converted data."""
        import arviz as az

        samples = self._make_samples(iterations=500, chains=4)
        idata = from_pyjags(samples)
        ess = az.ess(idata)
        rhat = az.rhat(idata)
        for var in ess.data_vars:
            assert float(ess[var]) > 0
        for var in rhat.data_vars:
            assert abs(float(rhat[var]) - 1.0) < 0.1

    def test_log_likelihood_string(self):
        """Log-likelihood extraction with a string variable name."""
        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(samples, log_likelihood="loglik")
        assert "loglik" not in idata["posterior"].data_vars
        assert "loglik" in idata["log_likelihood"].data_vars
        assert "mu" in idata["posterior"].data_vars

    def test_log_likelihood_list(self):
        """Log-likelihood extraction with a list of variable names."""
        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "ll_y1": np.random.randn(5, iterations, chains),
            "ll_y2": np.random.randn(3, iterations, chains),
        }
        idata = from_pyjags(samples, log_likelihood=["ll_y1", "ll_y2"])
        assert "ll_y1" in idata["log_likelihood"].data_vars
        assert "ll_y2" in idata["log_likelihood"].data_vars
        assert "ll_y1" not in idata["posterior"].data_vars

    def test_log_likelihood_mapping(self):
        """Log-likelihood extraction with a name mapping."""
        iterations, chains = 100, 4
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik_obs": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(samples, log_likelihood={"y": "loglik_obs"})
        assert "y" in idata["log_likelihood"].data_vars
        assert "loglik_obs" not in idata["posterior"].data_vars

    def test_prior(self):
        """Prior samples should go into a prior group."""
        iterations, chains = 100, 4
        posterior = {"mu": np.random.randn(1, iterations, chains)}
        prior = {"mu": np.random.randn(1, 50, chains)}
        idata = from_pyjags(posterior, prior=prior)
        assert "mu" in idata["posterior"].data_vars
        assert "mu" in idata["prior"].data_vars

    def test_warmup_saved(self):
        """Warmup iterations should be split into warmup_posterior."""
        iterations, chains = 200, 4
        warmup = 50
        samples = {"mu": np.random.randn(1, iterations, chains)}
        idata = from_pyjags(samples, save_warmup=True, warmup_iterations=warmup)
        assert dict(idata["posterior"].sizes)["draw"] == 150
        assert dict(idata["warmup_posterior"].sizes)["draw"] == 50

    def test_warmup_discarded(self):
        """Warmup iterations should be discarded when save_warmup is False."""
        iterations, chains = 200, 4
        warmup = 50
        samples = {"mu": np.random.randn(1, iterations, chains)}
        idata = from_pyjags(samples, save_warmup=False, warmup_iterations=warmup)
        assert dict(idata["posterior"].sizes)["draw"] == 150
        assert "warmup_posterior" not in [c for c in idata.children]

    def test_warmup_saved_with_log_likelihood(self):
        """Warmup + log_likelihood combined: both should be split correctly."""
        iterations, chains = 200, 4
        warmup = 50
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(
            samples,
            log_likelihood="loglik",
            save_warmup=True,
            warmup_iterations=warmup,
        )
        assert dict(idata["posterior"].sizes)["draw"] == 150
        assert dict(idata["warmup_posterior"].sizes)["draw"] == 50
        assert "loglik" in idata["log_likelihood"].data_vars
        assert "loglik" in idata["warmup_log_likelihood"].data_vars
        assert "loglik" not in idata["posterior"].data_vars

    def test_warmup_discarded_with_log_likelihood(self):
        """Warmup discarded + log_likelihood: warmup removed from both groups."""
        iterations, chains = 200, 4
        warmup = 50
        samples = {
            "mu": np.random.randn(1, iterations, chains),
            "loglik": np.random.randn(10, iterations, chains),
        }
        idata = from_pyjags(
            samples,
            log_likelihood="loglik",
            save_warmup=False,
            warmup_iterations=warmup,
        )
        assert dict(idata["posterior"].sizes)["draw"] == 150
        assert dict(idata["log_likelihood"].sizes)["draw"] == 150
        assert "warmup_posterior" not in [c for c in idata.children]

    def test_pyjags_public_api(self):
        """from_pyjags should be importable from the top-level pyjags package."""
        import pyjags
        assert callable(pyjags.from_pyjags)


# ---------------------------------------------------------------------------
# _split_warmup
# ---------------------------------------------------------------------------

class TestSplitWarmup:

    def test_split_at_boundary(self):
        arr = np.arange(60).reshape(1, 20, 3).astype(float)
        warmup, actual = _split_warmup({"x": arr}, warmup_iterations=5)
        assert warmup["x"].shape == (1, 5, 3)
        assert actual["x"].shape == (1, 15, 3)
        np.testing.assert_array_equal(warmup["x"], arr[:, :5, :])
        np.testing.assert_array_equal(actual["x"], arr[:, 5:, :])

    def test_split_warmup_zero(self):
        """Zero warmup iterations: all samples go to actual."""
        arr = np.random.randn(1, 20, 3)
        warmup, actual = _split_warmup({"x": arr}, warmup_iterations=0)
        assert warmup["x"].shape == (1, 0, 3)
        assert actual["x"].shape == (1, 20, 3)

    def test_split_warmup_equals_total(self):
        """Warmup == total iterations: all samples go to warmup."""
        arr = np.random.randn(1, 20, 3)
        warmup, actual = _split_warmup({"x": arr}, warmup_iterations=20)
        assert warmup["x"].shape == (1, 20, 3)
        assert actual["x"].shape == (1, 0, 3)

    def test_split_warmup_exceeds_total(self):
        """Warmup > total iterations: warmup gets all, actual is empty."""
        arr = np.random.randn(1, 10, 3)
        warmup, actual = _split_warmup({"x": arr}, warmup_iterations=50)
        assert warmup["x"].shape == (1, 10, 3)
        assert actual["x"].shape == (1, 0, 3)