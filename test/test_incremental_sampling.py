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

from pyjags.incremental_sampling import (
    EffectiveSampleSizeAndRHatCriterion,
    EffectiveSampleSizeCriterion,
    RHatDeviationCriterion,
)

# ---------------------------------------------------------------------------
# EffectiveSampleSizeCriterion
# ---------------------------------------------------------------------------


class TestESSCriterion:
    def test_properties(self):
        c = EffectiveSampleSizeCriterion(minimum_ess=200, variable_names=["mu"])
        assert c.minimum_ess == 200
        assert c.variable_names == ["mu"]

    def test_properties_default_vars(self):
        c = EffectiveSampleSizeCriterion(minimum_ess=100)
        assert c.variable_names is None

    def test_satisfied_with_iid_draws(self):
        # iid draws across 4 chains -> high ESS
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = EffectiveSampleSizeCriterion(minimum_ess=100)
        assert criterion(samples, verbose=False)

    def test_not_satisfied_with_few_draws(self):
        # very few draws -> low ESS
        samples = {"mu": np.random.randn(1, 5, 4)}
        criterion = EffectiveSampleSizeCriterion(minimum_ess=1000)
        assert not criterion(samples, verbose=False)


# ---------------------------------------------------------------------------
# RHatDeviationCriterion
# ---------------------------------------------------------------------------


class TestRHatCriterion:
    def test_properties(self):
        c = RHatDeviationCriterion(maximum_rhat_deviation=0.05, variable_names=["mu"])
        assert c.maximum_rhat_deviation == 0.05
        assert c.variable_names == ["mu"]

    def test_satisfied_with_converged_chains(self):
        # iid draws from same distribution -> rhat near 1
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = RHatDeviationCriterion(maximum_rhat_deviation=0.1)
        assert criterion(samples, verbose=False)

    def test_not_satisfied_with_divergent_chains(self):
        # chains with very different means -> rhat far from 1
        arr = np.zeros((1, 500, 4))
        for chain in range(4):
            arr[0, :, chain] = np.random.randn(500) + chain * 100
        samples = {"mu": arr}
        criterion = RHatDeviationCriterion(maximum_rhat_deviation=0.01)
        assert not criterion(samples, verbose=False)


# ---------------------------------------------------------------------------
# EffectiveSampleSizeAndRHatCriterion
# ---------------------------------------------------------------------------


class TestCombinedCriterion:
    def test_properties(self):
        c = EffectiveSampleSizeAndRHatCriterion(
            minimum_ess=100,
            maximum_rhat_deviation=0.05,
            variable_names=["mu"],
        )
        assert c.minimum_ess == 100
        assert c.maximum_rhat_deviation == 0.05
        assert c.variable_names == ["mu"]

    def test_both_satisfied(self):
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = EffectiveSampleSizeAndRHatCriterion(
            minimum_ess=100, maximum_rhat_deviation=0.1
        )
        assert criterion(samples, verbose=False)

    def test_ess_not_satisfied(self):
        samples = {"mu": np.random.randn(1, 5, 4)}
        criterion = EffectiveSampleSizeAndRHatCriterion(
            minimum_ess=1000, maximum_rhat_deviation=0.5
        )
        assert not criterion(samples, verbose=False)

    def test_rhat_not_satisfied(self):
        arr = np.zeros((1, 500, 4))
        for chain in range(4):
            arr[0, :, chain] = np.random.randn(500) + chain * 100
        samples = {"mu": arr}
        criterion = EffectiveSampleSizeAndRHatCriterion(
            minimum_ess=1, maximum_rhat_deviation=0.01
        )
        assert not criterion(samples, verbose=False)


# ---------------------------------------------------------------------------
# Verbose output (coverage for print branches)
# ---------------------------------------------------------------------------


class TestVerboseOutput:
    def test_ess_verbose(self, capsys):
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = EffectiveSampleSizeCriterion(minimum_ess=100)
        criterion(samples, verbose=True)
        captured = capsys.readouterr()
        assert "minimum ess" in captured.out

    def test_rhat_verbose(self, capsys):
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = RHatDeviationCriterion(maximum_rhat_deviation=0.1)
        criterion(samples, verbose=True)
        captured = capsys.readouterr()
        assert "maximum rhat deviation" in captured.out

    def test_combined_verbose(self, capsys):
        samples = {"mu": np.random.randn(1, 500, 4)}
        criterion = EffectiveSampleSizeAndRHatCriterion(
            minimum_ess=100, maximum_rhat_deviation=0.1
        )
        criterion(samples, verbose=True)
        captured = capsys.readouterr()
        assert "minimum ess" in captured.out
        assert "maximum rhat deviation" in captured.out


# ---------------------------------------------------------------------------
# sample_until (JAGS integration)
# ---------------------------------------------------------------------------


class TestSampleUntil:
    """Integration tests for sample_until — require a running JAGS engine."""

    @pytest.fixture()
    def simple_model(self):
        import pyjags

        return pyjags.Model(
            code="model { mu ~ dnorm(0, 1) }",
            chains=4,
            adapt=500,
        )

    @pytest.mark.slow
    def test_sample_until_ess(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        criterion = EffectiveSampleSizeCriterion(minimum_ess=100)
        result = sample_until(
            simple_model,
            criterion=criterion,
            chunk_size=200,
            max_iterations=5000,
        )
        assert "mu" in result
        # Shape: (*param_dims, iterations, chains)
        assert result["mu"].shape[-1] == 4  # 4 chains
        assert result["mu"].shape[-2] >= 200  # at least one chunk

    @pytest.mark.slow
    def test_sample_until_with_previous_samples(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        # Get initial samples
        initial = simple_model.sample(100)
        criterion = EffectiveSampleSizeCriterion(minimum_ess=200)
        result = sample_until(
            simple_model,
            criterion=criterion,
            previous_samples=initial,
            chunk_size=200,
            max_iterations=5000,
        )
        assert "mu" in result
        # Should have more iterations than initial 100
        assert result["mu"].shape[-2] >= 100

    @pytest.mark.slow
    def test_sample_until_already_satisfied(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        # Pre-generate enough samples that criterion is already met
        initial = simple_model.sample(1000)
        criterion = EffectiveSampleSizeCriterion(minimum_ess=10)
        result = sample_until(
            simple_model,
            criterion=criterion,
            previous_samples=initial,
            chunk_size=200,
            max_iterations=5000,
        )
        # Should return immediately with the same samples
        assert result["mu"].shape[-2] == 1000

    @pytest.mark.slow
    def test_sample_until_max_iterations(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        # Set impossibly high ESS requirement with low max_iterations
        criterion = EffectiveSampleSizeCriterion(minimum_ess=1_000_000)
        result = sample_until(
            simple_model,
            criterion=criterion,
            chunk_size=50,
            max_iterations=100,
        )
        assert "mu" in result
        # Should stop after max_iterations
        assert result["mu"].shape[-2] <= 100

    @pytest.mark.slow
    def test_sample_until_iteration_function(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        calls = []

        def track(samples, satisfied, total_iters):
            calls.append((satisfied, total_iters))

        criterion = EffectiveSampleSizeCriterion(minimum_ess=50)
        sample_until(
            simple_model,
            criterion=criterion,
            chunk_size=200,
            max_iterations=2000,
            iteration_function=track,
        )
        assert len(calls) >= 1
        # Last call should be satisfied (or max_iterations reached)
        last_satisfied, last_iters = calls[-1]
        assert last_satisfied or last_iters >= 2000

    @pytest.mark.slow
    def test_sample_until_chunk_exceeds_max_raises(self, simple_model):
        from pyjags.incremental_sampling import sample_until

        criterion = EffectiveSampleSizeCriterion(minimum_ess=100)
        with pytest.raises(ValueError, match="chunk_size must be less than"):
            sample_until(
                simple_model,
                criterion=criterion,
                chunk_size=1000,
                max_iterations=500,
            )
