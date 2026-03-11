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
    EffectiveSampleSizeCriterion,
    RHatDeviationCriterion,
    EffectiveSampleSizeAndRHatCriterion,
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
        c = RHatDeviationCriterion(
            maximum_rhat_deviation=0.05, variable_names=["mu"]
        )
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