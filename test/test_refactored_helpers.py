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

"""Tests for refactored internal helpers: _rng, _multi_console, diagnostics."""

import numpy as np
import pytest


class TestSeedToChainInits:
    """Tests for _rng.seed_to_chain_inits."""

    def test_returns_correct_count(self):
        from pyjags._rng import seed_to_chain_inits

        result = seed_to_chain_inits(42, 4)
        assert len(result) == 4

    def test_each_entry_has_rng_keys(self):
        from pyjags._rng import seed_to_chain_inits

        for entry in seed_to_chain_inits(42, 3):
            assert ".RNG.name" in entry
            assert ".RNG.seed" in entry

    def test_cycles_through_rng_names(self):
        from pyjags._rng import JAGS_RNG_NAMES, seed_to_chain_inits

        result = seed_to_chain_inits(42, 8)
        names = [r[".RNG.name"] for r in result]
        # Should cycle: 0,1,2,3,0,1,2,3
        for i, name in enumerate(names):
            assert name == JAGS_RNG_NAMES[i % len(JAGS_RNG_NAMES)]

    def test_same_seed_same_result(self):
        from pyjags._rng import seed_to_chain_inits

        a = seed_to_chain_inits(123, 4)
        b = seed_to_chain_inits(123, 4)
        assert a == b

    def test_different_seeds_different_result(self):
        from pyjags._rng import seed_to_chain_inits

        a = seed_to_chain_inits(1, 4)
        b = seed_to_chain_inits(2, 4)
        seeds_a = [r[".RNG.seed"] for r in a]
        seeds_b = [r[".RNG.seed"] for r in b]
        assert seeds_a != seeds_b

    def test_seeds_are_valid_integers(self):
        from pyjags._rng import seed_to_chain_inits

        for entry in seed_to_chain_inits(42, 10):
            seed = entry[".RNG.seed"]
            assert isinstance(seed, int)
            assert 0 <= seed < 2**31


class TestMergeSeedIntoInit:
    """Tests for Model._merge_seed_into_init static method."""

    def test_no_seed_returns_init_unchanged(self):
        from pyjags.model import Model

        init = {"mu": np.array([1.0])}
        result = Model._merge_seed_into_init(None, init, 4)
        assert result is init

    def test_seed_without_init_creates_inits(self):
        from pyjags.model import Model

        result = Model._merge_seed_into_init(42, None, 4)
        assert len(result) == 4
        assert all(".RNG.name" in r for r in result)

    def test_seed_with_dict_init_merges(self):
        from pyjags.model import Model

        init = {"mu": np.array([1.0])}
        result = Model._merge_seed_into_init(42, init, 4)
        assert len(result) == 4
        for r in result:
            assert "mu" in r
            assert ".RNG.name" in r

    def test_seed_with_list_init_merges(self):
        from pyjags.model import Model

        init = [{"mu": np.array([float(i)])} for i in range(3)]
        result = Model._merge_seed_into_init(42, init, 3)
        assert len(result) == 3
        for i, r in enumerate(result):
            assert r["mu"] == np.array([float(i)])
            assert ".RNG.name" in r

    def test_seed_with_rng_keys_in_init_raises(self):
        from pyjags.model import Model

        init = {".RNG.name": "base::Mersenne-Twister"}
        with pytest.raises(ValueError, match="Cannot specify both"):
            Model._merge_seed_into_init(42, init, 4)


class TestComputeDiagnosticHelpers:
    """Tests for _compute_min_ess and _compute_max_rhat_deviation."""

    @pytest.fixture()
    def converged_samples(self):
        """IID samples that should show good convergence."""
        np.random.seed(42)
        return {"mu": np.random.randn(1, 1000, 4)}

    def test_min_ess_positive(self, converged_samples):
        from pyjags.incremental_sampling import _compute_min_ess

        ess = _compute_min_ess(converged_samples)
        assert ess > 0

    def test_min_ess_high_for_iid(self, converged_samples):
        from pyjags.incremental_sampling import _compute_min_ess

        ess = _compute_min_ess(converged_samples)
        assert ess > 100

    def test_max_rhat_deviation_small_for_iid(self, converged_samples):
        from pyjags.incremental_sampling import _compute_max_rhat_deviation

        dev = _compute_max_rhat_deviation(converged_samples)
        assert dev < 0.05

    def test_max_rhat_deviation_large_for_divergent(self):
        from pyjags.incremental_sampling import _compute_max_rhat_deviation

        # Chains with very different means
        samples = {
            "mu": np.concatenate(
                [np.random.randn(1, 100, 1) + i * 10 for i in range(4)], axis=2
            )
        }
        dev = _compute_max_rhat_deviation(samples)
        assert dev > 0.1


class TestMergeAlongAxis:
    """Tests for chain_utilities._merge_along_axis."""

    def test_consecutive_merge(self):
        from pyjags.chain_utilities import _merge_along_axis

        s1 = {"x": np.ones((1, 100, 4))}
        s2 = {"x": np.ones((1, 50, 4))}
        result = _merge_along_axis(
            [s1, s2], concat_axis=1, fixed_axis=2, fixed_axis_label="chains"
        )
        assert result["x"].shape == (1, 150, 4)

    def test_parallel_merge(self):
        from pyjags.chain_utilities import _merge_along_axis

        s1 = {"x": np.ones((1, 100, 2))}
        s2 = {"x": np.ones((1, 100, 3))}
        result = _merge_along_axis(
            [s1, s2], concat_axis=2, fixed_axis=1, fixed_axis_label="chain lengths"
        )
        assert result["x"].shape == (1, 100, 5)

    def test_inconsistent_fixed_axis_raises(self):
        from pyjags.chain_utilities import _merge_along_axis

        s1 = {"x": np.ones((1, 100, 4))}
        s2 = {"x": np.ones((1, 100, 3))}
        with pytest.raises(ValueError, match="chains"):
            _merge_along_axis(
                [s1, s2], concat_axis=1, fixed_axis=2, fixed_axis_label="chains"
            )

    def test_inconsistent_param_dim_raises(self):
        from pyjags.chain_utilities import _merge_along_axis

        s1 = {"x": np.ones((1, 100, 4))}
        s2 = {"x": np.ones((2, 100, 4))}
        with pytest.raises(ValueError, match="dimension"):
            _merge_along_axis(
                [s1, s2], concat_axis=1, fixed_axis=2, fixed_axis_label="chains"
            )
