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
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from pyjags.chain_utilities import (
    get_chain_length,
    discard_burn_in_samples,
    merge_consecutive_chains,
    merge_parallel_chains,
    extract_final_iteration_from_samples_for_initialization,
)


# ---------------------------------------------------------------------------
# get_chain_length
# ---------------------------------------------------------------------------

class TestGetChainLength:

    def test_single_variable(self):
        samples = {"mu": np.random.randn(1, 100, 4)}
        assert get_chain_length(samples) == 100

    def test_multiple_variables_consistent(self):
        samples = {
            "mu": np.random.randn(1, 50, 3),
            "beta": np.random.randn(5, 50, 3),
        }
        assert get_chain_length(samples) == 50

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            get_chain_length({})

    def test_inconsistent_lengths_raises(self):
        samples = {
            "mu": np.random.randn(1, 100, 4),
            "beta": np.random.randn(5, 50, 4),
        }
        with pytest.raises(ValueError):
            get_chain_length(samples)

    @given(
        iterations=st.integers(min_value=1, max_value=50),
        chains=st.integers(min_value=1, max_value=4),
        param_dim=st.integers(min_value=1, max_value=5),
    )
    def test_returns_iteration_dim(self, iterations, chains, param_dim):
        samples = {"x": np.random.randn(param_dim, iterations, chains)}
        assert get_chain_length(samples) == iterations


# ---------------------------------------------------------------------------
# discard_burn_in_samples
# ---------------------------------------------------------------------------

class TestDiscardBurnIn:

    def test_discard_zero(self):
        arr = np.random.randn(2, 100, 3)
        result = discard_burn_in_samples({"x": arr}, burn_in=0)
        assert result["x"].shape == (2, 100, 3)
        np.testing.assert_array_equal(result["x"], arr)

    def test_discard_some(self):
        arr = np.arange(60).reshape(2, 10, 3).astype(float)
        result = discard_burn_in_samples({"x": arr}, burn_in=3)
        assert result["x"].shape == (2, 7, 3)
        np.testing.assert_array_equal(result["x"], arr[:, 3:, :])

    def test_discard_all(self):
        arr = np.random.randn(1, 10, 2)
        result = discard_burn_in_samples({"x": arr}, burn_in=10)
        assert result["x"].shape == (1, 0, 2)

    @given(
        param_dim=st.integers(min_value=1, max_value=5),
        iterations=st.integers(min_value=2, max_value=50),
        chains=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_output_length_property(self, param_dim, iterations, chains, data):
        burn_in = data.draw(st.integers(min_value=0, max_value=iterations))
        arr = np.random.randn(param_dim, iterations, chains)
        result = discard_burn_in_samples({"x": arr}, burn_in=burn_in)
        assert result["x"].shape[1] == iterations - burn_in

    def test_multiple_variables(self):
        samples = {
            "mu": np.random.randn(1, 100, 4),
            "beta": np.random.randn(5, 100, 4),
        }
        result = discard_burn_in_samples(samples, burn_in=20)
        assert result["mu"].shape == (1, 80, 4)
        assert result["beta"].shape == (5, 80, 4)


# ---------------------------------------------------------------------------
# merge_consecutive_chains
# ---------------------------------------------------------------------------

class TestMergeConsecutiveChains:

    def test_two_chunks(self):
        a = np.random.randn(2, 30, 3)
        b = np.random.randn(2, 20, 3)
        result = merge_consecutive_chains([{"x": a}, {"x": b}])
        assert result["x"].shape == (2, 50, 3)
        np.testing.assert_array_equal(result["x"][:, :30, :], a)
        np.testing.assert_array_equal(result["x"][:, 30:, :], b)

    def test_single_chunk(self):
        arr = np.random.randn(1, 50, 2)
        result = merge_consecutive_chains([{"x": arr}])
        np.testing.assert_array_equal(result["x"], arr)

    def test_many_chunks(self):
        chunks = [{"x": np.random.randn(1, 10, 2)} for _ in range(5)]
        result = merge_consecutive_chains(chunks)
        assert result["x"].shape == (1, 50, 2)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            merge_consecutive_chains([])

    def test_none_sequence_raises(self):
        with pytest.raises(ValueError):
            merge_consecutive_chains(None)

    def test_mismatched_variables_raises(self):
        a = {"x": np.random.randn(1, 10, 2)}
        b = {"y": np.random.randn(1, 10, 2)}
        with pytest.raises(ValueError):
            merge_consecutive_chains([a, b])

    def test_mismatched_param_dim_raises(self):
        a = {"x": np.random.randn(2, 10, 3)}
        b = {"x": np.random.randn(3, 10, 3)}
        with pytest.raises(ValueError):
            merge_consecutive_chains([a, b])

    def test_mismatched_chains_raises(self):
        a = {"x": np.random.randn(1, 10, 2)}
        b = {"x": np.random.randn(1, 10, 3)}
        with pytest.raises(ValueError):
            merge_consecutive_chains([a, b])

    @given(
        param_dim=st.integers(min_value=1, max_value=5),
        chains=st.integers(min_value=1, max_value=4),
        n_chunks=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_total_iterations_property(self, param_dim, chains, n_chunks, data):
        lengths = [
            data.draw(st.integers(min_value=1, max_value=20))
            for _ in range(n_chunks)
        ]
        chunks = [
            {"x": np.random.randn(param_dim, l, chains)} for l in lengths
        ]
        result = merge_consecutive_chains(chunks)
        assert result["x"].shape == (param_dim, sum(lengths), chains)


# ---------------------------------------------------------------------------
# merge_parallel_chains
# ---------------------------------------------------------------------------

class TestMergeParallelChains:

    def test_two_sets(self):
        a = np.random.randn(2, 50, 2)
        b = np.random.randn(2, 50, 3)
        result = merge_parallel_chains([{"x": a}, {"x": b}])
        assert result["x"].shape == (2, 50, 5)
        np.testing.assert_array_equal(result["x"][:, :, :2], a)
        np.testing.assert_array_equal(result["x"][:, :, 2:], b)

    def test_single_set(self):
        arr = np.random.randn(1, 50, 4)
        result = merge_parallel_chains([{"x": arr}])
        np.testing.assert_array_equal(result["x"], arr)

    def test_mismatched_chain_lengths_raises(self):
        a = {"x": np.random.randn(1, 30, 2)}
        b = {"x": np.random.randn(1, 50, 2)}
        with pytest.raises(ValueError):
            merge_parallel_chains([a, b])

    def test_mismatched_param_dim_raises(self):
        a = {"x": np.random.randn(2, 50, 2)}
        b = {"x": np.random.randn(3, 50, 2)}
        with pytest.raises(ValueError):
            merge_parallel_chains([a, b])

    @given(
        param_dim=st.integers(min_value=1, max_value=5),
        iterations=st.integers(min_value=1, max_value=30),
        n_sets=st.integers(min_value=1, max_value=4),
        data=st.data(),
    )
    def test_total_chains_property(self, param_dim, iterations, n_sets, data):
        chain_counts = [
            data.draw(st.integers(min_value=1, max_value=4))
            for _ in range(n_sets)
        ]
        sets = [
            {"x": np.random.randn(param_dim, iterations, c)}
            for c in chain_counts
        ]
        result = merge_parallel_chains(sets)
        assert result["x"].shape == (param_dim, iterations, sum(chain_counts))


# ---------------------------------------------------------------------------
# extract_final_iteration_from_samples_for_initialization
# ---------------------------------------------------------------------------

class TestExtractFinalIteration:

    def test_scalar_variable(self):
        arr = np.arange(12).reshape(1, 4, 3).astype(float)
        result = extract_final_iteration_from_samples_for_initialization(
            {"mu": arr}, {"mu"}
        )
        assert len(result) == 3  # one dict per chain
        for chain in range(3):
            # scalar: param_dim=1, squeeze removes it
            assert result[chain]["mu"] == arr[0, -1, chain]

    def test_vector_variable(self):
        arr = np.arange(60).reshape(3, 5, 4).astype(float)
        result = extract_final_iteration_from_samples_for_initialization(
            {"beta": arr}, {"beta"}
        )
        assert len(result) == 4
        for chain in range(4):
            np.testing.assert_array_equal(
                result[chain]["beta"], arr[:, -1, chain]
            )

    def test_multiple_variables(self):
        samples = {
            "mu": np.random.randn(1, 50, 3),
            "sigma": np.random.randn(1, 50, 3),
        }
        result = extract_final_iteration_from_samples_for_initialization(
            samples, {"mu", "sigma"}
        )
        assert len(result) == 3
        for chain_init in result:
            assert "mu" in chain_init
            assert "sigma" in chain_init

    def test_subset_of_variables(self):
        samples = {
            "mu": np.random.randn(1, 50, 3),
            "sigma": np.random.randn(1, 50, 3),
        }
        result = extract_final_iteration_from_samples_for_initialization(
            samples, {"mu"}
        )
        assert len(result) == 3
        for chain_init in result:
            assert "mu" in chain_init
            assert "sigma" not in chain_init