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

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import settings
from hypothesis.extra.numpy import arrays

# ---------------------------------------------------------------------------
# Hypothesis profiles: keep tests fast
# ---------------------------------------------------------------------------
settings.register_profile("dev", max_examples=10, deadline=2000)
settings.register_profile("ci", max_examples=50, deadline=2000)
settings.load_profile("dev")


# ---------------------------------------------------------------------------
# Hypothesis strategies for PyJAGS sample dictionaries
# ---------------------------------------------------------------------------


@st.composite
def sample_dicts(
    draw,
    min_vars=1,
    max_vars=3,
    min_iterations=2,
    max_iterations=50,
    min_chains=1,
    max_chains=4,
):
    """Generate a valid PyJAGS sample dictionary.

    Each value has shape ``(param_dim, iterations, chains)`` with consistent
    iterations and chains across all variables.
    """
    n_vars = draw(st.integers(min_value=min_vars, max_value=max_vars))
    iterations = draw(st.integers(min_value=min_iterations, max_value=max_iterations))
    chains = draw(st.integers(min_value=min_chains, max_value=max_chains))
    samples = {}
    for i in range(n_vars):
        param_dim = draw(st.integers(min_value=1, max_value=5))
        arr = draw(
            arrays(
                dtype=np.float64,
                shape=(param_dim, iterations, chains),
                elements=st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
            )
        )
        samples[f"var_{i}"] = arr
    return samples


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scalar_samples():
    """Scalar variable: shape (1, iterations, chains)."""
    return {"mu": np.random.randn(1, 100, 4)}


@pytest.fixture
def vector_samples():
    """Vector variable: shape (param_dim, iterations, chains)."""
    return {"beta": np.random.randn(5, 100, 4)}


@pytest.fixture
def multi_variable_samples():
    """Multiple variables with different shapes."""
    return {
        "mu": np.random.randn(1, 100, 4),
        "beta": np.random.randn(5, 100, 4),
    }


@pytest.fixture
def tmp_hdf5(tmp_path):
    """Temporary HDF5 file path (auto-cleaned by pytest)."""
    return str(tmp_path / "samples.hdf5")
