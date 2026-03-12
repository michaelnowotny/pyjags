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

from pyjags.io import (
    save_samples_dictionary_to_file,
    load_samples_dictionary_from_file,
)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_basic_round_trip(self, tmp_path):
        path = str(tmp_path / "samples.hdf5")
        samples = {
            "mu": np.random.randn(1, 100, 4),
            "beta": np.random.randn(5, 100, 4),
        }
        save_samples_dictionary_to_file(path, samples)
        loaded = load_samples_dictionary_from_file(path)
        assert set(loaded.keys()) == set(samples.keys())
        for key in samples:
            np.testing.assert_array_equal(loaded[key], samples[key])

    def test_with_compression(self, tmp_path):
        path = str(tmp_path / "compressed.hdf5")
        samples = {"x": np.random.randn(3, 50, 2)}
        save_samples_dictionary_to_file(path, samples, compression=True)
        loaded = load_samples_dictionary_from_file(path)
        np.testing.assert_array_equal(loaded["x"], samples["x"])

    def test_without_compression(self, tmp_path):
        path = str(tmp_path / "uncompressed.hdf5")
        samples = {"x": np.random.randn(3, 50, 2)}
        save_samples_dictionary_to_file(path, samples, compression=False)
        loaded = load_samples_dictionary_from_file(path)
        np.testing.assert_array_equal(loaded["x"], samples["x"])

    def test_empty_samples(self, tmp_path):
        path = str(tmp_path / "empty.hdf5")
        save_samples_dictionary_to_file(path, {})
        loaded = load_samples_dictionary_from_file(path)
        assert loaded == {}

    def test_nonexistent_file_raises(self, tmp_path):
        path = str(tmp_path / "nonexistent.hdf5")
        with pytest.raises(OSError):
            load_samples_dictionary_from_file(path)

    @given(data=st.data())
    def test_round_trip_property(self, data):
        n_vars = data.draw(st.integers(min_value=1, max_value=3))
        iterations = data.draw(st.integers(min_value=2, max_value=20))
        chains = data.draw(st.integers(min_value=1, max_value=4))
        samples = {}
        for i in range(n_vars):
            param_dim = data.draw(st.integers(min_value=1, max_value=5))
            arr = data.draw(arrays(
                dtype=np.float64,
                shape=(param_dim, iterations, chains),
                elements=st.floats(min_value=-1e6, max_value=1e6,
                                   allow_nan=False, allow_infinity=False),
            ))
            samples[f"var_{i}"] = arr

        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".hdf5")
        os.close(fd)
        try:
            save_samples_dictionary_to_file(path, samples)
            loaded = load_samples_dictionary_from_file(path)
            assert set(loaded.keys()) == set(samples.keys())
            for key in samples:
                np.testing.assert_array_equal(loaded[key], samples[key])
        finally:
            os.unlink(path)