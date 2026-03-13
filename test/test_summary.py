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

import pyjags


class TestSummary:
    def test_returns_dataframe(self, multi_variable_samples):
        df = pyjags.summary(multi_variable_samples)
        assert hasattr(df, "columns")  # pandas DataFrame
        assert len(df) > 0

    def test_contains_diagnostics(self, scalar_samples):
        df = pyjags.summary(scalar_samples)
        # ArviZ summary includes these columns
        assert "mean" in df.columns
        assert "sd" in df.columns
        assert "ess_bulk" in df.columns
        assert "r_hat" in df.columns

    def test_var_names_filter(self, multi_variable_samples):
        df_all = pyjags.summary(multi_variable_samples)
        df_mu = pyjags.summary(multi_variable_samples, var_names=["mu"])
        assert len(df_mu) < len(df_all)

    def test_scalar_variable(self):
        # IID draws should have good diagnostics
        rng = np.random.default_rng(42)
        samples = {"x": rng.standard_normal((1, 1000, 4))}
        df = pyjags.summary(samples)
        assert len(df) == 1
        # IID draws should have r_hat close to 1
        # ArviZ 1.0 returns formatted strings in the summary
        r_hat = float(df["r_hat"].iloc[0])
        assert abs(r_hat - 1.0) < 0.1

    def test_accessible_from_pyjags_namespace(self):
        assert callable(pyjags.summary)
