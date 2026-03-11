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

from pyjags.dic import DIC, DiffDIC


# ---------------------------------------------------------------------------
# DiffDIC
# ---------------------------------------------------------------------------

class TestDiffDIC:

    def test_from_array(self):
        delta = np.array([1.0, 2.0, 3.0])
        dd = DiffDIC(delta)
        np.testing.assert_array_equal(dd.delta, delta)
        assert dd._n == 3

    def test_from_scalar(self):
        dd = DiffDIC(5.0)
        assert dd.delta == 5.0
        assert dd._n == 1

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            DiffDIC("not a number")

    def test_str_contains_difference(self):
        dd = DiffDIC(np.array([1.0, 2.0]))
        s = str(dd)
        assert "Difference:" in s
        assert "Sample standard error:" in s

    def test_repr_equals_str(self):
        dd = DiffDIC(np.array([1.0]))
        assert repr(dd) == str(dd)


# ---------------------------------------------------------------------------
# DIC
# ---------------------------------------------------------------------------

class TestDIC:

    def _make_dic(self, deviance=None, penalty=None, type="pD"):
        if deviance is None:
            deviance = np.array([10.0, 12.0])
        if penalty is None:
            penalty = np.array([2.0, 3.0])
        return DIC(deviance=deviance, penalty=penalty, type=type)

    def test_properties(self):
        dic = self._make_dic()
        np.testing.assert_array_equal(dic.deviance, np.array([10.0, 12.0]))
        np.testing.assert_array_equal(dic.penalty, np.array([2.0, 3.0]))
        assert dic.type == "pD"

    def test_construct_report(self):
        dic = self._make_dic()
        report = dic.construct_report(digits=2)
        assert "Mean deviance:" in report
        assert "penalty:" in report
        assert "Penalized deviance:" in report
        # deviance sum = 22.0, penalty sum = 5.0, penalized = 27.0
        assert "22.00" in report
        assert "5.00" in report
        assert "27.00" in report

    def test_construct_report_custom_digits(self):
        dic = self._make_dic()
        report = dic.construct_report(digits=4)
        assert "22.0000" in report

    def test_str_and_repr(self):
        dic = self._make_dic()
        assert str(dic) == dic.construct_report()
        assert repr(dic) == str(dic)

    def test_sub_same_type(self):
        dic1 = self._make_dic(
            deviance=np.array([10.0]),
            penalty=np.array([2.0]),
            type="pD",
        )
        dic2 = self._make_dic(
            deviance=np.array([8.0]),
            penalty=np.array([1.0]),
            type="pD",
        )
        diff = dic1 - dic2
        assert isinstance(diff, DiffDIC)
        # (10+2) - (8+1) = 3
        np.testing.assert_array_equal(diff.delta, np.array([3.0]))

    def test_sub_different_type_raises(self):
        dic1 = self._make_dic(type="pD")
        dic2 = self._make_dic(type="popt")
        with pytest.raises(ValueError, match="incompatible"):
            dic1 - dic2

    def test_sub_non_dic_raises(self):
        dic = self._make_dic()
        with pytest.raises(TypeError):
            dic - 5

    def test_popt_type(self):
        dic = self._make_dic(type="popt")
        assert dic.type == "popt"