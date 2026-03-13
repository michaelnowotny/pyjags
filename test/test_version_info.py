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

import pyjags


class TestVersionInfo:
    def test_returns_dict(self):
        info = pyjags.version_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        info = pyjags.version_info()
        for key in ("pyjags", "jags", "numpy", "python"):
            assert key in info, f"Missing key: {key}"

    def test_pyjags_version_matches(self):
        info = pyjags.version_info()
        assert info["pyjags"] == pyjags.__version__

    def test_jags_version_is_dotted(self):
        info = pyjags.version_info()
        parts = info["jags"].split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_numpy_version_present(self):
        info = pyjags.version_info()
        assert info["numpy"]
        assert "." in info["numpy"]

    def test_arviz_present(self):
        info = pyjags.version_info()
        assert "arviz" in info
        # arviz is a dependency, so it should be installed
        assert info["arviz"] != "not installed"

    def test_h5py_present(self):
        info = pyjags.version_info()
        assert "h5py" in info
        assert info["h5py"] != "not installed"
