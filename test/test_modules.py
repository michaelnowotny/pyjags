# Copyright (C) 2016 Tomasz Miasko
#               2026 Michael Nowotny
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


def test_get_modules_dir():
    assert pyjags.get_modules_dir() is not None


def test_module_loading():
    pyjags.load_module("basemod")
    pyjags.load_module("bugs")
    pyjags.load_module("lecuyer")
    assert pyjags.list_modules() == ["basemod", "bugs", "lecuyer"]


def test_version():
    v = pyjags.version()
    assert isinstance(v, tuple)
    assert len(v) == 3
    assert all(isinstance(x, int) for x in v)


def test_version_is_at_least_4():
    """JAGS 4.x is the minimum supported version."""
    v = pyjags.version()
    assert v[0] >= 4


def test_set_modules_dir(tmp_path):
    """set_modules_dir should update the global and get_modules_dir reflects it."""
    original = pyjags.get_modules_dir()
    try:
        custom = str(tmp_path / "custom_modules")
        pyjags.set_modules_dir(custom)
        assert pyjags.get_modules_dir() == custom
    finally:
        # Restore original so other tests aren't affected
        pyjags.set_modules_dir(original)


def test_set_modules_dir_none_triggers_auto_detect():
    """Setting modules_dir to None should trigger auto-detection again."""
    original = pyjags.get_modules_dir()
    pyjags.set_modules_dir(None)
    # Auto-detection should find the same directory
    detected = pyjags.get_modules_dir()
    assert detected is not None
    assert detected == original
