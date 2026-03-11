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