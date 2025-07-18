# Copyright (C) 2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import unittest

import pyjags


class TestModules(unittest.TestCase):
    def test_get_modules_dir(self):
        self.assertIsNotNone(pyjags.get_modules_dir())

    def test_module_loading(self):
        pyjags.load_module("basemod")
        pyjags.load_module("bugs")
        pyjags.load_module("lecuyer")

        self.assertEqual(["basemod", "bugs", "lecuyer"], pyjags.list_modules())


if __name__ == "__main__":
    unittest.main()
