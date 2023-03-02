# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Tests for Settings module."""

import os

import numpy as np

import hvsrpy

from testtools import TestCase, unittest


class TestSettings(TestCase):

    def test_init_settings(self):
        settings = hvsrpy.HvsrTraditionalProcessingSettings()
        self.assertTrue(isinstance(settings, hvsrpy.settings.Settings))

    def test_settings_repr_and_str(self):
        settings = hvsrpy.HvsrTraditionalProcessingSettings()
        self.assertTrue(isinstance(settings.__str__(), str))
        self.assertTrue(isinstance(settings.__repr__(), str))

    def test_settings_save_and_load(self):
        settings_a = hvsrpy.HvsrTraditionalProcessingSettings()
        settings_a.window_length_in_seconds = 120.
        fname = "temp_settings.json"
        settings_a.save(fname)

        settings_b = hvsrpy.HvsrTraditionalProcessingSettings()
        settings_b.load(fname)
        os.remove(fname)
        
        self.assertDictEqual(settings_a.attr_dict, settings_b.attr_dict)

if __name__ == "__main__":
    unittest.main()
