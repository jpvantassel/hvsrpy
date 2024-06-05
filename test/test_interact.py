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

"""Tests for the interact module."""

import logging

import numpy as np

import hvsrpy.interact as interact
from testing_tools import unittest, TestCase

logging.basicConfig(level=logging.ERROR)


class TestInteract(TestCase):

    def test_conversion_relative_to_absolute_linear(self):
        relative = 0.75
        range_absolute = 0, 100
        absolute = interact._relative_to_absolute(relative, range_absolute, scale="linear")
        self.assertAlmostEqual(absolute, 75)

    def test_conversion_relative_to_absolute_log(self):
        relative = 0.5
        range_absolute = 10, 100
        absolute = interact._relative_to_absolute(relative, range_absolute, scale="log")
        self.assertAlmostEqual(absolute, 31.6227766, places=4)

    def test_conversion_absolute_to_relative_linear(self):
        absolute = 75
        range_absolute = 0, 100
        relative = interact._absolute_to_relative(absolute, range_absolute, scale="linear")
        self.assertAlmostEqual(relative, 0.75)

    def test_conversion_absolute_to_relative_log(self):
        absolute = 31.6227766
        range_absolute = 10, 100
        relative = interact._absolute_to_relative(absolute, range_absolute, scale="log")
        self.assertAlmostEqual(relative, 0.5, places=4)

if __name__ == "__main__":
    unittest.main()