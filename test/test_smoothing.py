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

"""Test smoothing algorithms."""

import logging

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestSmoothing(TestCase):

    @classmethod
    def setUpClass(cls):
        n_points = 100
        cls.frequency = np.linspace(3, 30, n_points)
        cls.amplitude = np.zeros_like(cls.frequency)
        time = np.linspace(0, 1, n_points)
        cls.amplitude += 5*np.sin(2*np.pi*0.5*time)
        cls.amplitude += 0.5*np.sin(2*np.pi*80*time)
        cls.amplitude = np.atleast_2d(cls.amplitude)

    def test_smoothing_konno_and_ohmachi(self):
        value = hvsrpy.smoothing.konno_and_ohmachi(self.frequency, self.amplitude,
                                                   fcs=np.array([10.]), bandwidth=40)
        self.assertAlmostEqual(value[0, 0], 3.6707, places=3)

    def test_smoothing_parzen(self):
        value = hvsrpy.smoothing.parzen(self.frequency, self.amplitude,
                                        fcs=np.array([10.]), bandwidth=0.5)
        self.assertAlmostEqual(value[0, 0], 3.8134, places=3)

    def test_smoothing_savitzky_and_golay(self):
        value = hvsrpy.smoothing.savitzky_and_golay(self.frequency, self.amplitude,
                                                    fcs=np.array([9.]), bandwidth=9)
        self.assertAlmostEqual(value[0, 0], 3.1929, places=3)

    def test_smoothing_linear_rectangular(self):
        value = hvsrpy.smoothing.linear_rectangular(self.frequency, self.amplitude,
                                                    fcs=np.array([10]), bandwidth=1)
        self.assertAlmostEqual(value[0, 0], 3.7072, places=3)

    def test_smoothing_log_rectangular(self):
        value = hvsrpy.smoothing.log_rectangular(self.frequency, self.amplitude,
                                                    fcs=np.array([10]), bandwidth=0.15)
        self.assertAlmostEqual(value[0, 0], 3.6514, places=3)

    def test_smoothing_linear_triangular(self):
        value = hvsrpy.smoothing.linear_triangular(self.frequency, self.amplitude,
                                                   fcs=np.array([10]), bandwidth=1)
        self.assertAlmostEqual(value[0, 0], 3.7680, places=3)

    def test_smoothing_log_triangular(self):
        value = hvsrpy.smoothing.log_triangular(self.frequency, self.amplitude,
                                                   fcs=np.array([10]), bandwidth=0.15)
        self.assertAlmostEqual(value[0, 0], 3.6593, places=3)


if __name__ == "__main__":
    unittest.main()
