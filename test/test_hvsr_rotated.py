# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""Tests for HvsrRotated object."""

import numpy as np
import hvsrpy
from testtools import unittest, TestCase
import logging
logging.basicConfig(level=logging.WARNING)


class Test_HvsrRotated(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        frq = np.array([1, 2, 3, 4, 5])
        hv1 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 4, 1, 5, 1]]),
                          frq)
        hv2 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 1, 3, 1]]),
                          frq)
        hv3 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 2, 1, 1]]),
                          frq)
        hv4 = hvsrpy.Hvsr(np.array([[1, 1, 1, 2, 1],
                                    [1, 2, 1, 3, 1]]),
                          frq)

        cls.hvrot = hvsrpy.HvsrRotated.from_iter([hv1, hv2, hv3, hv4],
                                                 [0, 45, 90, 135])

    def test_init(self):
        # Simple case
        frq = np.array([1, 2, 3])
        hv = hvsrpy.Hvsr(np.array([1, 2, 1]), frq)
        az = 20
        hvrot = hvsrpy.HvsrRotated(hv, az)
        self.assertEqual(hv, hvrot.hvsrs[0])
        self.assertEqual(az, hvrot.azimuths[0])
        self.assertEqual(1, hvrot.azimuth_count)

        # Bad hv
        hv = np.array([1, 2, 1])
        self.assertRaises(TypeError, hvsrpy.HvsrRotated, hv, az)

        # Bad azimuths
        hv = hvsrpy.Hvsr(np.array([1, 2, 1]), frq)
        bad_azs = [-5, 180, 190]
        for az in bad_azs:
            self.assertRaises(ValueError, hvsrpy.HvsrRotated, hv, az)

    def test_properties(self):
        # peak_frq
        expecteds = [[3, 4], [3, 4], [3, 3], [4, 4]]
        returneds = self.hvrot.peak_frq

        for expected, returned in zip(expecteds, returneds):
            self.assertListEqual(expected, returned.tolist())

        # peak_amp
        expecteds = [[2, 5], [2, 3], [2, 2], [2, 3]]
        returneds = self.hvrot.peak_amp

        for expected, returned in zip(expecteds, returneds):
            self.assertListEqual(expected, returned.tolist())

        # azimuth_count
        self.assertEqual(4, self.hvrot.azimuth_count)

    def test_mean_factory(self):
        # Single-Window
        values = [np.array([1, 2, 3, 4, 5])]
        # Normal
        returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
        expected = np.mean([np.mean(x) for x in values])
        self.assertEqual(expected, returned)
        # Log-normal
        returned = hvsrpy.HvsrRotated._mean_factory("log-normal", values)
        expected = np.exp(np.mean([np.mean(np.log(x)) for x in values]))
        self.assertEqual(expected, returned)

        # Multi-Window
        values = [np.array([1, 2, 3, 4, 5]),
                  np.array([2, 1, .5, 1]),
                  np.array([1, 4, 5, 7, 1, 2, 3, 4, 5]),
                  np.array([1, 2, 1, 2, 1, 2])]
        # Normal
        returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
        expected = np.mean([np.mean(x) for x in values])
        self.assertEqual(expected, returned)
        # Log-normal
        returned = hvsrpy.HvsrRotated._mean_factory("log-normal", values)
        expected = np.exp(np.mean([np.mean(np.log(x)) for x in values]))
        self.assertEqual(expected, returned)

        # Bad distribution
        self.assertRaises(NotImplementedError,
                          hvsrpy.HvsrRotated._mean_factory, "exponential",
                          values)

    def test_std_factory(self):
        # Single-Window
        values = [np.array([1, 2, 3, 4, 5])]
        # Normal
        returned = hvsrpy.HvsrRotated._std_factory("normal", values)
        expected = np.std(values[0], ddof=1)
        self.assertEqual(expected, returned)
        # Log-normal
        returned = hvsrpy.HvsrRotated._std_factory("log-normal", values)
        expected = np.std(np.log(values[0]), ddof=1)
        self.assertAlmostEqual(expected, returned)
        
        # Multi-Window
        values = [np.array([1, 2, 3, 4, 5]),
                  np.array([2, 1, .5, 1]),
                  np.array([1, 4, 5, 7, 1, 2, 3, 4, 5]),
                  np.array([1, 2, 1, 2, 1, 2])]
        # Normal
        returned = hvsrpy.HvsrRotated._std_factory("normal", values)
        expected = 1.635604659
        self.assertAlmostEqual(expected, returned)
        # Log-normal
        returned = hvsrpy.HvsrRotated._std_factory("log-normal", values)
        expected = 0.707120567
        self.assertAlmostEqual(expected, returned)

        # Bad distribution
        self.assertRaises(NotImplementedError,
                          hvsrpy.HvsrRotated._std_factory, "exponential",
                          values)


if __name__ == "__main__":
    unittest.main()
