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


class Test_HvsrRotated(TestCase):

    @classmethod
    def setUpClass(cls):
        frq = np.array([1, 2, 3, 4, 5])
        hv1 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 4, 1, 5, 1],
                                    [1, 1, 3, 1, 1],
                                    [1, 2, 4, 5, 1]]),
                          frq)
        hv2 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 1, 3, 1],
                                    [4, 5, 6, 7, 5]]),
                          frq)
        hv3 = hvsrpy.Hvsr(np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 2, 1, 1],
                                    [1, 1, 2, 1, 1],
                                    [1, 3, 1, 1, 1]]),
                          frq)
        hv4 = hvsrpy.Hvsr(np.array([[1, 1, 1, 2, 1],
                                    [1, 2, 1, 3, 1],
                                    [1, 2, 1, 1, 1]]),
                          frq)

        cls.hvrot = hvsrpy.HvsrRotated.from_iter([hv1, hv2, hv3, hv4],
                                                 [0, 45, 90, 135])
        cls.hvrot_for_rej = hvsrpy.HvsrRotated.from_iter([hv1, hv2, hv3, hv4],
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
        bad_azs = [-5, 181, 190]
        for az in bad_azs:
            self.assertRaises(ValueError, hvsrpy.HvsrRotated, hv, az)

    def test_properties(self):
        # peak_frq
        expecteds = [[3, 4, 3, 4], [3, 4, 4], [3, 3, 3, 2], [4, 4, 2]]
        returneds = self.hvrot.peak_frq

        for expected, returned in zip(expecteds, returneds):
            self.assertListEqual(expected, returned.tolist())

        # peak_amp
        expecteds = [[2, 5, 3, 5], [2, 3, 7], [2, 2, 2, 3], [2, 3, 2]]
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

        ## Multi-Window Example 1
        values = [np.array([1 ]),
                  np.array([2, 1, .5]),
                  np.array([1, 7, 2]),
                  np.array([1, 2])]

        # Normal
        returned = hvsrpy.HvsrRotated._std_factory("normal", values)
        expected = 1.783458127
        self.assertAlmostEqual(expected, returned)

        # Log-normal
        adj_vals = [np.exp(vals) for vals in values]
        returned = hvsrpy.HvsrRotated._std_factory("log-normal", adj_vals)
        expected = 1.783458127
        self.assertAlmostEqual(expected, returned)

        # Bad distribution
        self.assertRaises(NotImplementedError,
                          hvsrpy.HvsrRotated._std_factory, "exponential",
                          values)

        ## Multi-Winodw Exampe 2
        values = [np.array([1, 3]),
                  np.array([1, 5, 6]),
                  np.array([1, 1]),
                  np.array([3, 2])]
        
        # Normal - Mean
        returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
        expected = 2.375
        self.assertAlmostEqual(expected, returned)
        # Normal - Stddev
        returned = hvsrpy.HvsrRotated._std_factory("normal", values)
        expected = 1.73035188
        self.assertAlmostEqual(expected, returned)

        ## Multi-Winodw Exampe 3
        values = [np.array([1]),
                  np.array([2, 1, 0.5]),
                  np.array([1, 7, 2]),
                  np.array([1, 2])]
        
        # Normal - Mean
        returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
        expected = 1.75
        self.assertAlmostEqual(expected, returned)

        # Log-Normal - Mean
        adj_values = [ np.exp(vals) for vals in values]
        returned = hvsrpy.HvsrRotated._mean_factory("log-normal", adj_values)
        expected = 1.75
        self.assertAlmostEqual(expected, np.log(returned))

        # Normal - Stddev
        returned = hvsrpy.HvsrRotated._std_factory("normal", values)
        expected = 1.783458127
        self.assertAlmostEqual(expected, returned)

        # Log-Normal - Stddev
        adj_values = [ np.exp(vals) for vals in values]
        returned = hvsrpy.HvsrRotated._std_factory("log-normal", adj_values)
        expected = 1.783458127
        self.assertAlmostEqual(expected, returned)


    def test_mean_curve(self):
        # Normal
        returned = self.hvrot.mean_curve("normal")
        expected = np.array([1.250, 1.875, 2.063, 2.417, 1.333])
        self.assertArrayAlmostEqual(expected, returned, places=2)

        # Log-normal
        returned = self.hvrot.mean_curve("log-normal")
        expected = np.array([1.122, 1.566, 1.709, 1.830, 1.144])
        self.assertArrayAlmostEqual(expected, returned, places=2)

    def test_std_curve(self):
        # Normal
        returned = self.hvrot.std_curve("normal")
        expected = np.array([0.8611, 1.318, 1.505, 2.009, 1.148])
        self.assertArrayAlmostEqual(expected, returned, places=2)

        # Log-normal
        returned = self.hvrot.std_curve("log-normal")
        expected = np.array([0.3979, 0.5880, 0.602, 0.7457, 0.462])
        self.assertArrayAlmostEqual(expected, returned, places=2)

    # TODO (jpv): More meaningful test here.
    def test_reject_windows(self):
        expecteds = self.hvrot.peak_frq
        self.hvrot_for_rej.reject_windows(n=4)
        returneds = self.hvrot_for_rej.peak_frq
        for expected, returned in zip(expecteds, returneds):
            self.assertArrayEqual(expected, returned)

    def test_basic_stats(self):
        # Mean f0 - Frequency
        for dist, expected in zip(["normal", "log-normal"], [3.313, 3.226]):
            returned = self.hvrot.mean_f0_frq(distribution=dist)
            self.assertAlmostEqual(expected, returned, places=2)

        # Std f0 - Frequency
        for dist, expected in zip(["normal", "log-normal"], [0.7392, 0.2471]):
            returned = self.hvrot.std_f0_frq(distribution=dist)
            self.assertAlmostEqual(expected, returned, places=2)

        # Mean f0 - Amplitude
        for dist, expected in zip(["normal", "log-normal"], [3.083, 2.802]):
            returned = self.hvrot.mean_f0_amp(distribution=dist)
            self.assertAlmostEqual(expected, returned, places=2)

        # Std f0 - Amplitude
        for dist, expected in zip(["normal", "log-normal"], [1.584, 0.4282]):
            returned = self.hvrot.std_f0_amp(distribution=dist)
            self.assertAlmostEqual(expected, returned, places=2)


if __name__ == "__main__":
    unittest.main()
