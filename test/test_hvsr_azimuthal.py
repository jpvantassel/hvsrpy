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

"""Tests for HvsrAzimuthal object."""

import logging

import numpy as np

import hvsrpy
from testtools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.WARNING)


class TestHvsrAzimuthal(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frequency = np.array([1, 2, 3, 4, 5], dtype=float)

        cls.amplitude_1 = np.array([[1, 1, 2, 1, 1],
                                    [1, 4, 1, 5, 1],
                                    [1, 1, 3, 1, 1],
                                    [1, 2, 4, 5, 1]], dtype=float)
        cls.hvsr_1 = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude_1)

        cls.amplitude_2 = np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 1, 3, 1],
                                    [4, 5, 6, 7, 5]], dtype=float)
        cls.hvsr_2 = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude_2)

        cls.amplitude_3 = np.array([[1, 1, 2, 1, 1],
                                    [1, 1, 2, 1, 1],
                                    [1, 1, 2, 1, 1],
                                    [1, 3, 1, 1, 1]], dtype=float)
        cls.hvsr_3 = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude_3)

        cls.amplitude_4 = np.array([[1, 1, 1, 2, 1],
                                    [1, 2, 1, 3, 1],
                                    [1, 2, 1, 1, 1]], dtype=float)
        cls.hvsr_4 = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude_4)

        cls.azimuths = [0., 45, 90, 135]
        hvsrs = [cls.hvsr_1, cls.hvsr_2, cls.hvsr_3, cls.hvsr_4]
        cls.ahvsr = hvsrpy.HvsrAzimuthal(hvsrs,
                                         cls.azimuths)

        cls.full_path = get_full_path(__file__)

    def test_init_hvsrazimuthal_single_azimuth(self):
        ahvsr = hvsrpy.HvsrAzimuthal([self.hvsr_1], azimuths=[0])
        self.assertEqual(ahvsr.hvsrs[0], self.hvsr_1)
        self.assertEqual(ahvsr.azimuths[0], 0)
        self.assertEqual(ahvsr.n_azimuths, 1)

    def test_init_hvsrazimuthal_bad_hvsr(self):
        hvsr = np.array([1, 2, 1])
        self.assertRaises(TypeError, hvsrpy.HvsrAzimuthal, [hvsr], [0])

    def test_init_hvsrazimuthal_bad_azimuths(self):
        hvsr = hvsrpy.HvsrTraditional([1, 2, 3], [1, 2, 1])
        bad_azs = [-5, 181, 190]
        for az in bad_azs:
            self.assertRaises(ValueError, hvsrpy.HvsrAzimuthal, [hvsr], [az])

    # def test_properties(self):
    #     # frq
    #     expected = self.frq
    #     returned = self.hvrot.frq
    #     self.assertArrayEqual(expected, returned)

    #     # peak_frq
    #     expecteds = [[3, 4, 3, 4], [3, 4, 4], [3, 3, 3, 2], [4, 4, 2]]
    #     returneds = self.hvrot.peak_frq

    #     for expected, returned in zip(expecteds, returneds):
    #         self.assertListEqual(expected, returned.tolist())

    #     # peak_amp
    #     expecteds = [[2, 5, 3, 5], [2, 3, 7], [2, 2, 2, 3], [2, 3, 2]]
    #     returneds = self.hvrot.peak_amp

    #     for expected, returned in zip(expecteds, returneds):
    #         self.assertListEqual(expected, returned.tolist())

    #     # azimuth_count
    #     self.assertEqual(4, self.hvrot.azimuth_count)

    def test_mean_curve_by_azimuth(self):
        expected = np.empty((len(self.azimuths), len(self.frequency)))
        for cnt in range(len(self.azimuths)):
            expected[cnt, :] = np.nanmean(getattr(self, f"amplitude_{cnt+1}"), axis=0)
        returned = self.ahvsr.mean_curve_by_azimuth(distribution="normal")
        self.assertArrayEqual(expected, returned)

#     def test_mean_factory(self):
#         # Single-Window
#         values = [np.array([1, 2, 3, 4, 5])]

#         # Normal
#         returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
#         expected = np.mean([np.mean(x) for x in values])
#         self.assertEqual(expected, returned)

#         # Log-normal
#         returned = hvsrpy.HvsrRotated._mean_factory("lognormal", values)
#         expected = np.exp(np.mean([np.mean(np.log(x)) for x in values]))
#         self.assertEqual(expected, returned)

#         # Multi-Window
#         values = [np.array([1, 2, 3, 4, 5]),
#                   np.array([2, 1, .5, 1]),
#                   np.array([1, 4, 5, 7, 1, 2, 3, 4, 5]),
#                   np.array([1, 2, 1, 2, 1, 2])]
#         # Normal
#         returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
#         expected = np.mean([np.mean(x) for x in values])
#         self.assertEqual(expected, returned)
#         # Log-normal
#         returned = hvsrpy.HvsrRotated._mean_factory("lognormal", values)
#         expected = np.exp(np.mean([np.mean(np.log(x)) for x in values]))
#         self.assertEqual(expected, returned)

#         # Bad distribution
#         self.assertRaises(NotImplementedError,
#                           hvsrpy.HvsrRotated._mean_factory, "exponential",
#                           values)

#     def test_std_factory(self):
#         # Single-Window
#         values = [np.array([1, 2, 3, 4, 5])]

#         # Normal
#         returned = hvsrpy.HvsrRotated._std_factory("normal", values)
#         expected = np.std(values[0], ddof=1)
#         self.assertEqual(expected, returned)

#         # Log-normal
#         returned = hvsrpy.HvsrRotated._std_factory("lognormal", values)
#         expected = np.std(np.log(values[0]), ddof=1)
#         self.assertAlmostEqual(expected, returned)

#         # Multi-Window Example 1
#         values = [np.array([1]),
#                   np.array([2, 1, .5]),
#                   np.array([1, 7, 2]),
#                   np.array([1, 2])]

#         # Normal
#         returned = hvsrpy.HvsrRotated._std_factory("normal", values)
#         expected = 1.783458127
#         self.assertAlmostEqual(expected, returned)

#         # Log-normal
#         adj_vals = [np.exp(vals) for vals in values]
#         returned = hvsrpy.HvsrRotated._std_factory("lognormal", adj_vals)
#         expected = 1.783458127
#         self.assertAlmostEqual(expected, returned)

#         # Bad distribution
#         self.assertRaises(NotImplementedError,
#                           hvsrpy.HvsrRotated._std_factory, "exponential",
#                           values)

#         # Multi-Winodw Exampe 2
#         values = [np.array([1, 3]),
#                   np.array([1, 5, 6]),
#                   np.array([1, 1]),
#                   np.array([3, 2])]

#         # Normal - Mean
#         returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
#         expected = 2.375
#         self.assertAlmostEqual(expected, returned)
#         # Normal - Stddev
#         returned = hvsrpy.HvsrRotated._std_factory("normal", values)
#         expected = 1.73035188
#         self.assertAlmostEqual(expected, returned)

#         # Multi-Winodw Exampe 3
#         values = [np.array([1]),
#                   np.array([2, 1, 0.5]),
#                   np.array([1, 7, 2]),
#                   np.array([1, 2])]

#         # Normal - Mean
#         returned = hvsrpy.HvsrRotated._mean_factory("normal", values)
#         expected = 1.75
#         self.assertAlmostEqual(expected, returned)

#         # Log-Normal - Mean
#         adj_values = [np.exp(vals) for vals in values]
#         returned = hvsrpy.HvsrRotated._mean_factory("lognormal", adj_values)
#         expected = 1.75
#         self.assertAlmostEqual(expected, np.log(returned))

#         # Normal - Stddev
#         returned = hvsrpy.HvsrRotated._std_factory("normal", values)
#         expected = 1.783458127
#         self.assertAlmostEqual(expected, returned)

#         # Log-Normal - Stddev
#         adj_values = [np.exp(vals) for vals in values]
#         returned = hvsrpy.HvsrRotated._std_factory("lognormal", adj_values)
#         expected = 1.783458127
#         self.assertAlmostEqual(expected, returned)

    def test_hvsrazimuthal_mean_curve_normal(self):
        returned = self.ahvsr.mean_curve("normal")
        expected = np.array([1.250, 1.875, 2.063, 2.417, 1.333])
        self.assertArrayAlmostEqual(expected, returned, places=2)

    def test_hvsrazimuthal_mean_curve_normal_peak(self):
        f_peak, a_peak = self.ahvsr.mean_curve_peak("normal")
        self.assertAlmostEqual(f_peak, 4)
        self.assertAlmostEqual(a_peak, 2.417, places=2)

    def test_hvsrazimuthal_mean_curve_lognormal(self):
        returned = self.ahvsr.mean_curve("lognormal")
        expected = np.array([1.122, 1.566, 1.709, 1.830, 1.144])
        self.assertArrayAlmostEqual(expected, returned, places=2)

    def test_hvsrazimuthal_mean_curve_normal_peak(self):
        f_peak, a_peak = self.ahvsr.mean_curve_peak("lognormal")
        self.assertAlmostEqual(f_peak, 4)
        self.assertAlmostEqual(a_peak, 1.830, places=2)

#     def test_std_curve(self):
#         # Normal
#         returned = self.hvrot.std_curve("normal")
#         expected = np.array([0.8611, 1.318, 1.505, 2.009, 1.148])
#         self.assertArrayAlmostEqual(expected, returned, places=2)

#         # Lognormal
#         returned = self.hvrot.std_curve("lognormal")
#         expected = np.array([0.3979, 0.5880, 0.602, 0.7457, 0.462])
#         self.assertArrayAlmostEqual(expected, returned, places=2)

#     def test_reject_windows(self):
#         expecteds = self.hvrot.peak_frq
#         self.hvrot_for_rej.reject_windows(n=4)
#         returneds = self.hvrot_for_rej.peak_frq
#         for expected, returned in zip(expecteds, returneds):
#             self.assertArrayEqual(expected, returned)

#     def test_nstd(self):
#         for n in [-2, -1, -0.5, 0.5, 1, 2]:
#             # Mean Curve
#             mean = self.hvrot.mean_curve("lognormal")
#             stddev = self.hvrot.std_curve("lognormal")
#             expected = np.exp(np.log(mean) + n*stddev)
#             returned = self.hvrot.nstd_curve(n, "lognormal")
#             self.assertArrayEqual(expected, returned)

#             # f0_frq
#             mean = self.hvrot.mean_f0_frq("lognormal")
#             stddev = self.hvrot.std_f0_frq("lognormal")
#             expected = np.exp(np.log(mean) + n*stddev)
#             returned = self.hvrot.nstd_f0_frq(n, "lognormal")
#             self.assertEqual(expected, returned)

#     def test_basic_stats(self):
#         # Mean f0 - Frequency
#         for dist, expected in zip(["normal", "lognormal"], [3.313, 3.226]):
#             returned = self.hvrot.mean_f0_frq(distribution=dist)
#             self.assertAlmostEqual(expected, returned, places=2)

#         # Std f0 - Frequency
#         for dist, expected in zip(["normal", "lognormal"], [0.7392, 0.2471]):
#             returned = self.hvrot.std_f0_frq(distribution=dist)
#             self.assertAlmostEqual(expected, returned, places=2)

#         # Mean f0 - Amplitude
#         for dist, expected in zip(["normal", "lognormal"], [3.083, 2.802]):
#             returned = self.hvrot.mean_f0_amp(distribution=dist)
#             self.assertAlmostEqual(expected, returned, places=2)

#         # Std f0 - Amplitude
#         for dist, expected in zip(["normal", "lognormal"], [1.584, 0.4282]):
#             returned = self.hvrot.std_f0_amp(distribution=dist)
#             self.assertAlmostEqual(expected, returned, places=2)

    def test_hvsrazimuthal_str_repr(self):
        ahvsr = self.ahvsr
        self.assertTrue(isinstance(ahvsr.__str__(), str))
        self.assertTrue(isinstance(ahvsr.__repr__(), str))


if __name__ == "__main__":
    unittest.main()
