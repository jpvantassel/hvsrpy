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
from testing_tools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.WARNING)


class TestHvsrAzimuthal(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frequency = np.array([1, 2, 3, 4, 5], dtype=float)

        cls.amplitude_1 = np.array([[1, 1, 2, 1, 1],
                                    [1, 4, 1, 5, 1],
                                    [1, 1, 3, 1, 1],
                                    [1, 2, 4, 5, 1],
                                    [1, 1, 1, 1, 1]], dtype=float)
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

        cls.amplitude_4 = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 2, 1],
                                    [1, 2, 1, 3, 1],
                                    [1, 2, 1, 1, 1]], dtype=float)
        cls.hvsr_4 = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude_4)

        cls.azimuths = [0., 45, 90, 135]
        hvsrs = [cls.hvsr_1, cls.hvsr_2, cls.hvsr_3, cls.hvsr_4]
        cls.ahvsr = hvsrpy.HvsrAzimuthal(hvsrs,
                                         cls.azimuths)

        cls.full_path = get_full_path(__file__)

    def test_hvsrazimuthal_init_w_single_azimuth(self):
        ahvsr = hvsrpy.HvsrAzimuthal([self.hvsr_1], azimuths=[0])
        self.assertEqual(ahvsr.hvsrs[0], self.hvsr_1)
        self.assertEqual(ahvsr.azimuths[0], 0)
        self.assertEqual(ahvsr.n_azimuths, 1)

    def test_hvsrazimuthal_init_w_bad_hvsr(self):
        hvsr = np.array([1, 2, 1])
        self.assertRaises(TypeError, hvsrpy.HvsrAzimuthal, [hvsr], [0])

    def test_hvsrazimuthal_init_w_bad_azimuths(self):
        hvsr = hvsrpy.HvsrTraditional([1, 2, 3], [1, 2, 1])
        bad_azs = [-5, 181, 190]
        for az in bad_azs:
            self.assertRaises(ValueError, hvsrpy.HvsrAzimuthal, [hvsr], [az])

    def test_hvsrazimuthal_init_w_dissimilar_hvsr(self):
        hvsr_0 = hvsrpy.HvsrTraditional(self.frequency[:-1], self.amplitude_1[:, :-1])
        hvsrs = [hvsr_0, self.hvsr_1]
        azimuths = [0, 15]
        self.assertRaises(ValueError, hvsrpy.HvsrAzimuthal, hvsrs, azimuths)

    def test_hvsrazimuthal_amplitude(self):
        expected = np.vstack((self.amplitude_1, self.amplitude_2,
                              self.amplitude_3, self.amplitude_4))
        returned = np.vstack(self.ahvsr.amplitude)
        self.assertArrayAlmostEqual(expected, returned)

    def test_hvsrazimuthal_mean_fn_frequency_normal(self):
        returned = self.ahvsr.mean_fn_frequency(distribution="normal")
        self.assertAlmostEqual(returned, 3.3125, places=3)

    def test_hvsrazimuthal_mean_fn_frequency_lognormal(self):
        returned = self.ahvsr.mean_fn_frequency(distribution="lognormal")
        self.assertAlmostEqual(returned, 3.2263, places=3)

    def test_hvsrazimuthal_mean_fn_amplitude_normal(self):
        returned = self.ahvsr.mean_fn_amplitude(distribution="normal")
        self.assertAlmostEqual(returned, 3.0833, places=3)

    def test_hvsrazimuthal_mean_fn_amplitude_lognormal(self):
        returned = self.ahvsr.mean_fn_amplitude(distribution="lognormal")
        self.assertAlmostEqual(returned, 2.8020, places=3)

    def test_hvsrazimuthal_std_fn_frequency_normal(self):
        returned = self.ahvsr.std_fn_frequency(distribution="normal")
        self.assertAlmostEqual(returned, 0.7392, places=3)

    def test_hvsrazimuthal_std_fn_frequency_lognormal(self):
        returned = self.ahvsr.std_fn_frequency(distribution="lognormal")
        self.assertAlmostEqual(returned, 0.2471, places=3)

    def test_hvsrazimuthal_std_fn_amplitude_normal(self):
        returned = self.ahvsr.std_fn_amplitude(distribution="normal")
        self.assertAlmostEqual(returned, 1.5841, places=3)

    def test_hvsrazimuthal_std_fn_amplitude_lognormal(self):
        returned = self.ahvsr.std_fn_amplitude(distribution="lognormal")
        self.assertAlmostEqual(returned, 0.4282, places=3)

    def test_hvsrazimuthal_nth_std_fn_amplitude(self):
        returned = self.ahvsr.nth_std_fn_amplitude(n=1.5, distribution="normal")
        self.assertArrayAlmostEqual(returned, 5.4595, places=3)

    def test_hvsrazimuthal_nth_std_fn_frequency(self):
        returned = self.ahvsr.nth_std_fn_frequency(n=1.5, distribution="normal")
        self.assertArrayAlmostEqual(returned, 4.4213, places=3)

    def test_hvsrazimuthal_mean_curve_by_azimuth_normal(self):
        expected = np.array([[1.0000, 2.0000, 2.5000, 3.0000, 1.0000],
                             [2.0000, 2.3333, 3.0000, 3.6667, 2.3333],
                             [1.0000, 1.5000, 1.7500, 1.0000, 1.0000],
                             [1.0000, 1.6667, 1.0000, 2.0000, 1.0000]])
        returned = self.ahvsr.mean_curve_by_azimuth(distribution="normal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_mean_curve_peak_by_azimuth_normal(self):
        f_peaks, a_peaks = self.ahvsr.mean_curve_peak_by_azimuth(distribution="normal")
        self.assertArrayAlmostEqual(f_peaks,
                                    np.array([4, 4, 3, 4]), places=3)
        self.assertArrayAlmostEqual(a_peaks,
                                    np.array([3.0, 3.6667, 1.750, 2.0]), places=3)

    def test_hvsrazimuthal_mean_curve_by_azimuth_lognormal(self):
        expected = np.array([[1.0000, 1.6818, 2.2134, 2.2361, 1.0000],
                             [1.5874, 1.7100, 2.2894, 2.7589, 1.7100],
                             [1.0000, 1.3161, 1.6818, 1.0000, 1.0000],
                             [1.0000, 1.5874, 1.0000, 1.8171, 1.0000]])
        returned = self.ahvsr.mean_curve_by_azimuth(distribution="lognormal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_mean_curve_peak_by_azimuth_lognormal(self):
        f_peaks, a_peaks = self.ahvsr.mean_curve_peak_by_azimuth(distribution="lognormal")
        self.assertArrayAlmostEqual(f_peaks,
                                    np.array([4, 4, 3, 4]), places=3)
        self.assertArrayAlmostEqual(a_peaks,
                                    np.array([2.2361, 2.7589, 1.6818, 1.8171]), places=3)

    def test_hvsrazimuthal_mean_curve_normal(self):
        expected = np.array([[1.2500, 1.8750, 2.0625, 2.4167, 1.3333]])
        returned = self.ahvsr.mean_curve(distribution="normal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_mean_curve_lognormal(self):
        expected = np.array([[1.1225, 1.5656, 1.7086, 1.8298, 1.1435]])
        returned = self.ahvsr.mean_curve(distribution="lognormal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_std_curve_normal(self):
        expected = np.array([[0.8611, 1.3176, 1.5051, 2.0093, 1.1482]])
        returned = self.ahvsr.std_curve(distribution="normal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_std_curve_lognormal(self):
        expected = np.array([[0.3979, 0.5880, 0.6023, 0.7457, 0.4620]])
        returned = self.ahvsr.std_curve(distribution="lognormal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_nth_std_curve_normal(self):
        expected = np.array([2.1111, 3.1926, 3.5676, 4.4260, 2.4815])
        returned = self.ahvsr.nth_std_curve(n=1, distribution="normal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_nth_std_curve_lognormal(self):
        expected = np.array([1.6711, 2.8188, 3.1204, 3.8569, 1.8150])
        returned = self.ahvsr.nth_std_curve(n=1, distribution="lognormal")
        self.assertArrayAlmostEqual(expected, returned, places=3)

    def test_hvsrazimuthal_cov_fn_normal(self):
        cov = self.ahvsr.cov_fn(distribution="normal")
        self.assertAlmostEqual(np.sqrt(cov[0, 0]),
                               self.ahvsr.std_fn_frequency(distribution="normal"))
        self.assertAlmostEqual(np.sqrt(cov[1, 1]),
                               self.ahvsr.std_fn_amplitude(distribution="normal"))

    def test_hvsrazimuthal_cov_fn_lognormal(self):
        cov = self.ahvsr.cov_fn(distribution="lognormal")
        self.assertAlmostEqual(np.sqrt(cov[0, 0]),
                               self.ahvsr.std_fn_frequency(distribution="lognormal"))
        self.assertAlmostEqual(np.sqrt(cov[1, 1]),
                               self.ahvsr.std_fn_amplitude(distribution="lognormal"))

    def test_hvsrazimuthal_mean_curve_peak_normal(self):
        f_peak, a_peak = self.ahvsr.mean_curve_peak(distribution="normal")
        self.assertAlmostEqual(f_peak, 4, places=3)
        self.assertAlmostEqual(a_peak, 2.4167, places=3)

    def test_hvsrazimuthal_mean_curve_peak_lognormal(self):
        f_peak, a_peak = self.ahvsr.mean_curve_peak(distribution="lognormal")
        self.assertAlmostEqual(f_peak, 4, places=3)
        self.assertAlmostEqual(a_peak, 1.8298, places=3)

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

    def test_hvsrazimuthal_is_similar_and_equal(self):
        a = self.ahvsr
        b = hvsrpy.HvsrAzimuthal([self.hvsr_1, self.hvsr_2, self.hvsr_3, self.hvsr_4],
                                 self.azimuths)
        c = np.array([1, 2, 3])
        d = hvsrpy.HvsrAzimuthal([self.hvsr_1], [self.azimuths[0]])
        e = hvsrpy.HvsrAzimuthal([self.hvsr_4, self.hvsr_2, self.hvsr_3, self.hvsr_1],
                                 self.azimuths)
        f = hvsrpy.HvsrAzimuthal([self.hvsr_1, self.hvsr_2, self.hvsr_3, self.hvsr_4],
                                 self.azimuths[::-1])
        custom_hvsr = hvsrpy.HvsrTraditional(self.frequency[:-1], self.amplitude_1[:,:-1])
        g = hvsrpy.HvsrAzimuthal([custom_hvsr]*4,
                                 self.azimuths[::-1])

        self.assertTrue(a == b)
        self.assertTrue(a.is_similar(b))
        self.assertFalse(a == c)
        self.assertFalse(a.is_similar(c))
        self.assertFalse(a == d)
        self.assertFalse(a.is_similar(d))
        self.assertFalse(a == e)
        self.assertTrue(a.is_similar(e))
        self.assertFalse(a == f)
        self.assertFalse(a.is_similar(f))
        self.assertFalse(a == g)
        self.assertFalse(a.is_similar(g))

    def test_hvsrazimuthal_str_repr(self):
        ahvsr = self.ahvsr
        self.assertTrue(isinstance(ahvsr.__str__(), str))
        self.assertTrue(isinstance(ahvsr.__repr__(), str))


if __name__ == "__main__":
    unittest.main()
