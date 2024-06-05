# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Tests for HvsrCurve object."""

import logging

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase

logging.basicConfig(level=logging.ERROR)


class TestHvsrCurve(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frequency = [1, 2, 3, 4, 5, 6]
        cls.amplitude = [1, 3, 1, 1, 6, 1]
        cls.hvsr = hvsrpy.HvsrCurve(cls.frequency, cls.amplitude)

    def test_hvsrcurve_init_with_invalid_amplitude_type(self):
        frequency = np.ndarray([1, 2, 3])
        amplitude = "abc"
        self.assertRaises(TypeError, hvsrpy.HvsrCurve, frequency, amplitude)

    def test_hvsrcurve_init_with_negative_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, -1, 3])
        self.assertRaises(ValueError, hvsrpy.HvsrCurve, amp, frq)

    def test_hvsrcurve_init_with_nan_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, 2, np.nan])
        self.assertRaises(ValueError, hvsrpy.HvsrCurve, amp, frq)

    def test_hvsrcurve_init_with_incompatible_frequency_and_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, 2])
        self.assertRaises(ValueError, hvsrpy.HvsrCurve, amp, frq)

    def test_hvsrcurve_find_peak_unbounded(self):
        self.assertEqual(self.hvsr.peak_frequency, 5)
        self.assertEqual(self.hvsr.peak_amplitude, 6)

    def test_hvsrcurve_find_peak_bounded(self):
        hvsr = hvsrpy.HvsrCurve(self.frequency, self.amplitude)
        hvsr.update_peaks_bounded(search_range_in_hz=(0.9, 4.1))
        self.assertEqual(hvsr.peak_frequency, 2)
        self.assertEqual(hvsr.peak_amplitude, 3)

    def test_hvsrcurve_find_peak_half_bounded_lower(self):
        hvsr = hvsrpy.HvsrCurve(self.frequency, self.amplitude)
        hvsr.update_peaks_bounded(search_range_in_hz=(None, 4))
        self.assertEqual(hvsr.peak_frequency, 2)
        self.assertEqual(hvsr.peak_amplitude, 3)

    def test_hvsrcurve_find_peak_half_bounded_upper(self):
        hvsr = hvsrpy.HvsrCurve(self.frequency, self.amplitude)
        hvsr.update_peaks_bounded(search_range_in_hz=(4, None))
        self.assertEqual(hvsr.peak_frequency, 5)
        self.assertEqual(hvsr.peak_amplitude, 6)

    def test_hvsrcurve_find_peak_flat(self):
        hvsr = hvsrpy.HvsrCurve([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        self.assertTrue(np.isnan(hvsr.peak_frequency))
        self.assertTrue(np.isnan(hvsr.peak_amplitude))

    def test_is_similar(self):
        a = hvsrpy.HvsrCurve([1, 2, 3], [1, 2, 1])
        b = np.array([1, 2, 3])
        c = hvsrpy.HvsrCurve([1, 2, 3, 4], [1, 2, 1, 1])
        d = hvsrpy.HvsrCurve([1.1, 2.1, 3.1], [1, 2, 1])

        self.assertTrue(a.is_similar(a))
        self.assertFalse(a.is_similar(b))
        self.assertFalse(a.is_similar(c))
        self.assertFalse(a.is_similar(d))


if __name__ == "__main__":
    unittest.main()
