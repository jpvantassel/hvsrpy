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

"""Tests for HvsrTraditional object."""

import logging

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase

logging.basicConfig(level=logging.ERROR)


class TestHvsrTraditional(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frequency = [1, 2, 3, 4, 5, 6]
        cls.amplitude = [[1, 3, 1, 1, 1, 1],
                         [1, 1, 2, 5, 1, 1],
                         [1, 1, 1, 1, 6, 1]]
        cls.hvsr = hvsrpy.HvsrTraditional(cls.frequency, cls.amplitude)

    def test_hvsrtraditional_init_with_invalid_amplitude_type(self):
        frequency = np.ndarray([1, 2, 3])
        amplitude = "abc"
        self.assertRaises(TypeError, hvsrpy.HvsrTraditional, frequency, amplitude)

    def test_hvsrtraditional_init_with_negative_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, -1, 3])
        self.assertRaises(ValueError, hvsrpy.HvsrTraditional, amp, frq)

    def test_hvsrtraditional_init_with_nan_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, 2, np.nan])
        self.assertRaises(ValueError, hvsrpy.HvsrTraditional, amp, frq)

    def test_hvsrtraditional_init_with_incompatible_frequency_and_amplitude(self):
        frq = np.array([1, 2, 3])
        amp = np.array([1, 2])
        self.assertRaises(ValueError, hvsrpy.HvsrTraditional, amp, frq)

    def test_hvsrtraditional_from_hvsrcurves(self):
        hvsrs = []
        for amplitude in self.amplitude:
            hvsr = hvsrpy.HvsrCurve(self.frequency, amplitude)
            hvsrs.append(hvsr)
        hvsr = hvsrpy.HvsrTraditional.from_hvsr_curves(hvsrs)
        self.assertEqual(hvsr, self.hvsr)

    def test_hvsrtraditional_from_hvsrcurves_incompatible(self):
        hvsr_a = hvsrpy.HvsrCurve(self.frequency, self.amplitude[0])
        hvsr_b = hvsrpy.HvsrCurve([1, 2, 3, 4, 5], [1, 1, 2, 1, 1])
        hvsrs = [hvsr_a, hvsr_b]
        self.assertRaises(ValueError, hvsrpy.HvsrTraditional.from_hvsr_curves, hvsrs)

    def test_hvsrtraditional_peak_frequency(self):
        expected = np.array([2., 4, 5])
        returned = self.hvsr.peak_frequencies
        self.assertArrayEqual(expected, returned)

    def test_hvsrtraditional_peak_frequency_with_rejection(self):
        hvsr = hvsrpy.HvsrTraditional(self.frequency, self.amplitude)
        hvsr.update_peaks_bounded(search_range_in_hz=(2.6, None))
        expected = np.array([4., 5])
        returned = hvsr.peak_frequencies
        self.assertArrayEqual(expected, returned)

    def test_hvsrtraditional_peak_amplitude(self):
        expected = np.array([3., 5, 6])
        returned = self.hvsr.peak_amplitudes
        self.assertArrayEqual(expected, returned)

    def test_hvsrtraditional_peak_amplitude_with_rejection(self):
        hvsr = hvsrpy.HvsrTraditional(self.frequency, self.amplitude)
        hvsr.update_peaks_bounded(search_range_in_hz=(2.9, None))
        expected = np.array([5., 6])
        returned = hvsr.peak_amplitudes
        self.assertArrayEqual(expected, returned)

    def test_hvsrtraditional_mean_and_std_fn_frequency(self):
        frequency = np.arange(0, 10, 1)
        amplitude = np.ones((10, 10))
        col_idx = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amplitude[np.arange(0, 10, 1), col_idx] = 2
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        self.assertEqual(hvsr.mean_fn_frequency(distribution="lognormal"),
                         np.exp(np.mean(np.log(frequency[col_idx]))))
        self.assertEqual(hvsr.mean_fn_frequency(distribution="normal"),
                         np.mean(frequency[col_idx]))

        self.assertEqual(hvsr.std_fn_frequency(distribution="lognormal"),
                         np.std(np.log(frequency[col_idx]), ddof=1))
        self.assertEqual(hvsr.std_fn_frequency(distribution="normal"),
                         np.std(frequency[col_idx], ddof=1))

    def test_hvsrtraditional_mean_and_std_fn_frequency_invalid_distribution(self):
        self.assertRaises(NotImplementedError, self.hvsr.mean_fn_frequency,
                          distribution="exponential")
        self.assertRaises(NotImplementedError, self.hvsr.std_fn_frequency,
                          distribution="exponential")

    def test_hvsrtraditional_mean_and_std_fn_amplitude(self):
        frequency = np.arange(0, 10, 1)
        amplitude = np.zeros((10, 10))
        col_idx = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        peak_amp = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amplitude[np.arange(10), col_idx] = peak_amp
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        self.assertEqual(hvsr.mean_fn_amplitude(distribution="lognormal"),
                         np.exp(np.mean(np.log(peak_amp))))
        self.assertEqual(hvsr.mean_fn_amplitude(distribution="normal"),
                         np.mean(peak_amp))

        self.assertEqual(hvsr.std_fn_amplitude(distribution="lognormal"),
                         np.std(np.log(peak_amp), ddof=1))
        self.assertEqual(hvsr.std_fn_amplitude(distribution="normal"),
                         np.std(peak_amp, ddof=1))

    def test_hvsrtraditional_mean_and_std_curve(self):
        frequency = np.array([1, 2, 3])
        amplitude = np.array([[1, 4, 1],
                              [2, 4, 2],
                              [2, 5, 1]])
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        mean_curve = hvsr.mean_curve(distribution="lognormal")
        std_curve = hvsr.std_curve(distribution="lognormal")
        for c_idx in range(amplitude.shape[1]):
            self.assertAlmostEqual(np.exp(np.mean(np.log(amplitude[:, c_idx]))),
                                   mean_curve[c_idx])
            self.assertAlmostEqual(np.std(np.log(amplitude[:, c_idx]), ddof=1),
                                   std_curve[c_idx])

        mean_curve = hvsr.mean_curve(distribution="normal")
        std_curve = hvsr.std_curve(distribution="normal")
        for c_idx in range(amplitude.shape[1]):
            self.assertAlmostEqual(np.mean(amplitude[:, c_idx]),
                                   mean_curve[c_idx])
            self.assertAlmostEqual(np.std(amplitude[:, c_idx], ddof=1),
                                   std_curve[c_idx])

    def test_hvsrtraditional_cov_fn(self):
        frequency = np.arange(0, 10, 1)
        amplitude = np.zeros((10, 10))
        col_idx = np.array([1, 2, 3, 5, 3, 1, 1, 1, 1, 1])
        peak_amp = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amplitude[np.arange(10), col_idx] = peak_amp
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        for distribution in ["lognormal", "normal"]:
            cov = hvsr.cov_fn(distribution=distribution)
            self.assertAlmostEqual(np.sqrt(cov[0, 0]),
                                   hvsr.std_fn_frequency(distribution=distribution))
            self.assertAlmostEqual(np.sqrt(cov[1, 1]),
                                   hvsr.std_fn_amplitude(distribution=distribution))

    def test_hvsrtraditional_mean_and_std_curve_single_window(self):
        frequency = [1, 2, 3]
        amplitude = [1, 2, 1]
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        self.assertRaises(ValueError, hvsr.std_curve)
        self.assertListEqual(amplitude, hvsr.mean_curve().tolist())

    def test_hvsrtraditional_mean_curve_peak(self):
        hvsr = hvsrpy.HvsrTraditional(self.frequency, self.amplitude[0])
        (f_peak, a_peak) = hvsr.mean_curve_peak(distribution="lognormal")
        self.assertEqual(f_peak, 2)
        self.assertEqual(a_peak, 3)

    def test_hvsrtraditional_mean_curve_peak_flat(self):
        hvsr = hvsrpy.HvsrTraditional(self.frequency, [1.1]*len(self.frequency))
        self.assertRaises(ValueError, hvsr.mean_curve_peak, distribution="lognormal")

    def test_nth_std_curve(self):
        frequency = np.array([1, 2, 3, 4, 5])
        amplitude = np.array([[1, 6, 1, 1, 1],
                              [6, 1, 1, 1, 1],
                              [1, 1, 6, 1, 1],
                              [1, 1, 1, 1, 6],
                              [1, 1, 1, 6, 1]])
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        # Curve - normal
        distribution = "normal"
        mean_curve = hvsr.mean_curve(distribution=distribution)
        std_curve = hvsr.std_curve(distribution=distribution)
        expected = mean_curve + 2*std_curve
        returned = hvsr.nth_std_curve(2, distribution=distribution)
        self.assertArrayEqual(expected, returned)

        # Curve - lognormal
        distribution = "lognormal"
        mean_curve = hvsr.mean_curve(distribution=distribution)
        std_curve = hvsr.std_curve(distribution=distribution)
        expected = np.exp(np.log(mean_curve) + 2*std_curve)
        returned = hvsr.nth_std_curve(2, distribution=distribution)
        self.assertArrayEqual(expected, returned)

    def test_nth_std_fn_frequency_and_amplitude(self):
        frequency = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        amplitude = np.array([[1, 1, 2, 3, 5, 1, 1, 1, 1],
                              [1, 2, 3, 7, 3, 1, 1, 1, 1],
                              [1, 1, 12, 1, 1, 1, 1, 1, 1]])
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)

        # nth_std_fn_frequency
        distribution = "normal"
        n = 1.5
        fn_frequencies = [3, 4, 5]
        expected = np.mean(fn_frequencies) + n*np.std(fn_frequencies, ddof=1)
        returned = hvsr.nth_std_fn_frequency(n=n, distribution=distribution)
        self.assertEqual(expected, returned)

        # nth_std_fn_amplitude
        distribution = "lognormal"
        n = -2
        fn_amplitudes = [5, 7, 12]
        expected = np.exp(np.mean(np.log(fn_amplitudes)) + n*np.std(np.log(fn_amplitudes), ddof=1))
        returned = hvsr.nth_std_fn_amplitude(n=n, distribution=distribution)
        self.assertEqual(expected, returned)

    def test_hvsrtraditional_is_similar_and_equal(self):
        a = self.hvsr
        b = hvsrpy.HvsrTraditional(self.frequency, self.amplitude)
        c = hvsrpy.HvsrTraditional(self.frequency, self.amplitude[1:])
        new_amplitude = [amplitude[:-1] for amplitude in self.amplitude]
        d = hvsrpy.HvsrTraditional(self.frequency[:-1], new_amplitude)
        new_frequency = np.array(self.frequency, dtype=float)
        new_frequency[1] += 0.1
        e = hvsrpy.HvsrTraditional(new_frequency, self.amplitude)
        f = np.array([1, 2, 3])
        new_amplitude = np.array(self.amplitude, dtype=float)
        new_amplitude[0, 1] += 0.1
        g = hvsrpy.HvsrTraditional(self.frequency, new_amplitude)
        h = hvsrpy.HvsrTraditional(self.frequency, self.amplitude)
        h.update_peaks_bounded(search_range_in_hz=(2.5, None))

        self.assertTrue(a == b)
        self.assertTrue(a.is_similar(b))
        self.assertFalse(a == c)
        self.assertTrue(a.is_similar(c))
        self.assertFalse(a == d)
        self.assertFalse(a.is_similar(d))
        self.assertFalse(a == e)
        self.assertFalse(a.is_similar(e))
        self.assertFalse(a == f)
        self.assertFalse(a.is_similar(f))
        self.assertFalse(a == g)
        self.assertTrue(a.is_similar(g))
        self.assertFalse(a == h)
        self.assertTrue(a.is_similar(h))

    def test_hvsrtraditional_str_repr(self):
        self.assertTrue(isinstance(self.hvsr.__str__(), str))
        self.assertTrue(isinstance(self.hvsr.__repr__(), str))


if __name__ == "__main__":
    unittest.main()
