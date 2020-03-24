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

"""Tests for Hvsr object."""

import numpy as np
import hvsrpy as hv
from testtools import unittest, TestCase
import logging
logging.basicConfig(level=logging.WARNING)


class Test_Hvsr(TestCase):

    @classmethod
    def setUpClass(cls):
        frq = [1, 2, 3, 4]
        amp = [[1, 2, 1, 1], [1, 2, 4, 1], [1, 1, 5, 1]]
        cls.hv = hv.Hvsr(amp, frq)

    def test_init(self):
        # amp as 1d array
        frq = np.linspace(1, 10, 20)
        amp = np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10
        myhvsr = hv.Hvsr(amp, frq)
        self.assertArrayEqual(frq, myhvsr.frq)
        self.assertArrayEqual(amp, myhvsr.amp)

        # amp as 2d array
        frq = np.linspace(1, 10, 20)
        amp = (np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10)*np.ones((20, 20))
        myhvsr = hv.Hvsr(amp, frq)
        self.assertArrayEqual(frq, myhvsr.frq)
        self.assertArrayEqual(amp, myhvsr.amp)

        # amp as string
        frq = np.ndarray([1, 2, 3])
        amp = "abc"
        self.assertRaises(TypeError, hv.Hvsr, amp, frq)

        # negative amplitude
        frq = np.array([1, 2, 3])
        amp = np.array([1, -1, 3])
        self.assertRaises(ValueError, hv.Hvsr, amp, frq)

    def test_find_peaks(self):
        # amp as 1d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 0, 1, 0, 0])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual([2], hv.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual(
            [[2], [1], [2]], hv.Hvsr.find_peaks(myhvsr.amp)[0])

        # amp as 1d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 1, 0, 1, 0])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual(
            [1, 3], hv.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5, 6, 7])
        amp = np.array([[0, 1, 0, 1, 0, 5, 0],
                        [0, 2, 6, 5, 0, 0, 0],
                        [0, 0, 7, 6, 8, 0, 0]])
        myhvsr = hv.Hvsr(amp, frq)
        for known, test in zip([[1, 3, 5], [2], [2, 4]],
                               hv.Hvsr.find_peaks(myhvsr.amp)[0]):
            self.assertListEqual(known, test.tolist())

    def test_properties(self):
        # peak_frq
        expected = [2, 3, 3]
        self.hv._initialized_peaks = False
        returned = self.hv.peak_frq
        self.assertListEqual(expected, returned.tolist())

        # peak_amp
        expected = [2, 4, 5]
        self.hv._initialized_peaks = False
        returned = self.hv.peak_amp
        self.assertListEqual(expected, returned.tolist())

        # rejected_window_indices -> No rejection
        expected = []
        self.hv._initialized_peaks = False
        returned = self.hv.rejected_window_indices
        self.assertListEqual(expected, returned.tolist())

    def test_update_peaks(self):
        frq = np.arange(0, 1, 0.1)
        amp = np.zeros((10, 10))
        col = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 9])
        amp[np.arange(10), col] = 1
        myhv = hv.Hvsr(amp, frq)
        self.assertArrayEqual(myhv.peak_frq, frq[col[:-1]])

    def test_mean_std_f0_frq(self):
        frq = np.arange(0, 10, 1)
        amp = np.zeros((10, 10))
        col = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amp[np.arange(10), col] = 1
        myhv = hv.Hvsr(amp, frq)
        self.assertEqual(myhv.mean_f0_frq(distribution='log-normal'),
                         np.exp(np.mean(np.log(col))))
        self.assertEqual(myhv.mean_f0_frq(distribution='normal'), np.mean(col))
        self.assertEqual(myhv.std_f0_frq(distribution='log-normal'),
                         np.std(np.log(col), ddof=1))
        self.assertEqual(myhv.std_f0_frq(distribution='normal'),
                         np.std(col, ddof=1))

    def test_mean_std_f0_amp(self):
        frq = np.arange(0, 10, 1)
        amp = np.zeros((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        peak_amp = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amp[np.arange(10), col] = peak_amp
        myhv = hv.Hvsr(amp, frq)
        self.assertEqual(myhv.mean_f0_amp(distribution='log-normal'),
                         np.exp(np.mean(np.log(peak_amp))))
        self.assertEqual(myhv.mean_f0_amp(distribution='normal'),
                         np.mean(peak_amp))
        self.assertEqual(myhv.std_f0_amp(distribution='log-normal'),
                         np.std(np.log(peak_amp), ddof=1))
        self.assertEqual(myhv.std_f0_amp(distribution='normal'),
                         np.std(peak_amp, ddof=1))

    def test_mean_std_curve(self):
        frq = np.array([0, 1])
        amp = np.array([[1, 1],
                        [3, 4],
                        [5, 7]])
        myhv = hv.Hvsr(amp, frq, find_peaks=False)

        # Log-normal
        mean_curve = myhv.mean_curve(distribution='log-normal')
        std_curve = myhv.std_curve(distribution='log-normal')
        for col in range(amp.shape[1]):
            self.assertEqual(np.exp(np.mean(np.log(amp[:, col]))),
                             mean_curve[col])
            self.assertEqual(np.std(np.log(amp[:, col]), ddof=1),
                             std_curve[col])

        # Normal
        mean_curve = myhv.mean_curve(distribution='normal')
        std_curve = myhv.std_curve(distribution='normal')
        for col in range(amp.shape[1]):
            self.assertEqual(np.mean(amp[:, col]), mean_curve[col])
            self.assertEqual(np.std(amp[:, col], ddof=1), std_curve[col])

        # Single-Window
        frq = [1, 2]
        amp = [1, 2]
        _hv = hv.Hvsr(amp, frq, find_peaks=False)
        self.assertRaises(ValueError, _hv.std_curve)
        self.assertListEqual(amp, _hv.mean_curve().tolist())

    def test_mc_peak_frq(self):
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        self.assertEqual(1., myhv.mc_peak_frq())

    def test_nth_std(self):
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([[1, 6, 1, 1, 1],
                        [6, 1, 1, 1, 1],
                        [1, 1, 6, 1, 1],
                        [1, 1, 1, 1, 6],
                        [1, 1, 1, 6, 1]])
        _hv = hv.Hvsr(amp, frq)

        # Normal
        distribution = "normal"
        mean_curve = _hv.mean_curve(distribution=distribution)
        std_curve = _hv.std_curve(distribution=distribution)
        expected = mean_curve + std_curve
        returned = _hv.nstd_curve(1, distribution=distribution)
        self.assertArrayEqual(expected, returned)

        # Log-Normal
        distribution = "log-normal"
        mean_curve = _hv.mean_curve(distribution=distribution)
        std_curve = _hv.std_curve(distribution=distribution)
        expected = np.exp(np.log(mean_curve) + std_curve)
        returned = _hv.nstd_curve(1, distribution=distribution)
        self.assertArrayEqual(expected, returned)


    def test_reject_windows(self):
        # Reject single window, end due to zero stdev
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        myhv.reject_windows(n=2)
        self.assertEqual(myhv.mean_f0_frq(), 1.0)

        # Reject single window, end due to convergence criteria
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 2, 2, 2, 2, 1, 2, 2, 1, 9])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        myhv.reject_windows(n=2)
        self.assertArrayEqual(myhv.peak_frq, frq[col[:-1]])

    def test_stat_factories(self):
        distribution = "exponential"
        self.assertRaises(NotImplementedError, self.hv._mean_factory,
                          distribution, np.array([1, 2, 3, 4]))
        self.assertRaises(NotImplementedError, self.hv._std_factory,
                          distribution, np.array([1, 2, 3, 4]))
        self.assertRaises(NotImplementedError, self.hv._nth_std_factory,
                          distribution, np.array([1, 2, 3, 4]), 0, 0)


if __name__ == "__main__":
    unittest.main()
