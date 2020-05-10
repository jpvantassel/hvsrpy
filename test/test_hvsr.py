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

import logging

import numpy as np

import hvsrpy
from testtools import unittest, TestCase, get_full_path

logging.basicConfig(level=logging.ERROR)


class Test_Hvsr(TestCase):

    @classmethod
    def setUpClass(cls):
        frq = [1, 2, 3, 4]
        amp = [[1, 2, 1, 1], [1, 2, 4, 1], [1, 1, 5, 1]]
        cls.hv = hvsrpy.Hvsr(amp, frq)
        cls.full_path = get_full_path(__file__)

    def test_init(self):
        # amp as 1d array
        frq = np.linspace(1, 10, 20)
        amp = np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10
        myhvsr = hvsrpy.Hvsr(amp, frq)
        self.assertArrayEqual(frq, myhvsr.frq)
        self.assertArrayEqual(amp, myhvsr.amp)

        # amp as 2d array
        frq = np.linspace(1, 10, 20)
        amp = (np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10)*np.ones((20, 20))
        myhvsr = hvsrpy.Hvsr(amp, frq)
        self.assertArrayEqual(frq, myhvsr.frq)
        self.assertArrayEqual(amp, myhvsr.amp)

        # amp as string
        frq = np.ndarray([1, 2, 3])
        amp = "abc"
        self.assertRaises(TypeError, hvsrpy.Hvsr, amp, frq)

        # negative amplitude
        frq = np.array([1, 2, 3])
        amp = np.array([1, -1, 3])
        self.assertRaises(ValueError, hvsrpy.Hvsr, amp, frq)

        # nan value
        frq = np.array([1, 2, 3])
        amp = np.array([1, 2, np.nan])
        self.assertRaises(ValueError, hvsrpy.Hvsr, amp, frq)

    def test_find_peaks(self):
        # amp as 1d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 0, 1, 0, 0])
        myhvsr = hvsrpy.Hvsr(amp, frq)
        self.assertListEqual(
            [2], hvsrpy.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        myhvsr = hvsrpy.Hvsr(amp, frq)
        self.assertListEqual(
            [[2], [1], [2]], hvsrpy.Hvsr.find_peaks(myhvsr.amp)[0])

        # amp as 1d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 1, 0, 1, 0])
        myhvsr = hvsrpy.Hvsr(amp, frq)
        self.assertListEqual(
            [1, 3], hvsrpy.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5, 6, 7])
        amp = np.array([[0, 1, 0, 1, 0, 5, 0],
                        [0, 2, 6, 5, 0, 0, 0],
                        [0, 0, 7, 6, 8, 0, 0]])
        myhvsr = hvsrpy.Hvsr(amp, frq)
        for known, test in zip([[1, 3, 5], [2], [2, 4]],
                               hvsrpy.Hvsr.find_peaks(myhvsr.amp)[0]):
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
        peak_ids = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amp[np.arange(10), peak_ids] = 1
        myhv = hvsrpy.Hvsr(amp, frq)

        self.assertArrayEqual(myhv.peak_frq, frq[peak_ids])

    def test_mean_std_f0_frq(self):
        frq = np.arange(0, 10, 1)
        amp = np.zeros((10, 10))
        col = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amp[np.arange(10), col] = 1
        myhv = hvsrpy.Hvsr(amp, frq)
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
        myhv = hvsrpy.Hvsr(amp, frq)
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
        myhv = hvsrpy.Hvsr(amp, frq, find_peaks=False)

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
        _hv = hvsrpy.Hvsr(amp, frq, find_peaks=False)
        self.assertRaises(ValueError, _hv.std_curve)
        self.assertListEqual(amp, _hv.mean_curve().tolist())

    def test_mc_peak_frq(self):
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hvsrpy.Hvsr(amp, frq)
        self.assertEqual(1., myhv.mc_peak_frq())

    def test_nstd_curve(self):
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([[1, 6, 1, 1, 1],
                        [6, 1, 1, 1, 1],
                        [1, 1, 6, 1, 1],
                        [1, 1, 1, 1, 6],
                        [1, 1, 1, 6, 1]])
        hv = hvsrpy.Hvsr(amp, frq)

        # Curve - Normal
        distribution = "normal"
        mean_curve = hv.mean_curve(distribution=distribution)
        std_curve = hv.std_curve(distribution=distribution)
        expected = mean_curve + std_curve
        returned = hv.nstd_curve(1, distribution=distribution)
        self.assertArrayEqual(expected, returned)

        # Curve - Log-Normal
        distribution = "log-normal"
        mean_curve = hv.mean_curve(distribution=distribution)
        std_curve = hv.std_curve(distribution=distribution)
        expected = np.exp(np.log(mean_curve) + std_curve)
        returned = hv.nstd_curve(1, distribution=distribution)
        self.assertArrayEqual(expected, returned)

    def test_reject_windows(self):
        # Reject single window, end due to zero stdev
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hvsrpy.Hvsr(amp, frq)
        myhv.reject_windows(n=2)
        self.assertEqual(myhv.mean_f0_frq(), 1.0)

        # Reject single window, end due to convergence criteria
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 2, 2, 2, 2, 1, 2, 2, 1, 9])
        amp[np.arange(10), col] = 2
        myhv = hvsrpy.Hvsr(amp, frq, find_peaks=False, meta={})
        myhv.reject_windows(n=2)
        self.assertArrayEqual(myhv.peak_frq, frq[col[:-1]])

    def test_stat_factories(self):
        distribution = "exponential"
        self.assertRaises(NotImplementedError, self.hv._mean_factory,
                          distribution, np.array([1, 2, 3, 4]))
        self.assertRaises(NotImplementedError, self.hv._std_factory,
                          distribution, np.array([1, 2, 3, 4]))
        self.assertRaises(NotImplementedError, self.hv._nth_std_factory,
                          1, distribution, 0, 0)

    def test_mc_peak(self):
        frq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        amp = np.array([[1, 1, 2, 3, 2, 1, 1, 1, 1],
                        [1, 1.5, 3, 5, 3, 1.5, 1, 1, 1],
                        [1, 1, 5, 1, 1, 1, 1, 1, 1]])
        hv = hvsrpy.Hvsr(amp, frq)

        distribution = "normal"
        expected = np.mean(amp, axis=0)
        returned = hv.mean_curve(distribution=distribution)
        self.assertArrayEqual(expected, returned)

        self.assertEqual(10/3, hv.mc_peak_amp(distribution=distribution))
        self.assertEqual(3, hv.mc_peak_frq(distribution=distribution))

    def test_nstd_f0(self):
        frq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        amp = np.array([[1, 1, 2, 3, 5, 1, 1, 1, 1],
                        [1, 2, 3, 7, 3, 1, 1, 1, 1],
                        [1, 1, 12, 1, 1, 1, 1, 1, 1]])
        hv = hvsrpy.Hvsr(amp, frq)

        distribution = "normal"
        n = 1.5

        # nstd_f0_frq
        f0s = [3, 4, 5]
        expected = np.mean(f0s) + n*np.std(f0s, ddof=1)
        returned = hv.nstd_f0_frq(n=n, distribution=distribution)
        self.assertEqual(expected, returned)

        # nstd_f0_amp
        amps = [5, 7, 12]
        expected = np.mean(amps) + n*np.std(amps, ddof=1)
        returned = hv.nstd_f0_amp(n=n, distribution=distribution)
        self.assertEqual(expected, returned)

    def test_io(self):
        fname = self.full_path + "data/a2/UT.STN11.A2_C150.miniseed"
        windowlength = 60
        bp_filter = {"flag": False, "flow": 0.1, "maxf": 30, "order": 5}
        width = 0.1
        bandwidth = 40
        resampling = {"minf": 0.2, "maxf": 20, "nf": 128, "res_type": "log"}
        method = "geometric-mean"
        sensor = hvsrpy.Sensor3c.from_mseed(fname)
        sensor.meta["File Name"] = "UT.STN11.A2_C150.miniseed"
        hv = sensor.hv(windowlength, bp_filter, width,
                       bandwidth, resampling, method)
        distribution_f0 = "log-normal"
        distribution_mc = "log-normal"

        # Pre-rejection
        df = hv._stats(distribution_f0)
        returned = np.round(df.to_numpy(), 2)
        expected = np.array([[0.64, 0.28], [1.57, 0.28]])
        self.assertArrayEqual(expected, returned)

        n = 2
        n_iteration = 50
        hv.reject_windows(n, max_iterations=n_iteration,
                          distribution_f0=distribution_f0,
                          distribution_mc=distribution_mc)

        # Post-rejection
        df = hv._stats(distribution_f0)
        returned = np.round(df.to_numpy(), 2)
        expected = np.array([[0.72, 0.10], [1.39, 0.1]])
        self.assertArrayEqual(expected, returned)

        # data_format == "hvsrpy"
        returned = hv._hvsrpy_style_lines(distribution_f0, distribution_mc)
        with open(self.full_path+"data/output/example_output_hvsrpy.hv") as f:
            expected = f.readlines()
        self.assertListEqual(expected, returned)

        # data_format == "geopsy"
        returned = hv._geopsy_style_lines(distribution_f0, distribution_mc)
        with open(self.full_path+"data/output/example_output_geopsy.hv") as f:
            expected = f.readlines()
        self.assertListEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
