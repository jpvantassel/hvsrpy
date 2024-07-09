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

"""Test window rejection algorithms."""

import logging

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestTimeDomainWindowRejectionAlgorithms(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dt_in_seconds = 0.01
        cls.time = np.arange(0, 91, cls.dt_in_seconds)
        cls.rng = np.random.default_rng(seed=1824)
        cls.ns = hvsrpy.TimeSeries(cls.rng.normal(0, 1, len(cls.time)), cls.dt_in_seconds)
        cls.ew = hvsrpy.TimeSeries(cls.rng.normal(0, 1, len(cls.time)), cls.dt_in_seconds)
        cls.vt = hvsrpy.TimeSeries(cls.rng.normal(0, 1, len(cls.time)), cls.dt_in_seconds)
        cls.srecord = hvsrpy.SeismicRecording3C(cls.ns, cls.ew, cls.vt, 0)
        cls.srecords_before = cls.srecord.split(window_length_in_seconds=30)

    def test_sta_lta_window_rejection_no_rejection(self):
        srecords_after = hvsrpy.sta_lta_window_rejection(self.srecords_before, sta_seconds=5, lta_seconds=30,
                                                         min_sta_lta_ratio=0.9, max_sta_lta_ratio=1.1)
        self.assertEqual(len(self.srecords_before), len(srecords_after))

    def test_sta_lta_window_rejection_sta_seconds_longer_than_window(self):
        self.assertRaises(IndexError, hvsrpy.sta_lta_window_rejection,
                          self.srecords_before, sta_seconds=31, lta_seconds=30,
                          min_sta_lta_ratio=0.9, max_sta_lta_ratio=1.1)

    def test_sta_lta_window_rejection_lta_seconds_longer_than_full_record(self):
        self.assertRaises(IndexError, hvsrpy.sta_lta_window_rejection,
                          self.srecords_before, sta_seconds=5, lta_seconds=92,
                          min_sta_lta_ratio=0.9, max_sta_lta_ratio=1.1)

    def test_sta_lta_window_rejection_reject_first_window(self):
        new_ns_amplitude = np.array(self.ns.amplitude)
        n_samples = int(10/self.dt_in_seconds)
        new_ns_amplitude[:n_samples] += self.rng.normal(0, 2, n_samples)
        new_ns = hvsrpy.TimeSeries(new_ns_amplitude, self.dt_in_seconds)
        srecord = hvsrpy.SeismicRecording3C(new_ns, self.ew, self.vt, 0)
        srecords_before = srecord.split(window_length_in_seconds=30)
        srecords_after = hvsrpy.sta_lta_window_rejection(srecords_before, sta_seconds=5, lta_seconds=30,
                                                         min_sta_lta_ratio=0.9, max_sta_lta_ratio=1.1)
        self.assertEqual(len(srecords_after), 2)


class TestHvsrDomainWindowRejectionAlgorithms(TestCase):

    def test_fdwra_reject_single_window_end_due_to_to_zero_std(self):
        frequency = np.arange(0, 10, 1)
        amplitude = np.ones((10, 10))
        c_idx = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amplitude[np.arange(10), c_idx] = 2
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        hvsrpy.frequency_domain_window_rejection(hvsr, n=2)
        self.assertEqual(hvsr.mean_fn_frequency(), 1.0)
        self.assertEqual(hvsr.std_fn_frequency(), 0.0)

    def test_fdwra_reject_single_window_end_due_to_convergence_hvsrtraditional(self):
        frequency = np.arange(0, 10, 1, dtype=np.double)
        amplitude = np.ones((10, 10))
        c_idx = np.array([1, 2, 2, 2, 2, 1, 2, 2, 1, 9])
        amplitude[np.arange(10), c_idx] = 2
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        hvsrpy.frequency_domain_window_rejection(hvsr, n=2)
        self.assertArrayEqual(hvsr.peak_frequencies, frequency[c_idx[:-1]])

    def test_fdwra_reject_single_window_end_due_to_convergence_hvsrazimuthal(self):
        frequency = np.arange(0, 10, 1, dtype=np.double)
        amplitude = np.ones((10, 10))
        c_idx = np.array([1, 2, 2, 2, 2, 1, 2, 2, 1, 9])
        amplitude[np.arange(10), c_idx] = 2
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        ahvsr = hvsrpy.HvsrAzimuthal([hvsr], [0])
        hvsrpy.frequency_domain_window_rejection(ahvsr, n=2)
        self.assertArrayEqual(ahvsr.peak_frequencies[0], frequency[c_idx[:-1]])


if __name__ == "__main__":
    unittest.main()
