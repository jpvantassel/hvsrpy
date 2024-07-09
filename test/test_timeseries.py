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

"""Test functionality of TimeSeries."""

import logging

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestTimeSeries(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ex_dt = 0.001
        cls.ex_time = np.arange(0, 10, cls.ex_dt)
        cls.ex_tseries_sine = hvsrpy.TimeSeries(np.sin(2*np.pi*10*cls.ex_time),
                                                cls.ex_dt)

    def test_timeseries_trim(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        tseries.trim(0, 5)
        time = tseries.time()
        self.assertEqual(min(time), 0)
        self.assertEqual(max(time), 5)

    def test_timeseries_trim_fails_with_bad_start_time(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, -1, 5)

    def test_timeseries_trim_fails_with_bad_stop_time(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, 0, 15)

    def test_timeseries_trim_fails_with_bad_start_and_stop_times(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, 4, 2)

    def test_timeseries_split_to_one_second_windows(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        windows = tseries.split(window_length_in_seconds=1.0)
        self.assertTrue(len(windows), 10)
        self.assertTrue(isinstance(windows[0], hvsrpy.TimeSeries))

    def test_timeseries_split_where_window_length_is_too_large(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(ValueError, tseries.split, 11.0)

    def test_timeseries_init_with_bad_amplitude_non_numeric(self):
        self.assertRaises(TypeError, hvsrpy.TimeSeries,
                          amplitude=["a", "b", "c"], dt_in_seconds=0.001)

    def test_timeseries_init_with_bad_amplitude_non_1d(self):
        self.assertRaises(TypeError, hvsrpy.TimeSeries,
                          amplitude=[[1.0, 2.0], [1.0, 2.0]], dt_in_seconds=0.001)

    def test_timeseries_fs_and_fnyq(self):
        fs_in_hz = 100
        dt_in_seconds = 1/fs_in_hz
        tseries = hvsrpy.TimeSeries(amplitude=[1.0], dt_in_seconds=dt_in_seconds)
        self.assertEqual(tseries.fs, 100)
        self.assertEqual(tseries.fnyq, 50)

    def test_timeseries_window_with_bad_type(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(NotImplementedError, tseries.window, type="cosine")

    def test_timeseries_filter(self):
        unfilt_tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        filt_tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)

        filt_tseries.butterworth_filter(fcs_in_hz=(3, 5))
        self.assertTrue(unfilt_tseries.is_similar(filt_tseries))

        filt_tseries.butterworth_filter(fcs_in_hz=(None, 5))
        self.assertTrue(unfilt_tseries.is_similar(filt_tseries))

        filt_tseries.butterworth_filter(fcs_in_hz=(3, None))
        self.assertTrue(unfilt_tseries.is_similar(filt_tseries))

    def test_timeseries_is_similar_and_equal(self):
        # baseline
        a = hvsrpy.TimeSeries(amplitude=[1., 2.], dt_in_seconds=1.)

        # is_similar = True
        b = hvsrpy.TimeSeries(amplitude=[2., 3.], dt_in_seconds=1.)

        self.assertTrue(a == a)
        self.assertTrue(a.is_similar(b))

        # is_similar = False
        c = np.array([1., 2.])
        d = hvsrpy.TimeSeries(amplitude=[1., 2.], dt_in_seconds=2.)
        e = hvsrpy.TimeSeries(amplitude=[1., 2., 3.], dt_in_seconds=1.)

        self.assertTrue(a != c)
        self.assertFalse(a.is_similar(c))
        self.assertTrue(a != d)
        self.assertFalse(a.is_similar(d))
        self.assertTrue(a != e)
        self.assertFalse(a.is_similar(e))

    def test_timeseries_str_and_repr(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertTrue(isinstance(tseries.__str__(), str))
        self.assertTrue(isinstance(tseries.__repr__(), str))


if __name__ == "__main__":
    unittest.main()
