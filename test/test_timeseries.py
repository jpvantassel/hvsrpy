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
from testtools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestTimeSeries(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ex_dt = 0.001
        cls.ex_time = np.arange(0, 10, cls.ex_dt)
        cls.ex_tseries_sine = hvsrpy.TimeSeries(np.sin(2*np.pi*10*cls.ex_time),
                                                cls.ex_dt)

    def test_trimming_timeseries(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        tseries.trim(0, 5)
        time = tseries.time()
        self.assertEqual(min(time), 0)
        self.assertEqual(max(time), 5)

    def test_trimming_fails_with_bad_start_time(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, -1, 5)

    def test_trimming_fails_with_bad_stop_time(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, 0, 15)

    def test_trimming_fails_with_bad_start_and_stop_times(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(IndexError, tseries.trim, 4, 2)

    def test_split_to_one_second_windows(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        windows = tseries.split(window_length_in_seconds=1.0)
        self.assertTrue(len(windows), 10)
        self.assertTrue(isinstance(windows[0], hvsrpy.TimeSeries))

    def test_split_where_window_length_is_too_large(self):
        tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_sine)
        self.assertRaises(ValueError, tseries.split, 11.0)


if __name__ == "__main__":
    unittest.main()
