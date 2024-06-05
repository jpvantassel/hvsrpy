# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2022 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Test functionionality of SeismicRecording3C."""

import logging
import os
import pathlib

import numpy as np

import hvsrpy
from testing_tools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestSeismicRecording3C(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)
        cls.dt_in_seconds = 0.001
        cls.time = np.arange(0, 10, cls.dt_in_seconds)
        cls.ex_tseries_cosine = hvsrpy.TimeSeries(np.cos(2*np.pi*10*cls.time),
                                                  cls.dt_in_seconds)
        cls.ex_srecord3c_cosine = hvsrpy.SeismicRecording3C(cls.ex_tseries_cosine,
                                                            cls.ex_tseries_cosine,
                                                            cls.ex_tseries_cosine)
        cls.ex_tseries_sine = hvsrpy.TimeSeries(np.sin(2*np.pi*10*cls.time),
                                                cls.dt_in_seconds)
        cls.ex_srecord3c_sine = hvsrpy.SeismicRecording3C(cls.ex_tseries_sine,
                                                          cls.ex_tseries_sine,
                                                          cls.ex_tseries_sine)

    def test_srecord3c_split_to_one_second_srecord3c(self):
        ex = self.ex_srecord3c_cosine
        srecord3c = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(ex)
        windows = srecord3c.split(1.0)
        self.assertTrue(len(windows) == 10)
        self.assertTrue(isinstance(windows[0], hvsrpy.SeismicRecording3C))

    def test_srecord3c_save(self):
        ex = self.ex_srecord3c_cosine
        fname = self.full_path / "data/temp/ex_record3c_cosine_save.json"
        ex.save(fname)
        self.assertTrue(fname.exists())
        os.remove(fname)

    def test_srecord3c_load(self):
        org_ex = self.ex_srecord3c_cosine
        fname = self.full_path / "data/temp/ex_reocord3c_cosine_load.json"
        org_ex.save(fname)
        new_ex = hvsrpy.SeismicRecording3C.load(fname)
        self.assertTrue(org_ex == new_ex)
        os.remove(fname)

    def test_srecord3c_with_incompatible_timeseries(self):
        incompatible_tseries = hvsrpy.TimeSeries.from_timeseries(self.ex_tseries_cosine)
        incompatible_tseries.dt_in_seconds = self.dt_in_seconds*10
        self.assertRaises(ValueError, hvsrpy.SeismicRecording3C,
                          self.ex_tseries_cosine,
                          self.ex_tseries_cosine,
                          incompatible_tseries)

    def test_srecord3c_is_similar_and_equal(self):
        a = self.ex_srecord3c_cosine
        b = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(a)
        c = self.ex_srecord3c_sine
        d = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(a)
        d.vt.dt_in_seconds = float(self.dt_in_seconds * 10)
        d.ns.dt_in_seconds = d.vt.dt_in_seconds
        d.ew.dt_in_seconds = d.vt.dt_in_seconds
        e = self.ex_tseries_sine
        f = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(a)
        f.degrees_from_north = 3.14

        self.assertTrue(a.is_similar(b))
        self.assertTrue(a.is_similar(c))
        self.assertFalse(a.is_similar(d))
        self.assertFalse(a.is_similar(e))
        self.assertTrue(a.is_similar(f))

        self.assertTrue(a == b)
        self.assertTrue(a != c)
        self.assertTrue(a != d)
        self.assertTrue(a != e)
        self.assertTrue(a != f)

    def test_srecord3c_orient_sensor_to(self):
        vt = hvsrpy.TimeSeries(amplitude=[0.], dt_in_seconds=1)
        ew = hvsrpy.TimeSeries(amplitude=[3.], dt_in_seconds=1)
        ns = hvsrpy.TimeSeries(amplitude=[4.], dt_in_seconds=1)

        # rotate 90 degrees
        srecord = hvsrpy.SeismicRecording3C(ns, ew, vt, degrees_from_north=0)
        srecord.orient_sensor_to(90.)
        self.assertAlmostEqual(srecord.ew.amplitude[0], -4.)
        self.assertAlmostEqual(srecord.ns.amplitude[0], +3.)

        # rotate 360 degrees
        srecord = hvsrpy.SeismicRecording3C(ns, ew, vt, degrees_from_north=0)
        srecord.orient_sensor_to(360.)
        self.assertAlmostEqual(srecord.ew.amplitude[0], +3.)
        self.assertAlmostEqual(srecord.ns.amplitude[0], +4.)

    def test_srecord3c_trim(self):
        srecord = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(self.ex_srecord3c_sine)
        srecord.trim(1, 5)
        self.assertEqual(srecord.vt.n_samples, 4001)

    def test_srecord3c_str_and_repr(self):
        srecord = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(self.ex_srecord3c_sine)
        self.assertTrue(isinstance(srecord.__str__(), str))
        self.assertTrue(isinstance(srecord.__repr__(), str))


if __name__ == "__main__":
    unittest.main()
