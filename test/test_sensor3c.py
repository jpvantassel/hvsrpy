# This file is part of hvsrpysrpy, a Python module for
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

"""Tests for Sensor3c."""

from testtools import unittest, TestCase, get_full_path
import numpy as np
import sigpropy
import hvsrpy
import logging
logging.basicConfig(level=logging.WARNING)


class Test_Sensor3c(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_init(self):
        # Successful init
        ns = sigpropy.TimeSeries([1, 1, 1], dt=1)
        ew = sigpropy.TimeSeries([1, 1, 1], dt=1)
        vt = sigpropy.TimeSeries([1, 1, 1], dt=1)
        sensor = hvsrpy.Sensor3c(ns, ew, vt)

        # Check timeseries
        for attr, expected in zip(["ns", "ew", "vt"], [ns, ew, vt]):
            returned = getattr(sensor, attr)
            self.assertEqual(expected, returned)

        # Bad ns, should be TimeSeries
        _ns = [1, 1, 1]
        self.assertRaises(TypeError, hvsrpy.Sensor3c, _ns, ew, vt)

        # Bad ew, should be TimeSeries
        _ew = [1, 1, 1]
        self.assertRaises(TypeError, hvsrpy.Sensor3c, ns, _ew, vt)

        # Bad dt, should be 1
        ns = sigpropy.TimeSeries([1, 1, 1], dt=2)
        self.assertRaises(ValueError, hvsrpy.Sensor3c, ns, ew, vt)

        # Bad length, will trim
        amp = np.array([1, 1])
        ns = sigpropy.TimeSeries(amp, dt=1)
        hvsrpy.Sensor3c(ns, ew, vt)
        self.assertArrayEqual(amp, ew.amp)
        self.assertArrayEqual(amp, vt.amp)

    def test_normalization_factor(self):
        ns = sigpropy.TimeSeries([-1, 1, 1], dt=1)
        ew = sigpropy.TimeSeries([1, 2, 1], dt=1)
        vt = sigpropy.TimeSeries([1, 1, -5], dt=1)
        sensor = hvsrpy.Sensor3c(ns, ew, vt)

        expected = 5
        self.assertEqual(expected, sensor.normalization_factor)

        # Find second maximum
        sensor.vt.amp[2] = 0
        expected = 2
        self.assertEqual(expected, sensor.normalization_factor)

    def test_to_and_from_dict(self):
        # Simple Case
        ns = sigpropy.TimeSeries([1, 2, 3], dt=1)
        ew = sigpropy.TimeSeries([1, 4, 5], dt=1)
        vt = sigpropy.TimeSeries([1, -1, 1], dt=1)
        expected = hvsrpy.Sensor3c(ns, ew, vt)

        dict_repr = expected.to_dict()
        returned = hvsrpy.Sensor3c.from_dict(dict_repr)

        for comp in ["ns", "ew", "vt"]:
            exp = getattr(expected, comp).amp
            ret = getattr(returned, comp).amp
            self.assertArrayEqual(exp, ret)

    def test_to_and_from_json(self):
        # Simple Case
        ns = sigpropy.TimeSeries([1, 2, 3], dt=1)
        ew = sigpropy.TimeSeries([1, 4, 5], dt=1)
        vt = sigpropy.TimeSeries([1, -1, 1], dt=1)
        expected = hvsrpy.Sensor3c(ns, ew, vt)

        json_repr = expected.to_json()
        returned = hvsrpy.Sensor3c.from_json(json_repr)

        for comp in ["ns", "ew", "vt"]:
            exp = getattr(expected, comp).amp
            ret = getattr(returned, comp).amp
            self.assertArrayEqual(exp, ret)

    def test_from_miniseed(self):
        # 0101010 custom file
        fname = self.full_path+"data/custom/0101010.miniseed"
        sensor = hvsrpy.Sensor3c.from_mseed(fname)

        expected = np.array([0, 1, 0, 1, 0, 1, 0])
        for component in sensor:
            returned = component.amp
            self.assertArrayEqual(expected, returned)

        # Extra trace
        fname = self.full_path+"data/custom/extra_trace.miniseed"
        self.assertRaises(ValueError, hvsrpy.Sensor3c.from_mseed, fname)

        # Mislabeled trace
        fname = self.full_path+"data/custom/mislabeled_trace.miniseed"
        self.assertRaises(ValueError, hvsrpy.Sensor3c.from_mseed, fname)

    def test_split(self):
        # Simple Case
        component = sigpropy.TimeSeries([0, 1, 2, 3, 4, 5, 6], dt=1)
        sensor = hvsrpy.Sensor3c(component, component, component)
        wlen = 2
        sensor.split(windowlength=wlen)

        expected = sigpropy.WindowedTimeSeries.from_timeseries(component,
                                                               windowlength=wlen)
        for returned in sensor:
            self.assertEqual(expected, returned)

    def test_detrend(self):
        # Simple case
        signal = np.array([0, -0.2, -0.5, -0.2, 0, 0.2, 0.5, 0.2]*5)
        noise = np.linspace(0, 5, 40)
        component = sigpropy.TimeSeries(signal + noise, dt=1)
        sensor = hvsrpy.Sensor3c(component, component, component)
        sensor.detrend()

        expected = signal
        for returned in sensor:
            self.assertArrayAlmostEqual(expected, returned.amp, delta=0.1)

    def test_cosine_taper(self):

        def new_sensor():
            ns = sigpropy.TimeSeries(np.ones(10), dt=1)
            ew = sigpropy.TimeSeries(np.ones(10), dt=1)
            vt = sigpropy.TimeSeries(np.ones(10), dt=1)
            return hvsrpy.Sensor3c(ns, ew, vt) 

        # 0% Window - (i.e., no taper)
        sensor = new_sensor()
        sensor.cosine_taper(0)
        expected = np.ones(10)
        for returned in sensor:
            self.assertArrayEqual(expected, returned.amp)

        # 50% window
        sensor = new_sensor()
        sensor.cosine_taper(0.5)
        expected = np.array([0.000000000000000e+00, 4.131759111665348e-01,
                             9.698463103929542e-01, 1.000000000000000e+00,
                             1.000000000000000e+00, 1.000000000000000e+00,
                             1.000000000000000e+00, 9.698463103929542e-01,
                             4.131759111665348e-01, 0.000000000000000e+00])
        for returned in sensor:
            self.assertArrayAlmostEqual(expected, returned.amp, places=6)

        # 100% Window
        sensor = new_sensor()
        sensor.cosine_taper(1)
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
                             4.131759111665348e-01, 7.499999999999999e-01,
                             9.698463103929542e-01, 9.698463103929542e-01,
                             7.500000000000002e-01, 4.131759111665350e-01,
                             1.169777784405111e-01, 0.000000000000000e+00])
        for returned in sensor:
            self.assertArrayAlmostEqual(expected, returned.amp, places=6)



if __name__ == "__main__":
    unittest.main()
