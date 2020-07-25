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

"""Tests for Sensor3c."""

import logging
import json
import warnings

import numpy as np
import pandas as pd
import sigpropy

import hvsrpy
from testtools import unittest, TestCase, get_full_path

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
        ns = sigpropy.TimeSeries([1., 2, 3], dt=1)
        ew = sigpropy.TimeSeries([1., 4, 5], dt=1)
        vt = sigpropy.TimeSeries([1., 1, 1], dt=1)
        expected = hvsrpy.Sensor3c(ns, ew, vt, meta={"windowlength": 1})

        dict_repr = expected.to_dict()
        returned = hvsrpy.Sensor3c.from_dict(dict_repr)

        for comp in ["ns", "ew", "vt", "meta"]:
            exp = getattr(expected, comp)
            ret = getattr(returned, comp)
            self.assertEqual(exp, ret)

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

    def test_from_mseed(self):
        # fname is not None
        # -----------------

        # 0101010 custom file
        fname = self.full_path+"data/custom/0101010.mseed"
        sensor = hvsrpy.Sensor3c.from_mseed(fname)
        expected = np.array([0, 1, 0, 1, 0, 1, 0])
        for component in sensor:
            returned = component.amp
            self.assertArrayEqual(expected, returned)

        # Extra trace
        fname = self.full_path+"data/custom/extra_trace.mseed"
        self.assertRaises(ValueError, hvsrpy.Sensor3c.from_mseed, fname)

        # Mislabeled trace
        fname = self.full_path+"data/custom/mislabeled_trace.mseed"
        self.assertRaises(ValueError, hvsrpy.Sensor3c.from_mseed, fname)

        # fnames_1c is not None
        # -------------------------

        # 0101010 custom files
        prefix = self.full_path + "data/custom"
        fnames_1c = {c: f"{prefix}/channel_{c}.mseed" for c in list("enz")}
        sensor = hvsrpy.Sensor3c.from_mseed(fnames_1c=fnames_1c)
        base = np.array([0, 1, 0, 1, 0, 1, 0])
        for factor, component in enumerate(sensor, start=1):
            expected = base*factor
            returned = component.amp
            self.assertArrayEqual(expected, returned)

        # 0101010 custom files with components switched -> warning
        fnames_1c["n"], fnames_1c["e"] = fnames_1c["e"], fnames_1c["n"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sensor = hvsrpy.Sensor3c.from_mseed(fnames_1c=fnames_1c)
        for component, factor in zip(sensor, [2, 1, 3]):
            expected = base*factor
            returned = component.amp
            self.assertArrayEqual(expected, returned)

        # len(stream) > 1
        fnames_1c = {c: f"{prefix}/channel_{c}.mseed" for c in list("enz")}
        fnames_1c["z"] = f"{prefix}/0101010.mseed"
        self.assertRaises(IndexError, hvsrpy.Sensor3c.from_mseed,
                          fnames_1c=fnames_1c)

        # fname and fnames_1c are None
        # --------------------------------
        self.assertRaises(ValueError, hvsrpy.Sensor3c.from_mseed)

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

    def test_transform(self):
        amp = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        dt = 1

        expected_frq = np.array([0.00000000000000000, 0.09090909090909091,
                                 0.18181818181818182, 0.2727272727272727,
                                 0.36363636363636365, 0.4545454545454546])
        expected_amp = np.array([25.0+0.0*1j,
                                 -11.843537519677056+-3.477576385886737*1j,
                                 0.22844587066117938+0.14681324646918337*1j,
                                 -0.9486905697966428+-1.0948472814948405*1j,
                                 0.1467467171062613+0.3213304885841657*1j,
                                 -0.08296449829374097+-0.5770307602665046*1j])
        expected_amp *= 2/len(amp)

        # TimeSeries
        tseries = sigpropy.TimeSeries(amp, dt)
        sensor = hvsrpy.Sensor3c(tseries, tseries, tseries)
        fft = sensor.transform()

        for val in fft.values():
            for expected, returned in [(expected_frq, val.frq), (expected_amp, val.amp)]:
                self.assertArrayAlmostEqual(expected, returned)

        # WindowedTimeSeries
        amps = np.array([amp, amp])
        tseries = sigpropy.WindowedTimeSeries(amps, dt)
        sensor = hvsrpy.Sensor3c(tseries, tseries, tseries)
        fft = sensor.transform()

        for val in fft.values():
            for expected, returned in [(expected_frq, val.frq), (expected_amp, val.amp[0])]:
                self.assertArrayAlmostEqual(expected, returned)

        # Bad TimeSeries
        sensor.ew = "bad TimeSeries"
        self.assertRaises(NotImplementedError, sensor.transform)

    def test_hv(self):
        with open(self.full_path+"data/integration/int_singlewindow_cases.json", "r") as f:
            cases = json.load(f)

        bp_filter = {"flag": False, "flow": 0.001, "fhigh": 49.9, "order": 5}

        for key, value in cases.items():
            if key in ["f", "k", "l"]:
                continue
            fname = self.full_path+value["fname_miniseed"]
            sensor = hvsrpy.Sensor3c.from_mseed(fname)
            settings = value["settings"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_hv = sensor.hv(settings["length"],
                                  bp_filter,
                                  settings["width"],
                                  settings["b"],
                                  settings["resampling"],
                                  settings["method"],
                                  azimuth=settings.get("azimuth"))
            geopsy_hv = pd.read_csv(self.full_path+value["fname_geopsy"],
                                    delimiter="\t",
                                    comment="#",
                                    names=["frq", "avg", "min", "max"])
            self.assertArrayAlmostEqual(my_hv.amp[0],
                                        geopsy_hv["avg"].to_numpy(),
                                        delta=0.375)

    def test_str_and_repr(self):
        fname = self.full_path + "data/custom/0101010.mseed"
        sensor = hvsrpy.Sensor3c.from_mseed(fname=fname)

        # str
        self.assertEqual("Sensor3c", sensor.__str__())

        # repr
        expected = f"Sensor3c(ns={sensor.ns}, ew={sensor.ew}, vt={sensor.vt}, meta={sensor.meta})"
        returned = sensor.__repr__()
        self.assertEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
