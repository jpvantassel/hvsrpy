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

"""Tests associated with hvsrpy's ability to import data."""

import logging
import warnings

import hvsrpy
from testing_tools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestDataWrangler(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)
        cls.input_path = cls.full_path / "data/input"

    def test_read_single_on_mseed_combined(self):
        fname = self.input_path / "mseed_combined/ut.stn11.a2_c50.mseed"
        data = hvsrpy.data_wrangler.read_single(fname)
        # data.save(self.input_path / "srecord3c/ut.stn11.a2_c50.json")
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_mseed_individual(self):
        fnames = []
        directory = self.input_path / "mseed_individual"
        for component in ["bhe", "bhz", "bhn"]:
            fname = directory / f"ut.stn11.a2_c50_{component}.mseed"
            fnames.append(fname)
        data = hvsrpy.data_wrangler.read_single(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_saf(self):
        fname = self.input_path / "saf/mt_20211122_133110.saf"
        data = hvsrpy.data_wrangler.read_single(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_minishark(self):
        fname = self.input_path / "minishark/0003_181115_0441.minishark"
        data = hvsrpy.data_wrangler.read_single(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_sac_big_endian(self):
        fnames = []
        directory = self.input_path / "sac_big_endian"
        for component in ["e", "n", "z"]:
            fname = directory / f"ut.stn11.a2_c50_{component}.sac"
            fnames.append(fname)
        data = hvsrpy.data_wrangler.read_single(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_sac_little_endian(self):
        fnames = []
        directory = self.input_path / "sac_little_endian"
        for component in ["e", "n", "z"]:
            fname = directory / f"ut.stn11.a2_c50_{component}.sac"
            fnames.append(fname)
        data = hvsrpy.data_wrangler.read_single(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_gcf(self):
        fname = self.full_path / "data/input/gcf/sample.gcf"
        data = hvsrpy.data_wrangler.read_single(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_single_on_peer(self):
        fnames = []
        for component in ["090", "360", "-up"]:
            fname = self.input_path / f"peer/rsn942_northr_alh{component}.vt2"
            fnames.append(fname)
        data = hvsrpy.data_wrangler.read_single(fnames)
        # data.save(self.input_path / "srecord3c/rsn942_northr.json")
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_many_miniseed(self):
        fnames = [
            [self.input_path /
                f"mseed_individual/ut.stn11.a2_c50_bh{x}.mseed" for x in list("enz")],
            [self.input_path / "mseed_combined/ut.stn11.a2_c50.mseed"]
        ]
        data = hvsrpy.read(fnames)
        self.assertTrue(len(data) == 2)
        self.assertTrue(isinstance(data[0], hvsrpy.SeismicRecording3C))
        self.assertTrue(isinstance(data[1], hvsrpy.SeismicRecording3C))

    def test_read_on_many_but_single(self):
        fnames = self.input_path / "mseed_combined/ut.stn11.a2_c50.mseed"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = hvsrpy.read(fnames, obspy_read_kwargs={}, degrees_from_north=15)
        self.assertTrue(len(data) == 1)
        self.assertTrue(isinstance(data[0], hvsrpy.SeismicRecording3C))
        self.assertAlmostEqual(data[0].degrees_from_north, 15)

    def test_read_on_many_with_customizations(self):
        fnames = [
            self.input_path / "mseed_combined/ut.stn11.a2_c50.mseed",
            self.input_path / "mseed_combined/ut.stn11.a2_c50.mseed",
        ]
        data = hvsrpy.read(fnames,
                           obspy_read_kwargs=[{}, {}],
                           degrees_from_north=[10, 20])
        self.assertTrue(len(data) == 2)
        self.assertAlmostEqual(data[0].degrees_from_north, 10)
        self.assertAlmostEqual(data[1].degrees_from_north, 20)


if __name__ == "__main__":
    unittest.main()
