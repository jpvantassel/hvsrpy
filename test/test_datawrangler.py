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

import numpy as np

import hvsrpy
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)

class TestDataWrangler(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)

    def test_read_on_mseed_combined(self):
        fname = self.full_path / "data/input/mseed_combined/ut.stn11.a2_c50.miniseed"
        data = hvsrpy.read(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_mseed_individual(self):
        fnames = []
        for component in ["bhe", "bhz", "bhn"]:
             fnames.append(self.full_path / f"data/input/mseed_individual/ut.stn11.a2_c50_{component}.mseed")
        data = hvsrpy.read(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_saf(self):
        fname = self.full_path / "data/input/saf/mt_20211122_133110.saf"
        data = hvsrpy.read(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_minishark(self):
        fname = self.full_path / "data/input/minishark/0003_181115_0441.minishark"
        data = hvsrpy.read(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_sac_big_endian(self):
        fnames = []
        for component in ["e", "n", "z"]:
            fnames.append(self.full_path / f"data/input/sac_big_endian/ut.stn11.a2_c50_{component}.sac")
        data = hvsrpy.read(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_sac_little_endian(self):
        fnames = []
        for component in ["e", "n", "z"]:
            fnames.append(self.full_path / f"data/input/sac_little_endian/ut.stn11.a2_c50_{component}.sac")
        data = hvsrpy.read(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_gcf(self):
        fname = self.full_path / "data/input/gcf/sample.gcf"
        data = hvsrpy.read(fname)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))

    def test_read_on_peer(self):
        fnames = []
        for component in ["090", "360", "-up"]:
            fnames.append(self.full_path / f"data/input/peer/rsn942_northr_alh{component}.vt2")
        data = hvsrpy.read(fnames)
        self.assertTrue(isinstance(data, hvsrpy.SeismicRecording3C))
