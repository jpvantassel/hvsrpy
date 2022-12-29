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

import numpy as np

import hvsrpy
from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestSeismicRecording3C(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dt = 0.001
        cls.time = np.arange(0, 10, cls.dt)
        cls.ex_tseries_cosine = hvsrpy.TimeSeries(np.cos(2*np.pi*10*cls.time),
                                                  cls.dt)
        cls.ex_srecord3c_cosine = hvsrpy.SeismicRecording3C(cls.ex_tseries_cosine,
                                                            cls.ex_tseries_cosine,
                                                            cls.ex_tseries_cosine)

    def test_srecord3c_split_to_one_second_srecord3c(self):
        ex = self.ex_srecord3c_cosine
        srecord3c = hvsrpy.SeismicRecording3C.from_seismic_recording_3c(ex)
        windows = srecord3c.split(1.0)
        self.assertTrue(len(windows) == 10)
        self.assertTrue(isinstance(windows[0], hvsrpy.SeismicRecording3C))

    # TODO (jpv): Add degrees from north and associated rotation ability
    # throughout hvsrpy workflow.
    def test_srecord3c_rotate_to_orientation(self):
        pass

    # TODO (jpv): Add degrees from north and associated rotation ability
    # throughout hvsrpy workflow.
    def test_srecord3c_rotate_to_north(self):
        pass


if __name__ == "__main__":
    unittest.main()
