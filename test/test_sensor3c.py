# This file is part of hvsrpy, a Python module for
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

import unittest
import hvsrpy as hv
import logging
logging.basicConfig(level=logging.WARNING)


class Test_Sensor3c(unittest.TestCase):
    def test_init(self):
        # Successful init
        ns = hv.TimeSeries([1, 1, 1], dt=1)
        ew = hv.TimeSeries([1, 1, 1], dt=1)
        vt = hv.TimeSeries([1, 1, 1], dt=1)
        hv.Sensor3c(ns, ew, vt)

        # Bad ns, should be TimeSeries
        ns = [1, 1, 1]
        self.assertRaises(TypeError, hv.Sensor3c, ns, ew, vt)

        # Bad dt, should be 1
        ns = hv.TimeSeries([1, 1, 1], dt=2)
        self.assertRaises(ValueError, hv.Sensor3c, ns, ew, vt)

        # Bad length, will trim
        ns = hv.TimeSeries([1, 1], dt=1)
        hv.Sensor3c(ns, ew, vt)
        self.assertEqual(ew.amp.tolist(), [1, 1])
        self.assertEqual(vt.amp.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
