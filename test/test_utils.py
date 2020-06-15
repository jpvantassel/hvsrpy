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

"""Tests for utils module."""

from testtools import TestCase, unittest, get_full_path
import pandas as pd

from hvsrpy import utils

class Test_Utils(TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_sesame(self):
        df = pd.read_csv(self.full_path + "data/utils/ex0.csv")
        mean_curve = df["mean"].to_numpy()
        std_curve = df["p1"].to_numpy()

        utils.sesame_clarity(mean_curve, std_curve)

if __name__ == "__main__":
    unittest.main()

