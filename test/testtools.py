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

"""Testing tools."""

import unittest

def get_full_path(path):
    if path.count("/") > 1:
        file_name = path.split(r"/")[-1]
        full_path = path[:-len(file_name)]
    else:
        file_name = path.split(r"\\")[-1]
        full_path = path[:-len(file_name)]
    return full_path

class TestCase(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, **kwargs):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, **kwargs)

    def assertArrayEqual(self, array1, array2):
        self.assertListEqual(array1.tolist(), array2.tolist())

    def assertArrayAlmostEqual(self, array1, array2, **kwargs):
        if array1.size != array2.size:
            self.assertEqual(array1.size, array2.size)
        array1 = array1.flatten()
        array2 = array2.flatten()
        for v1, v2 in zip(array1, array2):
            self.assertAlmostEqual(v1, v2, **kwargs)
