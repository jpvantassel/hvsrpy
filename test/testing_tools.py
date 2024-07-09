# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2021 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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
import pathlib

import numpy as np


def get_full_path(path, result_as_string=True):
    full_path = pathlib.Path(path).resolve().parent
    return str(full_path) if result_as_string else full_path

class TestCase(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, **kwargs):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, **kwargs)

    def assertNestedListEqual(self, list1, list2, **kwargs):
        if len(list1) != len(list2):
            msg = f"\nExpected:\n{list1}\nReturned:\n{list2})"
            raise AssertionError(msg)

        for l1, l2 in zip(list1, list2):
            self.assertListEqual(l1, l2, **kwargs)

    def assertArrayEqual(self, array1, array2):
        try:
            self.assertTrue(np.equal(array1, array2, casting='no').all())
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2})"
            raise AssertionError(msg) from e

    def assertArrayAlmostEqual(self, array1, array2, **kwargs):
        if kwargs.get("places", False):
            kwargs["atol"] = 1/(10**kwargs["places"])
            del kwargs["places"]

        if kwargs.get("delta", False):
            kwargs["atol"] = kwargs["delta"]
            del kwargs["delta"]

        try:
            self.assertTrue(np.allclose(array1, array2, **kwargs))
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2})"
            raise AssertionError(msg) from e
