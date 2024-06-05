# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Tests hvsrpy's example Jupyter notebooks."""

import logging
import os
import pathlib

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from testing_tools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestExampleNotebooks(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)
        os.chdir(cls.full_path / "../examples/")

    def _test_notebook_boiler_plate(self, notebook):
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            try:
                ep.preprocess(nb)
            except:
                self.assertTrue(False)
        self.assertTrue(True)

    def test_notebook_example_ehvsr_traditional(self):
        notebook = "example_ehvsr_traditional.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_hvsr_cli(self):
        data_dir = pathlib.Path("./data")
        paths = list(data_dir.glob("UT*.png"))
        paths.extend(list(data_dir.glob("UT*.csv")))
        for path in paths:
            os.remove(path)
        notebook = "example_hvsr_cli.ipynb"
        self._test_notebook_boiler_plate(notebook)
        paths = list(data_dir.glob("UT*.png"))
        paths.extend(list(data_dir.glob("UT*.csv")))
        self.assertEqual(len(paths), 6)

    def test_notebook_example_hvsr_io(self):
        notebook = "example_hvsr_io.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_hvsr_smoothing(self):
        notebook = "example_hvsr_smoothing.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_mhvsr_azimuthal(self):
        notebook = "example_mhvsr_azimuthal.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_mhvsr_diffuse_field(self):
        notebook = "example_mhvsr_diffuse_field.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_mhvsr_traditional_sesame(self):
        notebook = "example_mhvsr_traditional_sesame.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_mhvsr_traditional_window_rejection(self):
        notebook = "example_mhvsr_traditional_window_rejection.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_mhvsr_traditional(self):
        notebook = "example_mhvsr_traditional.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_microtremor_preprocessing(self):
        notebook = "example_microtremor_preprocessing.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_psd_and_self_noise(self):
        notebook = "example_psd_and_self_noise.ipynb"
        self._test_notebook_boiler_plate(notebook)

    def test_notebook_example_spatial_hvsr(self):
        notebook = "example_spatial_hvsr.ipynb"
        self._test_notebook_boiler_plate(notebook)


if __name__ == "__main__":
    unittest.main()
