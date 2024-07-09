# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Tests for Settings module."""

import os

import numpy as np

import hvsrpy

from testing_tools import TestCase, unittest


class TestSettings(TestCase):

    def test_init_settings(self):
        settings = hvsrpy.HvsrTraditionalProcessingSettings()
        self.assertTrue(isinstance(settings, hvsrpy.settings.Settings))

    def test_settings_repr_and_str(self):
        settings = hvsrpy.HvsrTraditionalProcessingSettings()
        self.assertTrue(isinstance(settings.__str__(), str))
        self.assertTrue(isinstance(settings.__repr__(), str))

    def _test_save_and_load_boiler_plate(self, cls, customizations, fname):
        setting_to_save = cls()
        # must modify something from the default to ensure load()
        # does not just create a new default object.
        for key, value in customizations.items():
            setattr(setting_to_save, key, value)
        setting_to_save.save(fname)

        setting_to_load = cls()
        setting_to_load.load(fname)
        os.remove(fname)

        self.assertDictEqual(setting_to_save.attr_dict,
                             setting_to_load.attr_dict)

    def test_hvsrpreprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrPreProcessingSettings,
                                              dict(detrend="constant"),
                                              "temp_hvsrpreprocessingsettings.json")

    def test_psdpreprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.PsdPreProcessingSettings,
                                              dict(differentiate=True),
                                              "temp_psdpreprocessingsettings.json")

    def test_psdprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.PsdProcessingSettings,
                                              dict(window_type_and_width=["tukey", 0.2]),
                                              "temp_psdprocessingsettings.json")

    def test_hvsrtraditionalprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrTraditionalProcessingSettings,
                                              dict(method_to_combine_horizontals="squared_average"),
                                              "temp_hvsrtraditionalprocessingsettings.json")

    def test_hvsrtraditionalsingleazimuthprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrTraditionalSingleAzimuthProcessingSettings,
                                              dict(azimuth_in_degrees=10.),
                                              "temp_hvsrtraditionalsingleazimuthprocessingsettings.json")

    def test_hvsrtraditionalrotdppprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrTraditionalRotDppProcessingSettings,
                                              dict(azimuths_in_degrees=np.arange(0, 180, 20)),
                                              "temp_hvsrtraditionalrotdppprocessingsettings.json")

    def test_hvsrazimuthalprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrAzimuthalProcessingSettings,
                                              dict(azimuths_in_degrees=np.arange(0, 180, 20)),
                                              "temp_hvsrazimuthalprocessingsettings.json")

    def test_hvsrdiffusefieldprocessingsettings_save_and_load(self):
        self._test_save_and_load_boiler_plate(hvsrpy.HvsrDiffuseFieldProcessingSettings,
                                              dict(smoothing=dict(operator="konno_and_ohmachi",
                                                                  bandwidth=20,
                                                                  center_frequencies_in_hz=np.geomspace(0.1, 50, 128))),
                                              "temp_hvsrdiffusefieldprocessingsettings.json")


if __name__ == "__main__":
    unittest.main()
