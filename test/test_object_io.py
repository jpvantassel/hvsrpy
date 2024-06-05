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

"""Tests hvsrpy's input-output functionality."""

import logging
import os

import numpy as np

import hvsrpy

from testing_tools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestObjectIO(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)

    def _test_save_and_load_hvsr_boiler_plate(self, hvsr, fname, distribution_mc, distribution_fn):
        hvsrpy.write_hvsr_object_to_file(hvsr,
                                  fname,
                                  distribution_mc,
                                  distribution_fn)
        self.assertTrue(os.path.exists(fname))
        nhvsr = hvsrpy.read_hvsr_object_from_file(fname)
        os.remove(fname)
        self.assertTrue(hvsr.is_similar(nhvsr))
        self.assertEqual(hvsr, nhvsr)
        self.assertArrayAlmostEqual(hvsr.mean_curve(), nhvsr.mean_curve())
        if not isinstance(hvsr, hvsrpy.HvsrDiffuseField):
            self.assertAlmostEqual(
                hvsr.mean_fn_frequency(), nhvsr.mean_fn_frequency())
        self.assertDictEqual(hvsr.meta, nhvsr.meta)

    def test_write_and_read_hvsr_traditional(self):
        srecord_fname = self.full_path/"data/input/mseed_combined/ut.stn11.a2_c50.mseed"
        srecord = hvsrpy.read([[srecord_fname]])
        srecord = hvsrpy.preprocess(
            srecord,
            hvsrpy.HvsrPreProcessingSettings()
        )
        hvsr = hvsrpy.process(
            srecord,
            hvsrpy.HvsrTraditionalProcessingSettings()
        )
        self._test_save_and_load_hvsr_boiler_plate(
            hvsr,
            "temp_save_and_load_hvsr_traditional.csv",
            distribution_mc="lognormal",
            distribution_fn="lognormal"
        )

    def test_write_and_read_hvsr_traditional_with_rejected_windows(self):
        srecord_fname = self.full_path/"data/input/mseed_combined/ut.stn11.a2_c50.mseed"
        srecord = hvsrpy.read([[srecord_fname]])
        srecord = hvsrpy.preprocess(
            srecord,
            hvsrpy.HvsrPreProcessingSettings()
        )
        hvsr = hvsrpy.process(
            srecord,
            hvsrpy.HvsrTraditionalProcessingSettings()
        )
        hvsr.valid_peak_boolean_mask[2:8] = False
        hvsr.valid_window_boolean_mask[2:8] = False
        self._test_save_and_load_hvsr_boiler_plate(
            hvsr,
            "temp_save_and_load_hvsr_traditional_with_rejected_windows.csv",
            distribution_mc="lognormal",
            distribution_fn="lognormal"
        )

    def test_write_and_read_hvsr_azimuthal(self):
        srecord_fname = self.full_path/"data/input/mseed_combined/ut.stn11.a2_c50.mseed"
        srecord = hvsrpy.read([[srecord_fname]])
        srecord = hvsrpy.preprocess(
            srecord,
            hvsrpy.HvsrPreProcessingSettings()
        )
        hvsr = hvsrpy.process(
            srecord,
            hvsrpy.HvsrAzimuthalProcessingSettings()
        )
        self._test_save_and_load_hvsr_boiler_plate(
            hvsr,
            "temp_save_and_load_azimuthal.csv",
            distribution_mc="lognormal",
            distribution_fn="lognormal"
        )

    def test_write_and_read_hvsr_azimuthal_with_rejected_windows(self):
        srecord_fname = self.full_path/"data/input/mseed_combined/ut.stn11.a2_c50.mseed"
        srecord = hvsrpy.read([[srecord_fname]])
        srecord = hvsrpy.preprocess(
            srecord,
            hvsrpy.HvsrPreProcessingSettings()
        )
        hvsr = hvsrpy.process(
            srecord,
            hvsrpy.HvsrAzimuthalProcessingSettings()
        )
        hvsr.hvsrs[5].valid_window_boolean_mask[5:10] = False
        hvsr.hvsrs[5].valid_peak_boolean_mask[5:10] = False
        hvsr.hvsrs[7].valid_window_boolean_mask[:10] = False
        hvsr.hvsrs[7].valid_peak_boolean_mask[:10] = False
        hvsr.hvsrs[9].valid_window_boolean_mask[10:20] = False
        hvsr.hvsrs[9].valid_peak_boolean_mask[10:20] = False
        self._test_save_and_load_hvsr_boiler_plate(
            hvsr,
            "temp_save_and_load_azimuthal_with_rejected_windows.csv",
            distribution_mc="lognormal",
            distribution_fn="lognormal"
        )

    def test_write_and_read_hvsr_diffuse_field(self):
        srecord_fname = self.full_path/"data/input/mseed_combined/ut.stn11.a2_c50.mseed"
        srecord = hvsrpy.read([[srecord_fname]])
        srecord = hvsrpy.preprocess(
            srecord,
            hvsrpy.HvsrPreProcessingSettings()
        )
        hvsr = hvsrpy.process(
            srecord,
            hvsrpy.HvsrDiffuseFieldProcessingSettings()
        )
        self._test_save_and_load_hvsr_boiler_plate(
            hvsr,
            "temp_save_and_load_hvsr_diffuse_field.csv",
            distribution_mc="lognormal",
            distribution_fn="lognormal"
        )

    def _test_write_read_settings_boiler_plate(self, settings, fname):
        hvsrpy.write_settings_object_to_file(settings, fname)
        self.assertTrue(os.path.exists(fname))
        new_settings = hvsrpy.read_settings_object_from_file(fname)
        os.remove(fname)
        self.assertEqual(settings, new_settings)

    def test_write_and_read_hvsr_preprocess_settings(self):
        settings = hvsrpy.HvsrPreProcessingSettings()
        settings.orient_to_degrees_from_north = 25.
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_hvsr_preprocess_settings.json"
        )

    def test_write_and_read_psd_preprocess_settings(self):
        settings = hvsrpy.PsdPreProcessingSettings()
        settings.orient_to_degrees_from_north = 25.
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_psd_preprocess_settings.json"
        )

    def test_write_and_read_psd_process_settings(self):
        settings = hvsrpy.PsdProcessingSettings()
        settings.window_type_and_width = ["tukey", 0.5]
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_psd_process_settings.json"
        )

    def test_write_and_read_hvsr_traditional_process_settings(self):
        settings = hvsrpy.HvsrTraditionalProcessingSettings()
        settings.smoothing["center_frequencies_in_hz"] = np.linspace(0, 10, 20)
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_traditional_process_settings.json"
        )

    def test_write_and_read_hvsr_single_azimuth_processing_settings(self):
        settings = hvsrpy.HvsrTraditionalSingleAzimuthProcessingSettings()
        settings.azimuth_in_degrees = 50
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_single_azimuth_process_settings.json"
        )

    def test_write_and_read_hvsr_rotdpp_processing_settings(self):
        settings = hvsrpy.HvsrTraditionalRotDppProcessingSettings()
        settings.ppth_percentile_for_rotdpp_computation = 80.
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_rotdpp_process_settings.json"
        )

    def test_write_and_read_hvsr_azimuthal_process_settings(self):
        settings = hvsrpy.HvsrAzimuthalProcessingSettings()
        settings.azimuths_in_degrees = np.arange(0, 180, 30)
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_azimuthal_process_settings.json"
        )

    def test_write_and_read_hvsr_diffuse_field_process_settings(self):
        settings = hvsrpy.HvsrDiffuseFieldProcessingSettings()
        settings.smoothing["center_frequencies_in_hz"] = np.geomspace(0.1, 10, 30)
        self._test_write_read_settings_boiler_plate(
            settings,
            "temp_diffuse_field_process_settings.json"
        )


if __name__ == "__main__":
    unittest.main()
