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

import hvsrpy

from testtools import unittest, TestCase, get_full_path

logger = logging.getLogger('hvsrpy')
logger.setLevel(level=logging.CRITICAL)


class TestObjectIO(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)

    def _test_save_and_load_boiler_plate(self, hvsr, fname, distribution_mc, distribution_fn):
        hvsrpy.write_hvsr_to_file(hvsr,
                                  fname,
                                  distribution_mc,
                                  distribution_fn)
        self.assertTrue(os.path.exists(fname))
        nhvsr = hvsrpy.read_hvsr_from_file(fname)
        os.remove(fname)
        self.assertTrue(hvsr.is_similar(nhvsr))
        self.assertEqual(hvsr, nhvsr)
        self.assertArrayAlmostEqual(hvsr.mean_curve(), nhvsr.mean_curve())
        if not isinstance(hvsr, hvsrpy.HvsrDiffuseField):
            self.assertAlmostEqual(
                hvsr.mean_fn_frequency(), nhvsr.mean_fn_frequency())
        self.assertDictEqual(hvsr.meta, nhvsr.meta)

    def test_hvsr_traditional(self):
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
        self._test_save_and_load_boiler_plate(hvsr,
                                              "temp_save_and_load_hvsr_traditional.csv",
                                              distribution_mc="lognormal",
                                              distribution_fn="lognormal"
                                              )

    def test_hvsr_traditional_with_rejected_windows(self):
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
        self._test_save_and_load_boiler_plate(hvsr,
                                              "temp_save_and_load_hvsr_traditional_with_rejected_windows.csv",
                                              distribution_mc="lognormal",
                                              distribution_fn="lognormal"
                                              )

    def test_hvsr_azimuthal(self):
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
        self._test_save_and_load_boiler_plate(hvsr,
                                              "temp_save_and_load_azimuthal.csv",
                                              distribution_mc="lognormal",
                                              distribution_fn="lognormal"
                                              )

    def test_hvsr_azimuthal_with_rejected_windows(self):
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
        self._test_save_and_load_boiler_plate(hvsr,
                                              "temp_save_and_load_azimuthal_with_rejected_windows.csv",
                                              distribution_mc="lognormal",
                                              distribution_fn="lognormal"
                                              )

    def test_hvsr_diffuse_field(self):
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
        self._test_save_and_load_boiler_plate(hvsr,
                                              "temp_save_and_load_hvsr_diffuse_field.csv",
                                              distribution_mc="lognormal",
                                              distribution_fn="lognormal"
                                              )


if __name__ == "__main__":
    unittest.main()
