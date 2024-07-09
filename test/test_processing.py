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

import numpy as np

import hvsrpy
from hvsrpy import settings as hvsr_settings
from hvsrpy.processing import COMBINE_HORIZONTAL_REGISTER
from testing_tools import unittest, TestCase, get_full_path

class TestProcessing(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__, result_as_string=False)

        # ambient noise record
        fname = cls.full_path / "data/input/srecord3c/ut.stn11.a2_c50.json"
        cls.ambient_noise_record = hvsrpy.SeismicRecording3C.load(fname)
        cls.ambient_noise_records = [cls.ambient_noise_record]*2

        # earthquake record
        fname = cls.full_path / "data/input/srecord3c/rsn942_northr.json"
        cls.earthquake_record = hvsrpy.SeismicRecording3C.load(fname)
        cls.earthquake_records = [cls.earthquake_record]*2

    def test_preprocess_w_ambient_noise(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 60
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        self.assertAlmostEqual(preprocessed_records[0].vt.time()[-1], 60.)
        self.assertTrue(len(preprocessed_records) == 60)

    def test_preprocess_w_earthquake_records(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = None
        preprocessed_records = hvsrpy.preprocess(self.earthquake_records, settings)
        self.assertTrue(len(preprocessed_records) == 2)

    def test_process_traditional_default(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 360
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrTraditionalProcessingSettings()
        results = hvsrpy.process(preprocessed_records, settings)
        self.assertTrue(isinstance(results, hvsrpy.HvsrTraditional))

    def test_process_traditional_w_custom_horizontals(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 360
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrTraditionalProcessingSettings()
        for key in COMBINE_HORIZONTAL_REGISTER:
            settings.method_to_combine_horizontals = key
            results = hvsrpy.process(preprocessed_records, settings)
            self.assertTrue(isinstance(results, hvsrpy.HvsrTraditional))

    def test_process_traditional_single_azimuth(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 360
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrTraditionalSingleAzimuthProcessingSettings()
        results = hvsrpy.process(preprocessed_records, settings)
        self.assertTrue(isinstance(results, hvsrpy.HvsrTraditional))

    def test_process_traditional_rotdpp(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 120
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrTraditionalRotDppProcessingSettings()
        results = hvsrpy.process(preprocessed_records, settings)
        self.assertTrue(isinstance(results, hvsrpy.HvsrTraditional))

    def test_process_azimuthal(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 120
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrAzimuthalProcessingSettings()
        results = hvsrpy.process(preprocessed_records, settings)
        self.assertTrue(isinstance(results, hvsrpy.HvsrAzimuthal))

    def test_process_diffuse_field(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 120
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)
        settings = hvsr_settings.HvsrDiffuseFieldProcessingSettings()
        results = hvsrpy.process(preprocessed_records, settings)
        self.assertTrue(isinstance(results, hvsrpy.HvsrDiffuseField))

    def test_process_compare_single_azimuth_and_azimithal(self):
        settings = hvsr_settings.HvsrPreProcessingSettings()
        settings.window_length_in_seconds = 360
        preprocessed_records = hvsrpy.preprocess(self.ambient_noise_records, settings)

        # single_azimuth
        settings = hvsr_settings.HvsrTraditionalSingleAzimuthProcessingSettings()
        settings.azimuth=20.
        sa_results = hvsrpy.process(preprocessed_records, settings)

        # azimuthal
        settings = hvsr_settings.HvsrAzimuthalProcessingSettings()
        settings.azimuths_in_degrees=[20.]
        az_results = hvsrpy.process(preprocessed_records, settings)

        self.assertArrayEqual(sa_results.frequency, az_results.frequency)
        self.assertArrayEqual(sa_results.mean_curve(), az_results.mean_curve_by_azimuth()[0])
        self.assertArrayEqual(sa_results.mean_curve(), az_results.hvsrs[0].mean_curve())


if __name__ == "__main__":
    unittest.main()