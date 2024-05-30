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
        self.assertArrayEqual(hvsr.mean_curve(), nhvsr.mean_curve())
        self.assertEqual(hvsr.mean_fn_frequency(), nhvsr.mean_fn_frequency())
        self.assertDictEqual(hvsr.meta, nhvsr.meta)

    def test_hvsr_traditional(self):
        srecord_fname = self.full_path/"data/a2/UT.STN11.A2_C150.miniseed"
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
        srecord_fname = self.full_path/"data/a2/UT.STN11.A2_C150.miniseed"
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
        srecord_fname = self.full_path/"data/a2/UT.STN11.A2_C150.miniseed"
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
        srecord_fname = self.full_path/"data/a2/UT.STN11.A2_C150.miniseed"
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


if __name__ == "__main__":
    unittest.main()

#     @unittest.skip("Ignore for now")
#     def test_parse_hvsrpy_output(self):

#         def compare_data_dict(expected_dict, returned_dict):
#             for key, value in expected_dict.items():
#                 if isinstance(value, (str, int, bool)):
#                     self.assertEqual(expected_dict[key], returned_dict[key])
#                 elif isinstance(value, float):
#                     self.assertAlmostEqual(expected_dict[key],
#                                            returned_dict[key], places=3)
#                 elif isinstance(value, np.ndarray):
#                     self.assertArrayAlmostEqual(expected_dict[key],
#                                                 returned_dict[key], places=3)
#                 else:
#                     raise ValueError

#         # Ex 0
#         fname = self.full_path + "../other/ex0.hv"
#         expected = {"windowlength": 60.0,
#                     "total_windows": 30,
#                     "rejection_bool": True,
#                     "n_for_rejection": 2.0,
#                     "accepted_windows": 29,
#                     "distribution_f0": "normal",
#                     "mean_f0": 0.6976,
#                     "std_f0": 0.1353,
#                     "distribution_mc": "lognormal",
#                     "f0_mc": 0.7116,
#                     "amplitude_f0_mc": 3.8472,
#                     }
#         names = ["frequency", "curve", "lower", "upper"]
#         df = pd.read_csv(fname, comment="#", names=names)
#         for name in names:
#             expected[name] = df[name].to_numpy()
#         returned = utils.parse_hvsrpy_output(fname)
#         compare_data_dict(expected, returned)

#         # Ex 1
#         fname = self.full_path + "../other/ex1.hv"
#         expected = {"windowlength": 60.0,
#                     "total_windows": 60,
#                     "rejection_bool": True,
#                     "n_for_rejection": 2.0,
#                     "accepted_windows": 48,
#                     "distribution_f0": "normal",
#                     "mean_f0": 0.7155,
#                     "std_f0": 0.0759,
#                     "distribution_mc": "lognormal",
#                     "f0_mc": 0.7378,
#                     "amplitude_f0_mc": 3.9661,
#                     }
#         names = ["frequency", "curve", "lower", "upper"]
#         df = pd.read_csv(fname, comment="#", names=names)
#         for name in names:
#             expected[name] = df[name].to_numpy()
#         returned = utils.parse_hvsrpy_output(fname)
#         compare_data_dict(expected, returned)

#         # Ex 2
#         fname = self.full_path + "../other/ex2.hv"
#         expected = {"windowlength": 60.0,
#                     "total_windows": 30,
#                     "rejection_bool": True,
#                     "n_for_rejection": 2.0,
#                     "accepted_windows": 29,
#                     "distribution_f0": "normal",
#                     "mean_f0": 0.7035,
#                     "std_f0": 0.1391,
#                     "distribution_mc": "lognormal",
#                     "f0_mc": 0.7116,
#                     "amplitude_f0_mc": 3.9097,
#                     }
#         names = ["frequency", "curve", "lower", "upper"]
#         df = pd.read_csv(fname, comment="#", names=names)
#         for name in names:
#             expected[name] = df[name].to_numpy()
#         returned = utils.parse_hvsrpy_output(fname)
#         compare_data_dict(expected, returned)

#         # Ex 3
#         fname = self.full_path + "../other/ex3.hv"
#         expected = {"windowlength": 60.0,
#                     "total_windows": 60,
#                     "rejection_bool": True,
#                     "n_for_rejection": 2.0,
#                     "accepted_windows": 33,
#                     "distribution_f0": "normal",
#                     "mean_f0": 0.7994,
#                     "std_f0": 0.0365,
#                     "distribution_mc": "lognormal",
#                     "f0_mc": 0.7933,
#                     "amplitude_f0_mc": 4.8444,
#                     }
#         names = ["frequency", "curve", "lower", "upper"]
#         df = pd.read_csv(fname, comment="#", names=names)
#         for name in names:
#             expected[name] = df[name].to_numpy()
#         returned = utils.parse_hvsrpy_output(fname)
#         compare_data_dict(expected, returned)


# # import swprocess
# # import json
# # from testtools import unittest, TestCase, get_full_path
# # import logging
# # logging.basicConfig(level=logging.CRITICAL)


# # class Test_Hvsr(TestCase):

# #     def setUp(self):
# #         self.fpath = get_full_path(__file__)

# #     def test_from_geopsy_file(self):
# #         fname = self.fpath + "data/mm/test_ZM_STN01.hv"
# #         test = swprocess.Hvsr.from_geopsy_file(fname=fname,
# #                                                            identifier="TaDa")
# #         frq = [[0.1, 0.101224, 0.102462, 50]]
# #         amp = [[4.26219, 4.24461, 4.20394, 0.723993]]
# #         idn = "TaDa"

# #         self.assertListEqual(test.frq, frq)
# #         self.assertListEqual(test.amp, amp)
# #         self.assertTrue(test.idn, idn)

# #     def test_from_geopsy_folder(self):
# #         dirname = self.fpath + "data/mm/test_dir"
# #         hv = swprocess.Hvsr.from_geopsy_folder(dirname=dirname,
# #                                                            identifier="TADA")
# #         with open(self.fpath + "data/mm/test_dir_data.json", "r") as f:
# #             known = json.load(f)

# #         for test_frq, test_amp, known in zip(hv.frq, hv.amp, known):
# #             self.assertListEqual(test_frq, known["frequency"])
# #             self.assertListEqual(test_amp, known["amplitude"])


# # if __name__ == "__main__":
# #     unittest.main()
#     @unittest.skip("Ignore for now")
#     def test_io(self):
#         fname = self.full_path + "data/a2/UT.STN11.A2_C150.miniseed"
#         windowlength = 60
#         bp_filter = {"flag": False, "flow": 0.1, "maxf": 30, "order": 5}
#         width = 0.1
#         bandwidth = 40
#         resampling = {"minf": 0.2, "maxf": 20, "nf": 128, "res_type": "log"}
#         method = "geometric-mean"
#         sensor = hvsrpy.Sensor3c.from_mseed(fname)
#         sensor.meta["File Name"] = "UT.STN11.A2_C150.miniseed"
#         hv = sensor.hv(windowlength, bp_filter, width,
#                        bandwidth, resampling, method)
#         distribution_f0 = "lognormal"
#         distribution_mc = "lognormal"

#         # Pre-rejection
#         df = hv._stats(distribution_f0)
#         returned = np.round(df.to_numpy(), 2)
#         expected = np.array([[0.64, 0.28], [1.57, 0.28]])
#         self.assertArrayEqual(expected, returned)

#         n = 2
#         n_iteration = 50
#         hv.reject_windows(n, max_iterations=n_iteration,
#                           distribution_f0=distribution_f0,
#                           distribution_mc=distribution_mc)

#         # Post-rejection
#         df = hv._stats(distribution_f0)
#         returned = np.round(df.to_numpy(), 2)
#         expected = np.array([[0.72, 0.10], [1.39, 0.1]])
#         self.assertArrayEqual(expected, returned)

#         # data_format == "hvsrpy"
#         returned = hv._hvsrpy_style_lines(distribution_f0, distribution_mc)
#         with open(self.full_path+"data/output/example_output_hvsrpy.hv") as f:
#             expected = f.readlines()
#         self.assertListEqual(expected, returned)

#         # data_format == "geopsy"
#         returned = hv._geopsy_style_lines(distribution_f0, distribution_mc)
#         with open(self.full_path+"data/output/example_output_geopsy.hv") as f:
#             expected = f.readlines()
#         self.assertListEqual(expected, returned)
