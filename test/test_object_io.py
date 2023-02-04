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
