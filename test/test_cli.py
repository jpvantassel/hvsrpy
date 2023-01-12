# # This file is part of hvsrpy, a Python package for
# # horizontal-to-vertical spectral ratio processing.
# # Copyright (C) 2019-2021 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
# #
# #     This program is free software: you can redistribute it and/or modify
# #     it under the terms of the GNU General Public License as published by
# #     the Free Software Foundation, either version 3 of the License, or
# #     (at your option) any later version.
# #
# #     This program is distributed in the hope that it will be useful,
# #     but WITHOUT ANY WARRANTY; without even the implied warranty of
# #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #     GNU General Public License for more details.
# #
# #     You should have received a copy of the GNU General Public License
# #     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

# """Tests for HvsrRotated object."""

# import logging
# import os
# import subprocess

# from hvsrpy.cli import parse_config
# from testtools import unittest, TestCase, get_full_path

# logging.basicConfig(level=logging.ERROR)


# class TestCLI(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.full_path = get_full_path(__file__)

#     @unittest.skip("Ignore for now")
#     def test_parse_config(self):
#         # Typical settings
#         fname = "settings_typ.cfg"
#         expected = {
#             "windowlength": 60.,
#             "filter_bool": False,
#             "filter_flow": 0.1,
#             "filter_fhigh": 30.,
#             "filter_forder": 5,
#             "width": 0.1,
#             "bandwidth": 40.,
#             "resample_fmin": 0.1,
#             "resample_fmax": 50.,
#             "resample_fnum": 128,
#             "resample_type": "log",
#             "peak_f_lower": None,
#             "peak_f_upper": None,
#             "method": "geometric-mean",
#             "azimuth": 0.,
#             "azimuthal_interval": 15.,
#             "rejection_bool": True,
#             "n": 2.,
#             "max_iterations": 50,
#             "distribution_f0": "lognormal",
#             "distribution_mc": "lognormal",
#             "ymin": None,
#             "ymax": None
#         }
#         returned = parse_config(f"{self.full_path}data/cli/{fname}")
#         self.assertDictEqual(expected, returned)

#         # Typical settings (alternate)
#         fname = "settings_typ_alt.cfg"
#         expected = {
#             "windowlength": 30.,
#             "filter_bool": True,
#             "filter_flow": 0.1,
#             "filter_fhigh": 30.,
#             "filter_forder": 5,
#             "width": 0.15,
#             "bandwidth": 50.,
#             "resample_fmin": 0.2,
#             "resample_fmax": 20.,
#             "resample_fnum": 64,
#             "resample_type": "log",
#             "peak_f_lower": 0.2,
#             "peak_f_upper": 0.5,
#             "method": "geometric-mean",
#             "azimuth": 0.,
#             "azimuthal_interval": 15.,
#             "rejection_bool": True,
#             "n": 2.5,
#             "max_iterations": 40,
#             "distribution_f0": "lognormal",
#             "distribution_mc": "lognormal",
#             "ymin": 0,
#             "ymax": 10
#         }
#         returned = parse_config(f"{self.full_path}data/cli/{fname}")
#         self.assertDictEqual(expected, returned)

#     @unittest.skip("Ignore for now")
#     def test_cli(self):
#         # Simple - use config.
#         fname = "UT.STN11.A2_C150"
#         subprocess.run(["hvsrpy", "--no_time", "--config",
#                         f"{self.full_path}data/cli/settings_cli_simple.cfg",
#                         f"{self.full_path}data/a2/{fname}.miniseed"],
#                        check=True)
#         self.assertTrue(os.path.exists(f"{fname}_hvsrpy.hv"))
#         self.assertTrue(os.path.exists(f"{fname}.png"))
#         os.remove(f"{fname}_hvsrpy.hv")
#         os.remove(f"{fname}.png")

#         # Azimuthal - use config.
#         fname = "UT.STN11.A2_C150"
#         subprocess.run(["hvsrpy", "--no_time", "--config",
#                         f"{self.full_path}data/cli/settings_cli_azimuthal.cfg",
#                         f"{self.full_path}data/a2/{fname}.miniseed"],
#                        check=True)
#         self.assertTrue(os.path.exists(f"{fname}_hvsrpy_az.hv"))
#         self.assertTrue(os.path.exists(f"{fname}_az.png"))
#         os.remove(f"{fname}_hvsrpy_az.hv")
#         os.remove(f"{fname}_az.png")


# if __name__ == "__main__":
#     unittest.main()
