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

"""Tests for utils module."""

import sys
import os

import pandas as pd
import numpy as np

from hvsrpy import utils
from testtools import TestCase, unittest, get_full_path


class Test_Utils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.full_path = get_full_path(__file__)

    def test_sesame_by_condition(self):
        # Defaults
        frequency = np.array([0.7, 0.8, 0.9, 1, 1.1])
        mean_curve = np.array([1, 1, 2.1, 1, 1])
        std_curve = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        f0_std = 0.1

        # Condition 1: there exists an f in [f0/4, f0] where A(f) < A(f0)/2
        frequency = np.array([0.2, 0.4, 0.7, 1, 1.5])
        potential_curves = [(1, np.array([1.0, 1.0, 1.0, 3.0, 1.0])),
                            (0, np.array([2.9, 2.9, 2.9, 3.0, 1.0])),
                            (0, np.array([1.6, 2.0, 2.9, 3.0, 1.0])),
                            (1, np.array([1.4, 1.4, 2.0, 3.0, 1.0])),
                            (0, np.array([1.5, 1.5, 2.9, 3.0, 1.0]))
                            ]

        for expected, mean_curve in potential_curves:
            returned = utils.sesame_clarity(frequency, mean_curve,
                                            std_curve, f0_std,
                                            verbose=0)[0]
            self.assertEqual(expected, returned)

        # Condition 2: there exists an f in [f0, 4f0] where A(f) < A(f0)/2
        frequency = np.array([0.7, 1, 2.5, 3.5, 5])
        potential_curves = [(1, np.array([1.0, 3.0, 1.0, 1.0, 1.0])),
                            (0, np.array([1.0, 3.0, 2.9, 2.9, 2.9])),
                            (0, np.array([1.0, 3.0, 2.9, 2.0, 1.6])),
                            (1, np.array([1.0, 3.0, 2.0, 1.4, 1.4])),
                            (0, np.array([1.0, 3.0, 2.9, 1.5, 1.5]))
                            ]

        for expected, mean_curve in potential_curves:
            returned = utils.sesame_clarity(frequency, mean_curve,
                                            std_curve, f0_std,
                                            verbose=0)[1]
            self.assertEqual(expected, returned)

        # Condition 3: A0>2
        potential_curves = [(0, np.array([1.0, 1.0, 1.1, 1.0, 1.0])),
                            (0, np.array([1.0, 2.0, 1.0, 1.0, 1.0])),
                            (0, np.array([1.0, 1.0, 1.0, 1.2, 1.0])),
                            (1, np.array([1.0, 3.0, 2.0, 1.4, 1.4])),
                            (1, np.array([1.0, 5.0, 2.9, 1.5, 1.5])),
                            (1, np.array([1.0, 3.0, 2.9, 7.5, 1.5]))
                            ]
        for expected, mean_curve in potential_curves:
            returned = utils.sesame_clarity(frequency, mean_curve,
                                            std_curve, f0_std,
                                            verbose=0)[2]
            self.assertEqual(expected, returned)

        # Condition 4: fpeak[Ahv(f) +/- sigma_a(f)] = f0 +/- 5%
        frequency = np.array([0.90, 0.94, 0.96, 1, 1.04, 1.06, 1.1])

        mean_curves = [np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
                       np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])]

        uppers = [np.array([1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0])+0.1,
                  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0])+0.1,
                  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0])+0.1,
                  np.array([1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0])+0.1,
                  np.array([1.0, 1.0, 1.0, 2.1, 1.0, 1.0, 1.0])+0.1,
                  np.array([1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0])+0.1,
                  np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0])+0.1]

        rating = [0, 0, 0, 0, 1, 1, 1]

        for expected, upper, mean_curve in zip(rating, uppers, mean_curves):
            std_curve = np.log(upper) - np.log(mean_curve)
            curve = np.exp(np.log(mean_curve) + std_curve)
            self.assertArrayEqual(upper, curve)
            returned = utils.sesame_clarity(frequency, mean_curve,
                                            std_curve, f0_std,
                                            verbose=0)[3]
            self.assertEqual(expected, returned)

        # Condition 5: f0_std < epsilon(f0)
        frequency = np.array([0.05, 0.1, 0.35, 0.75, 1.5, 3, 5])
        factors = [0.25, 0.2, 0.15, 0.1, 0.05]

        for peak_frequency, factor in zip(frequency[1:-1], factors):
            mean_curve = np.ones_like(frequency)
            mean_curve[peak_frequency == frequency] = 2
            std_curve = 0.1*mean_curve
            for adjust, expected in zip([0.9, 1.1], [1, 0]):
                f0_std = factor*peak_frequency*adjust
                returned = utils.sesame_clarity(frequency, mean_curve,
                                                std_curve, f0_std,
                                                verbose=0)[4]
                self.assertEqual(expected, returned)

        # Condition 6: sigma_a(f0) < theta(f0)
        frequency = np.array([0.05, 0.1, 0.35, 0.75, 1.5, 3, 5])
        factors = [3.0, 2.5, 2.0, 1.78, 1.58]

        for peak_frequency, factor in zip(frequency[1:-1], factors):
            mean_curve = np.ones_like(frequency)
            mean_curve[peak_frequency == frequency] = 2
            for adjust, expected in zip([0.9, 1.1], [1, 0]):
                sigma_a = np.ones_like(frequency)*adjust*factor
                upper = mean_curve * sigma_a
                std_curve = np.log(upper) - np.log(mean_curve)
                curve = np.exp(np.log(mean_curve) + std_curve)
                self.assertArrayAlmostEqual(upper, curve, places=6)
                f0_std = factor*peak_frequency*adjust
                returned = utils.sesame_clarity(frequency, mean_curve,
                                                std_curve, f0_std,
                                                verbose=0)[5]
                self.assertEqual(expected, returned)

    def test_sesame_with_limits(self):
        frequency = np.array([0.3, 0.4, 0.5, 0.6, 0.9, 1, 1.1, 1.2, 1.3])
        mean_curve = np.array([1, 3, 1, 1, 1, 1, 1, 3, 1])
        std_curve = np.ones(9)*0.2
        f0_std = 0.07

        # reliability
        expecteds = [3, 3]
        settings = [(0.3, 0.5), (0.9, 1.3)]

        for expected, limits in zip(expecteds, settings):
            with open(os.devnull, "w") as sys.stdout:
                returned = np.sum(utils.sesame_reliability(60, 10, frequency,
                                                           mean_curve, std_curve, search_limits=limits,
                                                           verbose=2))
                sys.stdout = sys.__stdout__
            self.assertEqual(expected, returned)

        # clarity
        expecteds = [6, 6]
        settings = [(0.3, 0.5), (0.9, 1.3)]

        for expected, limits in zip(expecteds, settings):
            with open(os.devnull, "w") as sys.stdout:
                returned = np.sum(utils.sesame_clarity(frequency, mean_curve,
                                                       std_curve, f0_std,
                                                       search_limits=limits,
                                                       verbose=2))
                sys.stdout = sys.__stdout__
            self.assertEqual(expected, returned)

    def test_sesame_by_case(self):

        def load(fname, verbose=0):
            data = utils.parse_hvsrpy_output(fname)
            std_curve = np.log(data["upper"]) - np.log(data["curve"])
            clarity = utils.sesame_clarity(data["frequency"],
                                           data["curve"],
                                           std_curve,
                                           data["std_f0"],
                                           verbose=verbose)
            reliability = utils.sesame_reliability(data["windowlength"],
                                                   data["accepted_windows"],
                                                   data["frequency"],
                                                   data["curve"],
                                                   std_curve,
                                                   verbose=verbose)
            return (reliability, clarity)

        expecteds = [([1, 1, 1], [1, 1, 1, 1, 0, 1]),
                     ([1, 1, 1], [1, 1, 1, 1, 1, 1]),
                     ([1, 1, 1], [1, 1, 1, 1, 0, 1]),
                     ([1, 1, 1], [1, 1, 1, 1, 1, 1]),
                     ([1, 1, 1], [1, 1, 1, 1, 1, 1])]

        for count, expected_tuple in enumerate(expecteds):
            fname = self.full_path + f"data/utils/ex{count}.hv"
            with open(os.devnull, "w") as sys.stdout:
                returned_tuple = load(fname)
                sys.stdout = sys.__stdout__
            for expected, returned in zip(expected_tuple, returned_tuple):
                expected = np.array(expected)
                self.assertArrayEqual(expected, returned)

    def test_parse_hvsrpy_output(self):

        def compare_data_dict(expected_dict, returned_dict):
            for key, value in expected_dict.items():
                if isinstance(value, (str, int, bool)):
                    self.assertEqual(expected_dict[key], returned_dict[key])
                elif isinstance(value, float):
                    self.assertAlmostEqual(expected_dict[key],
                                           returned_dict[key], places=3)
                elif isinstance(value, np.ndarray):
                    self.assertArrayAlmostEqual(expected_dict[key],
                                                returned_dict[key], places=3)
                else:
                    raise ValueError

        # Ex 0
        fname = self.full_path + "data/utils/ex0.hv"
        expected = {"windowlength": 60.0,
                    "total_windows": 30,
                    "rejection_bool": True,
                    "n_for_rejection": 2.0,
                    "accepted_windows": 29,
                    "distribution_f0": "normal",
                    "mean_f0": 0.6976,
                    "std_f0": 0.1353,
                    "distribution_mc": "log-normal",
                    "f0_mc": 0.7116,
                    "amplitude_f0_mc": 3.8472,
                    }
        names = ["frequency", "curve", "lower", "upper"]
        df = pd.read_csv(fname, comment="#", names=names)
        for name in names:
            expected[name] = df[name].to_numpy()
        returned = utils.parse_hvsrpy_output(fname)
        compare_data_dict(expected, returned)

        # Ex 1
        fname = self.full_path + "data/utils/ex1.hv"
        expected = {"windowlength": 60.0,
                    "total_windows": 60,
                    "rejection_bool": True,
                    "n_for_rejection": 2.0,
                    "accepted_windows": 48,
                    "distribution_f0": "normal",
                    "mean_f0": 0.7155,
                    "std_f0": 0.0759,
                    "distribution_mc": "log-normal",
                    "f0_mc": 0.7378,
                    "amplitude_f0_mc": 3.9661,
                    }
        names = ["frequency", "curve", "lower", "upper"]
        df = pd.read_csv(fname, comment="#", names=names)
        for name in names:
            expected[name] = df[name].to_numpy()
        returned = utils.parse_hvsrpy_output(fname)
        compare_data_dict(expected, returned)

        # Ex 2
        fname = self.full_path + "data/utils/ex2.hv"
        expected = {"windowlength": 60.0,
                    "total_windows": 30,
                    "rejection_bool": True,
                    "n_for_rejection": 2.0,
                    "accepted_windows": 29,
                    "distribution_f0": "normal",
                    "mean_f0": 0.7035,
                    "std_f0": 0.1391,
                    "distribution_mc": "log-normal",
                    "f0_mc": 0.7116,
                    "amplitude_f0_mc": 3.9097,
                    }
        names = ["frequency", "curve", "lower", "upper"]
        df = pd.read_csv(fname, comment="#", names=names)
        for name in names:
            expected[name] = df[name].to_numpy()
        returned = utils.parse_hvsrpy_output(fname)
        compare_data_dict(expected, returned)

        # Ex 3
        fname = self.full_path + "data/utils/ex3.hv"
        expected = {"windowlength": 60.0,
                    "total_windows": 60,
                    "rejection_bool": True,
                    "n_for_rejection": 2.0,
                    "accepted_windows": 33,
                    "distribution_f0": "normal",
                    "mean_f0": 0.7994,
                    "std_f0": 0.0365,
                    "distribution_mc": "log-normal",
                    "f0_mc": 0.7933,
                    "amplitude_f0_mc": 4.8444,
                    }
        names = ["frequency", "curve", "lower", "upper"]
        df = pd.read_csv(fname, comment="#", names=names)
        for name in names:
            expected[name] = df[name].to_numpy()
        returned = utils.parse_hvsrpy_output(fname)
        compare_data_dict(expected, returned)


if __name__ == "__main__":
    unittest.main()
