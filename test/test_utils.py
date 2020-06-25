# This file is part of hvsrpy, a Python module for
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
                                            std_curve, f0_std)[0]
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
                                            std_curve, f0_std)[1]
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
                                            std_curve, f0_std)[2]
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
                                            std_curve, f0_std)[3]
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
                                                std_curve, f0_std)[4]
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
                                                std_curve, f0_std)[5]
                self.assertEqual(expected, returned)

    def test_sesame_by_case(self):
        # Case 1: All pass
        frequency = np.array([0.7, 0.8, 0.9, 1, 1.1])
        mean_curve = np.array([1, 1, 2.1, 1, 1])
        std_curve = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        f0_std = 0.1

        returned = utils.sesame_clarity(frequency, mean_curve, std_curve,
                                        f0_std)
        expected = np.array([1.0]*6)
        self.assertArrayEqual(expected, returned)

if __name__ == "__main__":
    unittest.main()
