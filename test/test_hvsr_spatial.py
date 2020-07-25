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

"""Tests for spatial module."""

import logging

import numpy as np
from numpy.random import PCG64

import hvsrpy.hvsr_spatial as spatial
from testtools import unittest, TestCase

logging.basicConfig(level=logging.WARNING)


class Test_Spatial(TestCase):
    def test_statistics(self):
        values = np.array([[1, 3, 5], [1, 5, 6], [1, 1, 1], [3, 2, 4]])
        weights = np.array([1, 2, 4, 3])

        # Mean
        expected = 2.4
        returned, _ = spatial._statistics(values, weights)
        self.assertEqual(expected, returned)

        # Stddev
        expected = 1.776388346
        _, returned = spatial._statistics(values, weights)

    def test_montecarlo_f0(self):
        means = np.array([0.2, 0.4, 0.6, 0.5])
        stds = np.array([0.05, 0.07, 0.1, 0.01])
        weights = np.array([1, 2, 4, 5])

        # Normal
        generator = PCG64(1994)
        vals = spatial.montecarlo_f0(means, stds, weights,
                                     dist_generators="normal",
                                     dist_spatial="normal",
                                     nrealizations=3,
                                     generator=generator)
        mean, stddev, realizations = vals

        # Realizations
        expected = np.array([[0.2165,  0.2017,  0.2639],
                             [0.4322,  0.3953,  0.4572],
                             [0.5411,  0.6164,  0.6436],
                             [0.5244,  0.5015,  0.5080]])
        returned = realizations
        self.assertArrayAlmostEqual(expected, returned, places=3)

        # Mean
        expected = 0.5035
        returned = mean
        self.assertAlmostEqual(expected, returned, places=3)

        # Stddev
        expected = 0.1124
        returned = stddev
        self.assertAlmostEqual(expected, returned, places=3)

        # LogNormal
        generator = PCG64(1994)
        vals = spatial.montecarlo_f0(means, stds, weights,
                                     dist_generators="lognormal",
                                     dist_spatial="lognormal",
                                     nrealizations=3,
                                     generator=generator)
        mean, stddev, realizations = vals

        # Realizations
        expected = np.exp(np.array([[0.2165,  0.2017,  0.2639],
                                    [0.4322,  0.3953,  0.4572],
                                    [0.5411,  0.6164,  0.6436],
                                    [0.5244,  0.5015,  0.5080]]))
        returned = realizations
        self.assertArrayAlmostEqual(expected, returned, places=3)

        # Mean
        expected = np.exp(0.5035)
        returned = mean
        self.assertAlmostEqual(expected, returned, places=3)

        # Stddev
        expected = 0.1124
        returned = stddev
        self.assertAlmostEqual(expected, returned, places=3)

        # Other generators
        for generator in ["PCG64", "MT19937"]:
            spatial.montecarlo_f0(means, stds, weights,
                                  generator=generator)

        # Bad generator
        generator = "my fancy generator"
        self.assertRaises(ValueError, spatial.montecarlo_f0,
                          means, stds, weights, generator=generator)

        # Bad dist_generator
        dist_generators = "my fancy generator"
        self.assertRaises(NotImplementedError,
                          spatial.montecarlo_f0, means, stds,
                          weights, dist_generators=dist_generators)

        # Bad dist_spatial
        dist_spatial = "my fancy distribution"
        self.assertRaises(NotImplementedError,
                          spatial.montecarlo_f0, means, stds,
                          weights, dist_spatial=dist_spatial)


if __name__ == "__main__":
    unittest.main()
