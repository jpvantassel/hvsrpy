# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2021 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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
from testing_tools import unittest, TestCase

logging.basicConfig(level=logging.WARNING)


class TestSpatial(TestCase):
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

    def test_montecarlo_fn(self):
        means = np.array([0.2, 0.4, 0.6, 0.5])
        stds = np.array([0.05, 0.07, 0.1, 0.01])
        weights = np.array([1, 2, 4, 5])

        # Normal
        rng = np.random.default_rng(1994)
        vals = spatial.montecarlo_fn(means, stds, weights,
                                     distribution_generators="normal",
                                     distribution_spatial="normal",
                                     n_realizations=3,
                                     rng=rng)
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
        rng = np.random.default_rng(1994)
        vals = spatial.montecarlo_fn(means, stds, weights,
                                     distribution_generators="lognormal",
                                     distribution_spatial="lognormal",
                                     n_realizations=3,
                                     rng=rng)
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

        # bad rng
        rng = "my fancy generator"
        self.assertRaises(AttributeError, spatial.montecarlo_fn,
                          means, stds, weights, rng=rng)

        # Bad dist_generator
        dist_generators = "my fancy generator"
        self.assertRaises(NotImplementedError,
                          spatial.montecarlo_fn, means, stds,
                          weights, distribution_generators=dist_generators)

        # Bad dist_spatial
        dist_spatial = "my fancy distribution"
        self.assertRaises(NotImplementedError,
                          spatial.montecarlo_fn, means, stds,
                          weights, distribution_spatial=dist_spatial)


if __name__ == "__main__":
    unittest.main()
