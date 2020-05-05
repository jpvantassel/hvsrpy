# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
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

"""HvsrVault class definition."""

import logging

import numpy as np
from numpy.random import default_rng, PCG64, MT19937, BitGenerator
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon

logger = logging.getLogger(name=__name__)

__all__ = ["montecarlo_f0", "HvsrVault"]

def _statistics(values, weights):
    """Calculate weighted mean and stddev.

    Parameters
    ----------
    values : ndarray
        Of shape `(M, N)`, where the rows are the realizations at each
        location and the columns are a given realization for the
        entire region.
    weights : ndarray
        Of size `N`, where `N` is the number of generating locations.
        `N`. Note that the weights will be normalized such that their
        sum is equal to 1.

    Returns
    -------
    tuple
        Of the form `(mean, stddev)` where `mean` is the weighted
        mean and `stddev` the weighted standard deviation.

    """
    norm_weights = weights/np.sum(weights)

    # Mean
    mean = 0
    for row_value, weight in zip(values, norm_weights):
        mean += weight*np.sum(row_value)
    mean /= len(row_value)

    # Stddev
    numerator = 0
    w2 = 0
    for row_value, weight in zip(values, norm_weights):
        diff = row_value - mean
        numerator += weight*np.sum(diff*diff)
        w2 += np.sum(weight*weight)
    numerator /= len(row_value)
    w2 /= len(row_value)
    stddev = np.sqrt(numerator/(1-w2))

    return (mean, stddev)

def montecarlo_f0(mean, stddev, weights, dist_generators="lognormal",
                  dist_spatial="lognormal", nrealizations=1000,
                  generator="PCG64"):
    """MonteCarlo calculation for spatial distribution of f0.

    Parameters
    ----------
    mean, stddev : ndarray
        Mean and standard deviation of each generating point.
        Meaning of these parameters are dictated by `dist_generators`.
    weights : ndarray
        Weights for each generating point.
    dist_generators : {'lognormal', 'normal'}, optional
        Assumed distribution of each generating point, default is
        `lognormal`.
    dist_spatial : {'lognormal', 'normal'}, optional
        Assumed distribution of spatial statistics on f0, default is
        `lognormal`.
    generator : {'PCG64', 'MT19937'}, optional
        Bit generator, default is `PCG64`.

    Returns
    -------
    tuple
        Of the form (f0_mean, f0_stddev, f0s_spatial, f0s_generators).

    """
    if generator == "PCG64":
        rng = default_rng(PCG64())
    elif generator == "MT19937":
        rng = default_rng(MT19937())
    elif isinstance(generator, BitGenerator):
        rng = default_rng(generator)
    else:
        raise ValueError(f"generator type {generator} not recognized.")

    if dist_generators == "normal":
        def realization(mean, stddev):
            return rng.normal(mean, stddev, size=nrealizations)
    elif dist_generators == "lognormal":
        def realization(_lambda, _zeta):
            print("Howdy")
            return np.log(rng.lognormal(_lambda, _zeta, size=nrealizations))
    else:
        raise NotImplementedError

    realizations = np.empty((mean.size, nrealizations))
    for r, (_mean, _stddev) in enumerate(zip(mean, stddev)):
        realizations[r, :] = realization(_mean, _stddev)

    f0_mean, f0_stddev = _statistics(realizations, weights)

    return (f0_mean, f0_stddev, realizations)


class HvsrVault():
    """A container for Hvsr objects.

    Attributes
    ----------
    TODO

    """

    @staticmethod
    def lat_lon_to_x_y(lat, lon):
        raise NotImplementedError
        # TODO (jpv): Will be quite useful.

    def __init__(self, points, f0, sigmalnf0=None):
        """For now we are just going to store the statistis.

        Parameters
        ----------
        points: ndarray
            With the relative x and y coordinates of the Voronoi generators,
            where each row is of the `ndarray` represents an x, y pair.
        boudary: ndarray
            Points which define the unique bounding points for the Voronoi
            tesselations.
        f0: ndarray
            f0 for each point.
        sigmalnf0: ndarray, optional
            sigmalnf0 for each point

        """

        # Validate points
        points = np.array(points)
        if points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N,2), not {points.shape}.")
        self.points = points

        self.f0 = f0
        self.siglnf0 = sigmalnf0

    def spatial_weights(self, boundary, dc_method="voronoi"):
        """Calculate the weights for each voronoi region.

        Parameters
        ----------
        boundary: ndarray
            x, y points defining the boundary boundary.shape must be
            (N, 2)
        dc_method: {"voronoi"}, optional
            Declustering method, default is 'voronoi'.

        Return
        ------
        ndarray
            Statistical weights for each point.

        """
        if dc_method == "voronoi":
            weights, indices = self._voronoi_weights(boundary)
        else:
            raise NotImplementedError
        return (weights, indices)

    @staticmethod
    def _boundary_to_mask(boundary):
        boundary = np.array(boundary)
        if boundary.shape[1] != 2:
            msg = f"boudary must have shape (N,2), not {boundary.shape}."
            raise ValueError(msg)
        bounding_pts = MultiPoint([Point(i) for i in boundary])
        mask = bounding_pts.convex_hull
        return mask

    def _voronoi_weights(self, boundary):
        """Calculate the voronoi geometry weights."""
        mask = self._boundary_to_mask(boundary)
        total_area = mask.area

        regions, indices = self._bounded_voronoi(mask)

        areas = np.empty(len(regions))
        for i, region in enumerate(regions):
            closed_points = np.vstack((region, region[0]))
            poly = Polygon(closed_points)
            areas[i] = poly.area

        return (areas/total_area, indices)

    def _culled_points(self, mask):
        # Remove points not within bounding region
        culled_points, passing_indices = [], []
        for index, (x, y) in enumerate(self.points):
            p = Point(x, y)
            if mask.contains(p):
                culled_points.append([x, y])
                passing_indices.append(index)
            else:
                logger.info(f"Discarding point ({x}, {y})")
        return np.array(culled_points), passing_indices

    def bounded_voronoi(self, boundary):
        mask = self._boundary_to_mask(boundary)
        return self._bounded_voronoi(mask)

    def _bounded_voronoi(self, mask):
        """Vertices of bounded voronoi region.

        Parameters
        ----------
        points: ndarray
            With the relative x and y coordinates of the Voronoi generators,
            where each row is of the `ndarray` represents an x, y pair.

        Returns
        -------
        tuple
            (List of ndarrays, list) new veritces and passing indices

        """
        # Points inside bounding mask
        points, indices = self._culled_points(mask)

        # Define semi-infinite Voronoi tesselations
        vor = Voronoi(points)
        regions, vertices = self._voronoi_finite_polygons_2d(vor, radius=1E6)

        # Define bounded Voronoi tesselations
        new_vertices = []
        for region in regions:
            unique_points = vertices[region]
            closed_points = np.vstack((unique_points, unique_points[0]))
            polygon_before = Polygon(closed_points)

            polygon_after = polygon_before.intersection(mask)

            xs, ys = polygon_after.boundary.xy
            new_unique_points = np.array(list(zip(xs[:-1], ys[:-1])))
            new_vertices.append(new_unique_points)

        return new_vertices, indices

    @staticmethod
    def _voronoi_finite_polygons_2d(vor, radius=None):
        """Convert infinite 2D Voronoi regions a finite regions.

        Parameters
        ----------
        vor: Voronoi
            Voronoi object
        radius: float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions: list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices: list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        Notes
        -----
        This function modified a function originally released by Pauli
        Virtanen. (https: // gist.github.com/pv/8036995).

        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge
                # tangent
                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                # normal
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return (new_regions, np.asarray(new_vertices))
