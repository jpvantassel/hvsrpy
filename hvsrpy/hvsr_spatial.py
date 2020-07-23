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
    """MonteCarlo simulation for spatial distribution of f0.

    Parameters
    ----------
    mean, stddev : ndarray
        Mean and standard deviation of each generating point.
        Meaning of these parameters is dictated by `dist_generators`.
    weights : ndarray
        Weights for each generating point.
    dist_generators : {'lognormal', 'normal'}, optional
        Assumed distribution of each generating point, default is
        `lognormal`.

        +-----------+------------------+-----------------+
        | if dist is| mean must be     | stddev must be  |
        +===========+==================+=================+
        | normal    |:math:`\\mu`       |:math:`\\sigma`   |
        +-----------+------------------+-----------------+
        | lognormal |:math:`\\lambda`   |:math:`\\zeta`    |
        +-----------+------------------+-----------------+

    dist_spatial : {'lognormal', 'normal'}, optional
        Assumed distribution of spatial statistics on f0, default is
        `lognormal`.
    generator : {'PCG64', 'MT19937'}, optional
        Bit generator, default is `PCG64`.

    Returns
    -------
    tuple
        Of the form `(f0_mean, f0_stddev, f0_realizations)`.

    """
    if generator == "PCG64":
        rng = default_rng(PCG64())
    elif generator == "MT19937":
        rng = default_rng(MT19937())
    elif isinstance(generator, BitGenerator):
        rng = default_rng(generator)
    else:
        raise ValueError(f"generator type {generator} not recognized.")

    if dist_generators not in ["normal", "lognormal"]:
        msg = f"dist_generators = {dist_generators} not recognized."
        raise NotImplementedError(msg)

    if dist_spatial not in ["normal", "lognormal"]:
        msg = f"dist_spatial = {dist_spatial} not recognized."
        raise NotImplementedError(msg)

    def realization(mean, stddev, nrealizations=nrealizations):
        return rng.normal(mean, stddev, size=nrealizations)

    realizations = np.empty((len(mean), nrealizations))
    for r, (_mean, _stddev) in enumerate(zip(mean, stddev)):
        realizations[r, :] = realization(_mean, _stddev)

    if dist_generators == "lognormal" and dist_spatial == "normal":
        realizations = np.exp(realizations)
    elif dist_generators == "normal" and dist_spatial == "lognormal":
        realizations = np.log(realizations)
    # elif dist_generators == "normal" and dist_spatial == "normal":
    #     pass
    # elif dist_generators == "lognormal" and dist_spatial == "lognormal":
    #     pass
    # else:
    #     pass

    f0_mean, f0_stddev = _statistics(realizations, weights)

    if dist_spatial == "lognormal":
        f0_mean = np.exp(f0_mean)
        realizations = np.exp(realizations)

    return (f0_mean, f0_stddev, realizations)


class HvsrVault():  # pragma: no cover
    """A container for Hvsr objects.

    Attributes
    ----------
    coordinates : ndarray
        Relative x and y coordinates of the sensors, where each row
        of the `ndarray` in an x, y pair.
    means : ndarray
        Mean f0 value for each location, meaning is determined by
        `distribution` keyword argument.
    stddevs : ndarray, optional
        Standard deviation for each location, meaning is determined
        by `distribution` keyword argument, default is `None`
        indicating no uncertainty is defined.

    """

    # TODO (jpv): Implement a converter from latitude and longitude, to x and y.
    # @staticmethod
    # def lat_lon_to_x_y(lat, lon):
    #     raise NotImplementedError

    def __init__(self, coordinates, means, stddevs=None):  # pragma: no cover
        """Create a container for `Hvsr` statistics.

        Parameters
        ----------
        coordinates : ndarray
            Relative x and y coordinates of the sensors, where each row
            of the `ndarray` in an x, y pair.
        means : ndarray
            Mean f0 value for each location, meaning is determined by
            `distribution` keyword argument.
        stddevs : ndarray, optional
            Standard deviation for each location, meaning is determined
            by `distribution` keyword argument, default is `None`
            indicating no uncertainty is defined.
        distribution : {'normal', 'lognormal'}, optional
            Distribution to which the mean and stddev for each point
            corresponds, default is 'lognormal'.

        """
        coordinates = np.array(coordinates, dtype=np.double)
        npts, dim = coordinates.shape
        if dim != 2:
            msg = f"coordinates must have shape (N,2), not {coordinates.shape}."
            raise ValueError(msg)
        if npts < 3:
            raise ValueError(f"Requires at least three coordinates")

        self.coordinates = coordinates
        self.means = np.array(means, dtype=np.double)

        if stddevs is None:
            self.stddevs = np.zeros_like(self.means)
        else:
            self.stddevs = np.array(stddevs, dtype=np.double)

    def spatial_weights(self, boundary, dc_method="voronoi"):  # pragma: no cover
        """Calculate the weights for each Voronoi region.

        Parameters
        ----------
        boundary: ndarray
            x, y coordinates defining the spatial boundary. Must be of
            shape `(N, 2)`.
        dc_method: {"voronoi"}, optional
            Declustering method, default is 'voronoi'.

        Return
        ------
        tuple
            Of the form `(weights, indices)` where `weights` are the
            statistical weights and `indicates` the bounding box of
            each cell.

        """
        if dc_method == "voronoi":
            weights, indices = self._voronoi_weights(boundary)
        else:
            raise NotImplementedError
        return (weights, indices)

    @staticmethod
    def _boundary_to_mask(boundary):  # pragma: no cover
        """Create mask from iterable of coordinate pairs."""
        boundary = np.array(boundary)
        if boundary.shape[1] != 2:
            msg = f"boundary must have shape (N,2), not {boundary.shape}."
            raise ValueError(msg)
        bounding_pts = MultiPoint([Point(i) for i in boundary])
        return bounding_pts.convex_hull

    def _voronoi_weights(self, boundary):  # pragma: no cover
        """Calculate the voronoi geometry weights."""
        mask = self._boundary_to_mask(boundary)
        total_area = mask.area

        regions, indices = self._bounded_voronoi(mask)

        areas = np.empty(len(regions))
        for i, region in enumerate(regions):
            closed_points = np.vstack((region, region[0]))
            areas[i] = Polygon(closed_points).area

        return (areas/total_area, indices)

    def _cull_points(self, mask):  # pragma: no cover
        """Remove points not within bounding region."""
        passing_points, passing_indices = [], []
        for index, (x, y) in enumerate(self.coordinates):
            p = Point(x, y)
            if mask.contains(p):
                passing_points.append([x, y])
                passing_indices.append(index)
            else:
                logger.info(f"Discarding point ({x}, {y})")
        return (np.array(passing_points), passing_indices)

    def bounded_voronoi(self, boundary):  # pragma: no cover
        """Vertices of bounded Voronoi region.

        Parameters
        ----------
        boundary : ndarray
            x, y coordinates defining the spatial boundary. Must be of
            shape `(N, 2)`.

        Returns
        -------
        tuple
            Of the form `(new_vertices, indices)` where `new_vertices`
            defines the vertices of each region and `indices` indicates
            how these vertices relate to the master statistics.

        """

        mask = self._boundary_to_mask(boundary)
        return self._bounded_voronoi(mask)

    def _bounded_voronoi(self, mask, radius=1E6):  # pragma: no cover
        """Vertices of bounded voronoi region.

        Parameters
        ----------
        mask: ndarray
            Bounding mask to define boundary.

        Returns
        -------
        tuple
            Of the form `(new_vertices, indices)` where `new_vertices`
            defines the vertices of each region and `indices` indicates
            how these vertices relate to master statistics.

        """
        # Points inside bounding mask
        points, indices = self._cull_points(mask)

        # Define semi-infinite Voronoi tesselations
        vor = Voronoi(points)
        regions, vertices = self._voronoi_finite_polygons_2d(vor,
                                                             radius=radius)

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

        return (new_vertices, indices)

    @staticmethod
    def _voronoi_finite_polygons_2d(vor, radius=None):  # pragma: no cover
        """Convert infinite 2D Voronoi regions to finite regions.

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
        This function is a modified version of the one originally
        released by Pauli Virtanen (https://gist.github.com/pv/8036995).

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
