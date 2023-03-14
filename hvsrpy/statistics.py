# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
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

"""Helper functions for statistical calculations."""

import numpy as np

from .constants import DISTRIBUTION_MAP

PRE_PROCESS_FUNCTION_MAP = {
    "normal": {"mean": lambda values: values,
               "std": lambda values: values},
    "lognormal": {"mean": lambda values: np.log(values),
                  "std": lambda values: np.log(values)}
}

POST_PROCESS_FUNCTION_MAP = {
    "normal": {"mean": lambda values: values,
               "std": lambda values: values},
    "lognormal": {"mean": lambda values: np.exp(values),
                  "std": lambda values: values}
}


def _distribution_factory(distribution, calculation="mean"):
    """Provides pre- and post-processing functions.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    try:
        distribution = DISTRIBUTION_MAP.get(distribution.lower(), None)
        preprocess_fxn = PRE_PROCESS_FUNCTION_MAP[distribution][calculation]
        postprocess_fxn = POST_PROCESS_FUNCTION_MAP[distribution][calculation]
    except KeyError:
        msg = f"distribution type {distribution} not recognized."
        raise NotImplementedError(msg)
    return (preprocess_fxn, postprocess_fxn)


def _nanmean_weighted(distribution, values, weights=None, mean_kwargs=None):
    """Calculates weighted mean of ``values`` consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    pre_fxn, post_fxn = _distribution_factory(distribution=distribution,
                                              calculation="mean")

    if mean_kwargs is None:
        mean_kwargs = {}

    values = pre_fxn(values)
    is_nan_mask = np.isnan(values)

    if weights is None:
        weights = np.full_like(values, 1)
        weights[is_nan_mask] = np.nan

    weighted_mean = np.nansum(values*weights, **mean_kwargs) / np.nansum(weights, **mean_kwargs)
    return post_fxn(weighted_mean)


def _nanstd_weighted(distribution, values, weights=None, std_kwargs=None, denominator="nist"):
    """Calculates weighted standard deviation of ``values`` consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    # compute weighted mean.
    mean = _nanmean_weighted(distribution=distribution, values=values,
                             weights=weights, mean_kwargs=std_kwargs)
    if distribution == "lognormal":
        mean = np.log(mean)

    pre_fxn, post_fxn = _distribution_factory(distribution=distribution,
                                              calculation="std")

    if std_kwargs is None:
        std_kwargs = {}

    values = pre_fxn(values)
    is_nan_mask = np.isnan(values)

    if weights is None:
        weights = np.full_like(values, 1)
        weights[is_nan_mask] = np.nan

    numerator = np.nansum(weights*(values-mean)**2, **std_kwargs)
    if denominator == "nist":
        non_nan_weights = np.sum(~np.isnan(weights), **std_kwargs)
        denominator = (1-(1/non_nan_weights))*np.nansum(weights, **std_kwargs)
    elif denominator == "cheng":
        denominator = 1-np.nansum(weights**2, **std_kwargs)
    else: # pragma: no cover
        raise NotImplementedError
    return post_fxn(np.sqrt(numerator / denominator))


def _nth_std_factory(n, distribution, mean, std):
    """Calculates nth standard deviation consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    distribution = DISTRIBUTION_MAP.get(distribution, None)
    if distribution == "normal":
        return (mean + n*std)
    elif distribution == "lognormal":
        return (np.exp(np.log(mean) + n*std))
    else: # pragma: no cover
        msg = f"distribution type {distribution} not recognized."
        raise NotImplementedError(msg)


def _flatten_list(unflattened_list):
    """Flattens ``list`` of lists to single flattened ``list``.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    flattened_list = []
    for _list in unflattened_list:
        flattened_list.extend(_list)
    return flattened_list
