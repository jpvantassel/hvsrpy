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


def mean_factory(distribution, values, mean_kwargs=None):
    """Calculates mean of ``values`` consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    if mean_kwargs is None:
        mean_kwargs = {}

    distribution = DISTRIBUTION_MAP.get(distribution.lower(), None)
    if distribution == "normal":
        return np.mean(values, **mean_kwargs)
    elif distribution == "lognormal":
        return np.exp(np.mean(np.log(values), **mean_kwargs))
    else:
        msg = f"distribution type {distribution} not recognized."
        raise NotImplementedError(msg)


def std_factory(distribution, values, std_kwargs=None):
    """Calculates standard deviation consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    if std_kwargs is None:
        std_kwargs = dict(ddof=1)

    distribution = DISTRIBUTION_MAP.get(distribution.lower(), None)
    if distribution == "normal":
        return np.std(values, **std_kwargs)
    elif distribution == "lognormal":
        return np.std(np.log(values), **std_kwargs)
    else:
        msg = f"distribution type {distribution} not recognized."
        raise NotImplementedError(msg)


def nth_std_factory(n, distribution, mean, std):
    """Calculates nth standard deviation consistent with distribution.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    distribution = DISTRIBUTION_MAP.get(distribution, None)
    if distribution == "normal":
        return (mean + n*std)
    elif distribution == "lognormal":
        return (np.exp(np.log(mean) + n*std))
    else:
        msg = f"distribution type {distribution} not recognized."
        raise NotImplementedError(msg)


def flatten_list(unflattened_list):
    """Flattens ``list`` of lists to single flattened ``list``.

    .. warning:: 
        Private methods are subject to change without warning.

    """
    flattened_list = []
    for _list in unflattened_list:
        flattened_list.extend(_list)
    return flattened_list
