# This file is part of hvsrpy a Python package for horizontal-to-vertical
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

"""This file contains the class HvsrRotated."""

import numpy as np
from hvsrpy import Hvsr
import logging
logger = logging.getLogger(__name__)

class HvsrRotated():
    """Class definition for rotated Horizontal-to-Vertical calculations.

    Attributes
    ----------
    hvsrs : list
        Container of `Hvsr` objects, one per azimuth.
    azimuths : ndarray
        Vector of rotation azimuths correpsonding to `Hvsr` objects.

    """

    def __init__(self, hvsr, azimuth):
        """Instantiate a `HvsrRotated` object.

        Parameters
        ----------
        hvsr : Hvsr
            `Hvsr` object.
        azimuth : float
            Rotation angle in degrees measured anti-clockwise positve
            from north (i.e., 0 degrees).

        Returns
        -------
        HvsrRotated
            Instantiated `HvsrRotated` object.

        """
        hvsr, azimuth = self._check_input(hvsr, azimuth)
        self.hvsrs = [hvsr]
        self.azimuths = [azimuth]

    def _check_input(self, hvsr, az):
        """Check input, specifically:
            1. `hvsr` is an instance of `Hvsr`.
            2. Cast `az` to float (if it is not already).
            3. `az` is greater than 0.

        """
        if not isinstance(hvsr, Hvsr):
            raise TypeError("`hvsr` must be an instance of `Hvsr`.")
        
        az = float(az)

        if az < 0:
            raise ValueError(f"`azimuth` must be greater than 0, not {az}.")

        return hvsr, az

    def append(self, hvsr, azimuth):
        """Append `Hvsr` object at a new azimuth.
        
        Parameters
        ----------
        hvsr : Hvsr
            `Hvsr` object.
        az : float
            Rotation angle in degrees measured anti-clockwise positve
            from north (i.e., 0 degrees).

        Returns
        -------
        HvsrRotated
            Instantiated `HvsrRotated` object.

        """
        hvsr, azimuth = self._check_input(hvsr, azimuth)
        self.hvsrs.append(hvsr)
        self.azimuths.append(azimuth)

    @classmethod
    def from_iter(cls, hvsrs, azimuths):
        """Create HvsrRotated from iterable of Hvsr objects."""
        obj = cls(hvsrs[0], azimuths[0])
        if len(azimuths)>1:
            for hvsr, az in zip(hvsrs[1:], azimuths[1:]):
                obj.append(hvsr, az)
        return obj

    def reject_windows(self, **kwargs):
        for hv in self.hvsrs:
            hv.reject_windows(**kwargs)
