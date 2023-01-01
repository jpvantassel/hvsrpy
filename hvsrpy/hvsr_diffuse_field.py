# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
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

"""Class definition for HvsrDiffuseField object."""

import logging

import numpy as np

from .metadata import __version__

logger = logging.getLogger(__name__)

__all__ = ["HvsrDiffuseField"]

class HvsrDiffuseField():
    """Class for creating and manipulating HvsrDiffuseField objects.

    Attributes
    ----------
    amplitude : ndarray
        Array of HVSR amplitudes. Each row represents an individual
        curve/time window and each column a frequency.
    frequency : ndarray
        Vector of frequencies corresponds to each column.
    nwindows : int
        Number of windows in `HvsrTraditional` object.
    valid_window_boolean_mask : ndarray
        Boolean array indicating valid windows.

    """
    @staticmethod
    def _check_input(name, value):
        """Basic check on input values.

        Specifically:
            1. `value` must be castable to `ndarray` of doubles.
            2. `value` must be real, (no `np.nan`).
            3. `value` must be > 0.

        Parameters
        ----------
        name : str
            Name of `value` to be checked, used for meaningful error messages.
        value : iterable
            Value to be checked.

        Returns
        -------
        ndarray
            `values` as `ndarray` of doubles.

        Raises
        ------
        TypeError
            If `value` is not castable to an `ndarray` of doubles.
        ValueError
            If `value` contains nan or a value less than or equal to zero.

        """
        try:
            value = np.array(value, dtype=np.double)
        except ValueError:
            msg = f"{name} must be castable to 2D-array of doubles, "
            msg += f"not {type(value)}."
            raise TypeError(msg)

        if np.isnan(value).any():
            raise ValueError(f"{name} may not contain nan.")

        if np.sum(value < 0):
            raise ValueError(f"{name} must be > 0.")

        return value

    def __init__(self, amplitude, frequency, meta=None):
        """Create `Hvsr` from iterable of amplitude and frequency.

        Parameters
        ----------
        amplitude : ndarray
            Array of HVSR amplitudes. Each row represents an individual
            curve/time window and each column a frequency.
        frequency : ndarray
            Vector of frequencies, corresponding to each column.
        meta : dict, optional
            Meta information about the object, default is `None`.

        Returns
        -------
        Hvsr
            Initialized with `amplitude` and `frequency`.

        """
        self.frequency = self._check_input("frequency", frequency)
        self.amplitude = np.atleast_2d(
            self._check_input("amplitude", amplitude))

        if len(self.frequency) != self.amplitude.shape[1]:
            msg = f"Shape of amplitude={self.amplitude.shape} and "
            msg += f"frequency={self.frequency.shape} must be compatible."
            raise ValueError(msg)

        self.nwindows = self.amplitude.shape[0]
        self.valid_window_boolean_mask = np.ones(self.nwindows, dtype=bool)
        self._main_peak_frq = np.empty(self.nwindows)
        self._main_peak_amp = np.empty(self.nwindows)
        self.update_peaks()

        self.meta = dict(meta) if isinstance(meta, dict) else dict()

    @property
    def rejected_window_boolean_mask(self):
        """Boolean array indicating invalid (i.e., rejected) windows."""
        return np.invert(self.valid_window_boolean_mask)

    @property
    def peak_frequencies(self):
        """Valid peak frequency vector, one per window."""
        return self._main_peak_frq[self.valid_window_boolean_mask]

    @property
    def peak_amplitudes(self):
        """Valid peak amplitude vector, one per window."""
        return self._main_peak_amp[self.valid_window_boolean_mask]
