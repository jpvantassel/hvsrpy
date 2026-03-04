# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2026 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Class definition for Fas object."""

import logging

import numpy as np

from .frequency_amplitude_curve import FrequencyAmplitudeCurve

logger = logging.getLogger(__name__)

__all__ = ["Fas"]


class Fas():
    """Class for creating and manipulating ``Fas`` objects.

    Attributes
    ----------
    frequency : ndarray
        Vector of frequencies, must be same length as ``amplitude``.
    amplitude : ndarray
        Vector of Fas amplitude values, one value per ``frequency``.

    """

    def __init__(self, frequency, amplitude, meta=None):
        """Create ``Fas`` from iterables of frequency and amplitude.

        Parameters
        ----------
        frequency : ndarray
            Vector of frequencies, one per ``amplitude``.
        amplitude : ndarray
            Array of Fas amplitudes, one row per ``curve`` one column per ``frequency``.
        meta : dict, optional
            Meta information about the object, default is ``None``.

        Returns
        -------
        Fas
            Initialized with ``amplitude`` and ``frequency``.

        """
        self.frequency = FrequencyAmplitudeCurve._check_input(frequency, "frequency")
        self.amplitude = np.atleast_2d(FrequencyAmplitudeCurve._check_input(amplitude, "amplitude"))

        if len(self.frequency) != self.amplitude.shape[1]:
            msg = f"Shape of amplitude={self.amplitude.shape} and "
            msg += f"frequency={self.frequency.shape} must be compatible."
            raise ValueError(msg)

        self.n_curves = len(self.amplitude)
        self.meta = dict(meta) if isinstance(meta, dict) else {}

    def is_similar(self, other):
        """Determine if ``other`` is similar to ``self``."""
        if not isinstance(other, Fas):
            return False

        if len(self.frequency) != len(other.frequency):
            return False

        if not np.allclose(self.frequency, other.frequency):
            return False

        return True

    def __eq__(self, other):
        """Determine if ``other`` is equal to ``self``."""
        if not self.is_similar(other):
            return False

        if self.n_curves != other.n_curves:
            return False

        if not np.allclose(self.amplitude, other.amplitude):
            return False

        return True

    def __str__(self):
        """Human-readable representation of ``Fas`` object."""
        return f"Fas at {id(self)}"

    def __repr__(self):
        """Unambiguous representation of ``Fas`` object."""
        return f"Fas(frequency={self.frequency}, amplitude={self.amplitude}, meta={self.meta})"
