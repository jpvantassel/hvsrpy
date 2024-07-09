# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Class definition for HvsrCurve object."""

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

__all__ = ["HvsrCurve"]


class HvsrCurve():
    """Class for creating and manipulating ``HvsrCurve`` objects.

    Attributes
    ----------
    frequency : ndarray
        Vector of frequencies, must be same length as ``amplitude``.
    amplitude : ndarray
        Vector of HVSR amplitude values, one value per ``frequency``.
    peak_frequency : float
        Frequency of highest amplitude peak of HVSR curve.
    peak_amplitude : float
        Amplitude of highest amplitude peak of HVSR curve.

    """
    @staticmethod
    def _check_input(value, name):
        """Check input values.

        .. warning:: 
            Private methods are subject to change without warning.

        Specifically:
            1. ``value`` must be castable to ``ndarray`` of doubles.
            2. ``value`` must be real; no ``np.nan``.
            3. ``value`` must be >= 0.

        Parameters
        ----------
        value : iterable
            Value to be checked.
        name : str
            Name of ``value`` to be checked, used for meaningful error
            messages.

        Returns
        -------
        ndarray
            ``values`` as ``ndarray`` of doubles.

        Raises
        ------
        TypeError
            If ``value`` is not castable to an ``ndarray`` of doubles.
        ValueError
            If ``value`` contains nan or a value less than or equal to zero.

        """
        try:
            value = np.array(value, dtype=np.double)
        except ValueError:
            msg = f"{name} must be castable to array of doubles, "
            msg += f"not {type(value)}."
            raise TypeError(msg)

        if np.isnan(value).any():
            raise ValueError(f"{name} may not contain nan.")

        if (value < 0).any():
            raise ValueError(f"{name} must be >= 0.")

        return value

    @staticmethod
    def _find_peak_unbounded(frequency, amplitude, find_peaks_kwargs=None):
        """Finds frequency and amplitude associated with highest peak.

        .. warning:: 
            Private methods are subject to change without warning.

        """
        if find_peaks_kwargs is None:
            find_peaks_kwargs = {}
        potential_peak_indices, _ = find_peaks(amplitude, **find_peaks_kwargs)

        # If no peaks found, then indices array will be empty.
        if len(potential_peak_indices) == 0:
            return (None, None)

        potential_peak_amplitudes = amplitude[potential_peak_indices]
        sub_idx = np.argmax(potential_peak_amplitudes)
        return (frequency[potential_peak_indices[sub_idx]],
                amplitude[potential_peak_indices[sub_idx]])

    @staticmethod
    def _search_range_to_index_range(frequency, search_range_in_hz):
        """Convert search range values in Hz to index range values.

        .. warning:: 
            Private methods are subject to change without warning.

        """
        f_low, f_high = search_range_in_hz

        # low frequency limit.
        if f_low is None:
            f_low_idx = 0
        else:
            f_low_idx = np.argmin(np.abs(frequency - f_low))

        # high frequency limit.
        if f_high is None:
            f_high_idx = len(frequency)
        else:
            f_high_idx = np.argmin(np.abs(frequency - f_high))

        return (f_low_idx, f_high_idx)

    @staticmethod
    def _find_peak_bounded(frequency, amplitude, search_range_in_hz=(None, None), find_peaks_kwargs=None):
        """Finds frequency and amplitude associated with highest peak over a bounded range.

        .. warning::
            Private methods are subject to change without warning.

        """
        f_low_idx, f_high_idx = HvsrCurve._search_range_to_index_range(frequency,
                                                                       search_range_in_hz)
        (frequency, amplitude) = HvsrCurve._find_peak_unbounded(frequency[f_low_idx:f_high_idx],
                                                                amplitude[f_low_idx:f_high_idx],
                                                                find_peaks_kwargs=find_peaks_kwargs)
        return (frequency, amplitude)

    def __init__(self, frequency, amplitude, meta=None):
        """Create ``HvsrCurve`` from iterables of frequency and amplitude.

        Parameters
        ----------
        frequency : ndarray
            Vector of frequencies, one per ``amplitude``.
        amplitude : ndarray
            Vector of HVSR amplitudes, one per ``frequency``.
        meta : dict, optional
            Meta information about the object, default is `None`.

        Returns
        -------
        HvsrCurve
            Initialized with ``amplitude`` and ``frequency``.

        """
        self.frequency = self._check_input(frequency, "frequency")
        self.amplitude = self._check_input(amplitude, "amplitude")

        if len(self.frequency) != len(self.amplitude):
            msg = f"Length of amplitude {len(self.amplitude)} and length"
            msg += f"of frequency {len(self.amplitude)} must be agree."
            raise ValueError(msg)

        self.meta = dict(meta) if isinstance(meta, dict) else dict()

        self._search_range_in_hz = None
        self._find_peaks_kwargs = None
        self.peak_frequency = None
        self.peak_amplitude = None
        self.update_peaks_bounded()

    def update_peaks_bounded(self, search_range_in_hz=(None, None), find_peaks_kwargs=None):
        """Update peak associated with HVSR curve, can be over bounded range.

        Parameters
        ----------
        search_range_in_hz : tuple, optional
            Frequency range to be searched for peaks.
            Half open ranges can be specified with ``None``, default is
            ``(None, None)`` indicating the full frequency range will be
            searched.
        find_peaks_kwargs : dict
            Keyword arguments for the ``scipy`` function
            `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            see ``scipy`` documentation for details.

        Returns
        -------
        None
            Updates internal peak-related attributes.

        """
        if (search_range_in_hz == self._search_range_in_hz) and (find_peaks_kwargs == self._find_peaks_kwargs):
            return
        else:
            self._search_range_in_hz = tuple(search_range_in_hz)
            self.meta["search_range_in_hz"] = self._search_range_in_hz
            if find_peaks_kwargs is None:
                self._find_peaks_kwargs = {}
                self.meta["find_peaks_kwargs"] = None
            else:
                self._find_peaks_kwargs = dict(find_peaks_kwargs)
                self.meta["find_peaks_kwargs"] = dict(find_peaks_kwargs)
                
        frq, amp = self._find_peak_bounded(self.frequency,
                                           self.amplitude,
                                           search_range_in_hz=search_range_in_hz,
                                           find_peaks_kwargs=find_peaks_kwargs)

        if frq is None:
            logger.info("No peak found in HVSR curve.")
            frq, amp = np.nan, np.nan

        self.peak_frequency, self.peak_amplitude = frq, amp

    def is_similar(self, other, atol=1E-9, rtol=0.):
        """Check if ``other`` is similar to ``self``."""
        if not isinstance(other, HvsrCurve):
            return False

        if len(self.frequency) != len(other.frequency):
            return False

        if not np.allclose(self.frequency, other.frequency, atol=atol, rtol=rtol):
            return False

        return True

    def __eq__(self, other: object) -> bool:
        if not self.is_similar(other):
            return False

        if not np.allclose(self.amplitude, other.amplitude):
            return False

        for attr in ["peak_frequency", "peak_amplitude"]:
            if not np.isclose(getattr(self, attr), getattr(self, attr)):
                return False

        return True
