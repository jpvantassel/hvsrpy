# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2022 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Class definition for HvsrTraditional object."""

import logging

import numpy as np
from scipy.signal import find_peaks

from .constants import DISTRIBUTION_MAP

logger = logging.getLogger(__name__)

__all__ = ["HvsrTraditional"]

class HvsrTraditional():
    """Class for creating and manipulating HvsrTraditional objects.

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

    def _search_range_to_index_range(self, search_range_in_hz):
        f_low, f_high = search_range_in_hz

        if f_low is None:
            f_low_idx = 0
        else:
            f_low_idx = np.argmin(self.frequency - f_low)

        if f_high is None:
            f_high_idx = len(self.frequency)
        else:
            f_high_idx = np.argmin(self.frequency - f_high)

        return (f_low_idx, f_high_idx)

    def update_peaks(self, search_range_in_hz=(None, None), **find_peaks_kwargs):
        """Update with the lowest frequency, highest amplitude peaks.

        Parameters
        ----------
        search_range_in_hz : tuple, optional
            Frequency range between which frequencies will be searched.
            Half open ranges can be specified with `None`. Default is
            `(None, None)` indicating the full frequency range will be
            searched.
        **find_peaks_kwargs : dict
            Keyword arguments for the `scipy` function
            `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            see `scipy` documentation for details.

        Returns
        -------
        None
            Updates internal peak-related attributes.

        """
        f_low_idx, f_high_idx = self._search_range_to_index_range(
            search_range_in_hz)

        peaks = []
        for window_idx, _amplitude in enumerate(self.amplitude):
            # find potential peaks, index is relative b/c curve passed in is relative.
            potential_peaks_relative, _ = find_peaks(_amplitude[f_low_idx:f_high_idx],
                                                     **find_peaks_kwargs)

            # if no peaks are found set window as invalid.
            if len(potential_peaks_relative) == 0:
                self.valid_window_boolean_mask[window_idx] = False
                logger.warning(f"No peak found in window {window_idx}.")
                continue

            # update peak_amplitude and peak_frequency information.
            potential_peaks_absolute = potential_peaks_relative + f_low_idx
            peak_idx_of_pot_peaks = np.argmax(_amplitude[potential_peaks_absolute])
            peak_idx = potential_peaks_absolute[peak_idx_of_pot_peaks]
            self._main_peak_amp[window_idx] = _amplitude[peak_idx]
            self._main_peak_frq[window_idx] = self.frequency[peak_idx]
            self.valid_window_boolean_mask[window_idx] = True

    @staticmethod
    def _mean_factory(distribution, values, **kwargs):
        distribution = DISTRIBUTION_MAP[distribution.lower()]
        if distribution == "normal":
            return np.mean(values, **kwargs)
        elif distribution == "lognormal":
            return np.exp(np.mean(np.log(values), **kwargs))
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

    def mean_f0_frq(self, distribution='lognormal'):
        """Mean `f0` of valid time windows.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Mean value of `f0` according to the distribution specified.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution, self.peak_frq)

    def mean_f0_amp(self, distribution='lognormal'):
        """Mean amplitude of `f0` of valid time windows.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Mean amplitude of `f0` according to the distribution
            specified.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution, self.peak_amp)

    @staticmethod
    def _std_factory(distribution, values, **kwargs):
        distribution = DISTRIBUTION_MAP[distribution.lower()]
        if distribution == "normal":
            return np.std(values, ddof=1, **kwargs)
        elif distribution == "lognormal":
            return np.std(np.log(values), ddof=1, **kwargs)
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

    def std_f0_frq(self, distribution='lognormal'):
        """Sample standard deviation of `f0` of valid time windows.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Sample standard deviation of `f0`.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution, self.peak_frequencies)

    def std_f0_amp(self, distribution='lognormal'):
        """Sample standard deviation of the amplitude of f0.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of `f0`
            considering only the valid time windows.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution, self.peak_amplitudes)

    def mean_curve(self, distribution='lognormal'):
        """Mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of mean curve, default is 'lognormal'.

        Returns
        -------
        ndarray
            Mean HVSR curve according to the distribution specified.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution,
                                  self.amplitude[self.valid_window_boolean_mask],
                                  axis=0)

    def std_curve(self, distribution='lognormal'):
        """Sample standard deviation of the mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of HVSR curve, default is 'lognormal'.

        Returns
        -------
        ndarray
            Sample standard deviation of HVSR curve according to the
            distribution specified.

        Raises
        ------
        ValueError
            If only single time window is defined.
        NotImplementedError
            If `distribution` does not match the available options.

        """
        if self.nseries > 1:
            return self._std_factory(distribution,
                                     self.amplitude[self.valid_window_boolean_mask],
                                     axis=0)
        else:
            msg = "The standard deviation of the mean curve is "
            msg += "not defined for a single window."
            raise ValueError(msg)

    def mean_curve_peak(self, distribution='lognormal',
                        search_range_in_hz=(None, None),
                        **find_peaks_kwargs):
        """Frequency of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        tuple
            Frequency and amplitude associated with the peak of the mean
            HVSR curve of the form
            `(mean_curve_peak_frequency, mean_curve_peak_amplitude)`.

        """
        f_low_idx, f_high_idx = self._search_range_to_index_range(search_range_in_hz)
        mc = self.mean_curve(distribution)
        relative_potential_peak_idxs, _ = find_peaks(mc[f_low_idx:f_high_idx],
                                                     **find_peaks_kwargs)
        if len(relative_potential_peak_idxs) == 0:
            msg = "Mean curve does not have a peak in the specified range."
            raise ValueError(msg)
        absolute_potential_peak_idxs = relative_potential_peak_idxs + f_low_idx
        peak_idx_of_potential_peaks = np.argmax(absolute_potential_peak_idxs)
        peak_idx = absolute_potential_peak_idxs[peak_idx_of_potential_peaks]
        return (self.frequency[peak_idx], mc[peak_idx])

    @staticmethod
    def _nth_std_factory(n, distribution, mean, std):
        distribution = DISTRIBUTION_MAP[distribution]
        if distribution == "normal":
            return (mean + n*std)
        elif distribution == "lognormal":
            return (np.exp(np.log(mean) + n*std))
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

    def nstd_f0_frq(self, n, distribution):
        """Value n standard deviations from mean `f0`.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean `f0` for
            the valid time windows.
        distribution : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0`, the default is 'lognormal'.

        Returns
        -------
        float
            Value n standard deviations from mean `f0`.

        """
        return self._nth_std_factory(n,
                                     distribution,
                                     self.mean_f0_frq(distribution),
                                     self.std_f0_frq(distribution))

    def nstd_f0_amp(self, n, distribution):
        """Value n standard deviations from mean `f0` amplitude.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean amplitude
            of `f0` from valid time windows.
        distribution : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0`, the default is 'lognormal'.

        Returns
        -------
        float
            Value n standard deviations from mean `f0` amplitude.

        """
        return self._nth_std_factory(n,
                                     distribution,
                                     self.mean_f0_amp(distribution),
                                     self.std_f0_amp(distribution))

    def nstd_curve(self, n, distribution="lognormal"):
        """nth standard deviation curve.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean curve.
        distribution : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, default is 'lognormal'.

        Returns
        -------
        ndarray
            nth standard deviation curve.

        """
        return self._nth_std_factory(n,
                                     distribution,
                                     self.mean_curve(distribution),
                                     self.std_curve(distribution))
