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

"""Class definition for HvsrTraditional object."""

import logging

import numpy as np

from .statistics import mean_factory, std_factory, nth_std_factory, DISTRIBUTION_MAP
from .hvsr_curve import HvsrCurve

logger = logging.getLogger(__name__)

__all__ = ["HvsrTraditional"]


class HvsrTraditional():
    """Class for creating and manipulating ``HvsrTraditional`` objects.

    Attributes
    ----------
    amplitude : ndarray
        Array of HVSR amplitudes. Each row represents an individual
        curve (e.g., from a time window or earthquake recording) and
        each column a frequency.
    frequency : ndarray
        Vector of frequencies, one per amplitude column.
    n_curves : int
        Number of HVSR curves in ``HvsrTraditional`` object; one HVSR
        curve per time window or earthquake recording.
    valid_curve_boolean_mask : ndarray
        Boolean array indicating whether each HVSR curve is valid
        (``True``) or invalid (``False``).

    """

    def __init__(self, frequency, amplitude, search_range_in_hz=(None, None),
                 find_peaks_kwargs=None, meta=None):
        """Create ``HvsrTraditional`` from amplitude and frequency.

        Parameters
        ----------
        amplitude : ndarray
            Array of HVSR amplitudes. Each row represents an individual
            curve (e.g., from a time window or earthquake record) and
            each column a frequency.
        frequency : ndarray
            Vector of frequencies, corresponding to each column of
            ``amplitude``.
        search_range_in_hz : tuple, optional
            Frequency range to be searched for peaks.
            Half open ranges can be specified with ``None``, default is
            ``(None, None)`` indicating the full frequency range will be
            searched.
        find_peaks_kwargs : dict
            Keyword arguments for the ``scipy`` function
            `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            see ``scipy`` documentation for details.    
        meta : dict, optional
            Meta information about the object, default is ``None``.

        Returns
        -------
        HvsrTraditional
            Initialized with ``amplitude`` and ``frequency``.

        """
        self.frequency = HvsrCurve._check_input(frequency, "frequency")
        self.amplitude = np.atleast_2d(HvsrCurve._check_input(amplitude, "amplitude"))

        if len(self.frequency) != self.amplitude.shape[1]:
            msg = f"Shape of amplitude={self.amplitude.shape} and "
            msg += f"frequency={self.frequency.shape} must be compatible."
            raise ValueError(msg)

        self.n_curves = len(self.amplitude)
        self.valid_window_boolean_mask = np.ones((self.n_curves,), dtype=bool)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()

        self._main_peak_frq = np.empty(self.n_curves)
        self._main_peak_amp = np.empty(self.n_curves)
        self._update_peaks_bounded(search_range_in_hz=search_range_in_hz,
                                   find_peaks_kwargs=find_peaks_kwargs)

    @ classmethod
    def from_hvsr_curves(cls, hvsr_curves, meta=None):
        """Instantiate `HvsrTraditional` from iterable of ``HvsrCurve``.

        Parameters
        ----------
        amplitude : iterable of HvsrCurve
            Iterable of HvsrCurve objects one curve for each time window
            or earthquake record and common frequency sampling.
        meta : dict, optional
            Meta information about the object, default is ``None``.

        Returns
        -------
        HvsrTraditional
            Instantiated from ``HvsrCurve`` data.

        """
        example = hvsr_curves[0]
        amplitude = np.empty((len(hvsr_curves), len(example.frequency)))
        for idx, hvsr_curve in enumerate(hvsr_curves):
            if hvsr_curve.is_similar(example):
                amplitude[idx] = hvsr_curve.amplitude
            else:
                msg = f"All HvsrCurve objects must be similar, index {idx} "
                msg += "is not similar to index 0."
                raise ValueError(msg)

        return cls(example.frequency, amplitude, meta=meta)

    @ property
    def rejected_window_boolean_mask(self):
        """Boolean array indicating rejected (i.e., invalid) windows."""
        return np.invert(self.valid_window_boolean_mask)

    @ property
    def peak_frequencies(self):
        """Valid peak frequency vector, one per window or earthquake recording."""
        return self._main_peak_frq[self.valid_window_boolean_mask]

    @ property
    def peak_amplitudes(self):
        """Valid peak amplitude vector, one per window or earthquake recording."""
        return self._main_peak_amp[self.valid_window_boolean_mask]

    def _update_peaks_bounded(self, search_range_in_hz=(None, None), find_peaks_kwargs=None):
        """Update peak associated with each HVSR curve, can be over bounded range.

        .. warning::
            Private methods are subject to change without warning.

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
        if find_peaks_kwargs is None:
            find_peaks_kwargs = {}

        for _idx, _amplitude in enumerate(self.amplitude):
            (f_peak, a_peak) = HvsrCurve._find_peak_bounded(self.frequency,
                                                            _amplitude,
                                                            search_range_in_hz=search_range_in_hz,
                                                            find_peak_kwargs=find_peaks_kwargs)

            # If no peaks are found set window as invalid.
            if f_peak is None:
                self.valid_window_boolean_mask[_idx] = False
                logger.warning(f"No peak found in window {_idx}.")
                continue

            # If peak found, update peak_frequencies and peak_amplitudes.
            self._main_peak_frq[_idx] = f_peak
            self._main_peak_amp[_idx] = a_peak
            self.valid_window_boolean_mask[_idx] = True

    def mean_fn_frequency(self, distribution="lognormal"):
        """Mean frequency of peaks associated with ``fn`` from valid HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Mean value of ``fn`` according to the distribution specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return mean_factory(distribution, self.peak_frequencies)

    def mean_fn_amplitude(self, distribution="lognormal"):
        """Mean amplitude of peaks associated with ``fn`` from valid HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Mean amplitude of ``fn`` according to the distribution
            specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return mean_factory(distribution, self.peak_amplitudes)

    def cov_fn(self, distribution="lognormal"):
        """Covariance of HVSR resonance across all valid HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of resonance, default is
            ``"lognormal"``.

        Returns
        -------
        ndarray
            Tensor of shape ``(2,2)`` that represents the
            covariance matrix of frequency and amplitude of HVSR
            resonance across all valid time windows.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        distribution = DISTRIBUTION_MAP[distribution]

        frequencies = []
        amplitudes = []
        weights = []
        for hvsr in self.hvsrs:
            n_valid = np.sum(hvsr.valid_window_boolean_mask)
            frequencies.extend(hvsr.peak_frequencies)
            amplitudes.extend(hvsr.peak_frequencies)
            weights.extend([1/n_valid]*n_valid)

        if distribution == "normal":
            pass
        elif distribution == "lognorma":
            frequencies = np.log(frequencies)
            amplitudes = np.log(amplitudes)
        else:
            raise NotImplementedError

        return np.cov(frequencies, amplitudes, aweights=weights)

    def std_fn_frequency(self, distribution="lognormal"):
        """Sample standard deviation of frequency of peaks associated with ``fn`` from valid HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Sample standard deviation of the frequency of ``fn``
            considering only the valid HVSR curves.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return std_factory(distribution, self.peak_frequencies)

    def std_fn_amplitude(self, distribution="lognormal"):
        """Sample standard deviation of amplitude of peaks associated with ``fn`` from valid HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of ``fn``
            considering only the valid HVSR curves.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return std_factory(distribution, self.peak_amplitudes)

    def mean_curve(self, distribution="lognormal"):
        """Mean HVSR curve.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of mean curve, default is "lognormal".

        Returns
        -------
        ndarray
            Mean HVSR curve according to the distribution specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return mean_factory(distribution,
                            self.amplitude[self.valid_window_boolean_mask],
                            mean_kwargs=dict(axis=0))

    def std_curve(self, distribution="lognormal"):
        """Sample standard deviation of the HVSR curves.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of HVSR curve, default is ``"lognormal"``.

        Returns
        -------
        ndarray
            Sample standard deviation of HVSR curve according to the
            distribution specified.

        Raises
        ------
        ValueError
            If only single HVSR curve is defined.
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        if self.ncurves > 1:
            return std_factory(distribution,
                               self.amplitude[self.valid_window_boolean_mask],
                               std_kwargs=dict(axis=0))
        else:
            msg = "The standard deviation of the mean curve is "
            msg += "not defined for a single window."
            raise ValueError(msg)

    # TODO(jpv): Replace **kwargs here (and elsewhere).
    def mean_curve_peak(self, distribution="lognormal",
                        search_range_in_hz=(None, None),
                        find_peaks_kwargs=None):
        """Frequency and amplitude of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of HVSR curve, default is ``"lognormal"``.
        search_range_in_hz : tuple, optional
            Frequency range to be searched for peaks.
            Half open ranges can be specified with ``None``, default is
            ``(None, None)`` indicating the full frequency range will be
            searched.
        find_peaks_kwargs : dict
            Keyword arguments for the ``scipy`` function
            `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            see ``scipy`` documentation for details, default is ``None``
            indicating defaults will be used.   

        Returns
        -------
        tuple
            Frequency and amplitude associated with the peak of the mean
            HVSR curve of the form
            ``(mean_curve_peak_frequency, mean_curve_peak_amplitude)``.

        """
        amplitude = self.mean_curve(distribution)
        f_peak, a_peak = HvsrCurve._find_peak_bounded(self.frequency,
                                                      amplitude,
                                                      search_range_in_hz=search_range_in_hz,
                                                      find_peak_kwargs=find_peaks_kwargs)

        if f_peak is None or a_peak is None:
            msg = "Mean curve does not have a peak in the specified range."
            raise ValueError(msg)

        return (f_peak, a_peak)

    def nth_std_fn_frequency(self, n, distribution="lognormal"):
        """Value n standard deviations from mean ``fn`` frequency.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean frequency
            of ``fn`` computed from valid HVSR curves.
        distribution : {"lognormal", "normal"}, optional
            Assumed distribution of ``fn``, the default is ``"lognormal"``.

        Returns
        -------
        float
            Value n standard deviations from mean ``fn`` frequency.

        """
        return nth_std_factory(n,
                               distribution,
                               self.mean_f0_frq(distribution),
                               self.std_f0_frq(distribution))

    def nth_std_fn_amplitude(self, n, distribution="lognormal"):
        """Value n standard deviations from mean ``fn`` amplitude.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean amplitude
            of ``fn`` computed from valid HVSR curves.
        distribution : {"lognormal", "normal"}, optional
            Assumed distribution of ``fn``, the default is ``"lognormal"``.

        Returns
        -------
        float
            Value n standard deviations from mean ``fn`` amplitude.

        """
        return nth_std_factory(n,
                               distribution,
                               self.mean_f0_amp(distribution),
                               self.std_f0_amp(distribution))

    def nth_std_curve(self, n, distribution="lognormal"):
        """nth standard deviation curve.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean curve.
        distribution : {"lognormal", "normal"}, optional
            Assumed distribution of mean curve, default is ``"lognormal"``.

        Returns
        -------
        ndarray
            nth standard deviation curve.

        """
        return nth_std_factory(n,
                               distribution,
                               self.mean_curve(distribution),
                               self.std_curve(distribution))
