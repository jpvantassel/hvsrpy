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

"""Class definition for HvsrAzimuthal, HVSR calculated across various azimuths."""

import logging

import numpy as np

from .hvsr_curve import HvsrCurve
from .hvsr_traditional import HvsrTraditional
from .metadata import __version__
from .constants import DISTRIBUTION_MAP
from .statistics import mean_factory, flatten_list, nth_std_factory

logger = logging.getLogger(__name__)

__all__ = ["HvsrAzimuthal"]


class HvsrAzimuthal():
    """For HVSR calculations made across various azimuths.

    Attributes
    ----------
    hvsrs : list
        Container of ``HvsrTraditional`` objects, one per azimuth.
    azimuths : list
        Vector of rotation azimuths corresponding to
        ``HvsrTraditional`` objects.

    """
    @staticmethod
    def _check_input(hvsr, azimuth):
        """Check input,

         Specifically:
            1. ``hvsr`` is an instance of ``HvsrTraditional``.
            2. ``azimuth`` is ``float``.
            3. ``azimuth`` is greater than 0 and less than 180.

        .. warning::
            Private methods are subject to change without warning.

        """
        if not isinstance(hvsr, HvsrTraditional):
            msg = "each hvsr must be an instance of HvsrTraditional; "
            msg += f"not {type(hvsr)}."
            raise TypeError(msg)

        azimuth = float(azimuth)

        if (azimuth < 0) or (azimuth > 180):
            msg = f"azimuth is {azimuth}; azimuth must be between 0 and 180."
            raise ValueError(msg)

        return (hvsr, azimuth)

    def __init__(self, hvsrs, azimuths, search_range_in_hz=(None, None),
                 find_peaks_kwargs=None, meta=None):
        """``HvsrAzimuthal`` from iterable of ``HvsrTraditional`` objects.

        Parameters
        ----------
        hvsrs : iterable of HvsrTraditional
            Iterable of ``HvsrTraditional`` objects, one per azimuth.
        azimuths : float
            Rotation angles in degrees measured clockwise positive from
            north (i.e., 0 degrees), one per ``HvsrTraditional``.
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
        HvsrAzimuthal
            Instantiated ``HvsrAzimuthal`` object with single azimuth.

        """
        self.hvsrs = []
        self.azimuths = []
        for hvsr, azimuth in zip(hvsrs, azimuths):
            hvsr, azimuth = self._check_input(hvsr, azimuth)
            self.hvsrs.append(HvsrTraditional(hvsr.frequency, hvsr.amplitude,
                                              search_range_in_hz=search_range_in_hz,
                                              find_peaks_kwargs=find_peaks_kwargs,
                                              meta=hvsr.meta))
            self.azimuths.append(azimuth)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()

    def _update_peaks_bounded(self, search_range_in_hz=(None, None), find_peaks_kwargs=None):
        """Update peaks associated with each HVSR curve, can be over bounded range.

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
        for hvsr in self.hvsrs:
            hvsr._update_peaks_bounded(search_range_in_hz=search_range_in_hz,
                                       find_peaks_kwargs=find_peaks_kwargs)

    @property
    def peak_frequencies(self):
        """Peak frequencies, one entry per azimuth, each entry has one value per curve."""
        return [hvsr.peak_frequencies for hvsr in self.hvsrs]

    @property
    def peak_amplitudes(self):
        """Peak amplitudes, one entry per azimuth, each entry has one value per curve."""
        return [hvsr.peak_amplitudes for hvsr in self.hvsrs]

    @property
    def n_azimuths(self):
        return len(self.azimuths)

    def mean_fn_frequency(self, distribution="lognormal"):
        """Mean frequency of ``fn`` across all valid HVSR curves and azimuths.

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
        return mean_factory(distribution=distribution,
                            values=flatten_list(self.peak_frequencies))

    def mean_fn_amplitude(self, distribution="lognormal"):
        """Mean amplitude of ``fn`` across all valid HVSR curves and azimuths.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Mean amplitude of ``fn`` across all valid time windows and
            azimuths.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return mean_factory(distribution=distribution,
                            values=flatten_list(self.peak_amplitudes))

    @staticmethod
    def _std_factory(distribution, values, sum_kwargs=None):
        """Calculates azimuthal standard deviation consistent with
        distribution using the approach of Cheng et al. (2020).

        .. warning::
            Private methods are subject to change without warning.

        """
        distribution = DISTRIBUTION_MAP.get(distribution, None)
        mean = mean_factory(distribution=distribution,
                            values=flatten_list(values))

        if distribution == "normal":
            def _diff(value, mean):
                return value - mean
        elif distribution == "lognormal":
            def _diff(value, mean):
                return np.log(value) - np.log(mean)
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

        if sum_kwargs is None:
            sum_kwargs = {}

        n = len(values)
        num = 0
        wi2 = 0
        for value in values:
            i = len(value)
            diff = _diff(value, mean)
            wi = 1/(n*i)
            num += np.sum(diff*diff*wi, **sum_kwargs)
            wi2 += wi*wi*i

        return np.sqrt(num/(1-wi2))

    def std_fn_frequency(self, distribution="lognormal"):
        """Sample standard deviation frequency of ``fn`` across all valid HVSR curves and azimuths.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Sample standard deviation of ``fn``.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_frequencies)

    def std_fn_amplitude(self, distribution="lognormal"):
        """Sample standard deviation amplitude of ``fn`` across all valid HVSR curves and azimuths.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of ``fn``, default is ``"lognormal"``.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of ``fn`` according
            to the distribution specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_amplitudes)

    @property
    def amplitude(self):
        return [hvsr.amplitude[hvsr.valid_window_boolean_mask] for hvsr in self.hvsrs]

    @property
    def frequency(self):
        return self.hvsrs[0].frequency

    def mean_curve_by_azimuth(self, distribution="lognormal"):
        """Mean curve associated with each azimuth.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of mean curve, default is ``"lognormal"``.

        Returns
        -------
        ndarray
            Each row corresponds to the mean curve from an azimuth and
            each column a frequency.

        """
        array = np.empty((self.n_azimuths, len(self.frequency)))
        for _idx, hvsr in enumerate(self.hvsrs):
            array[_idx, :] = hvsr.mean_curve(distribution=distribution)
        return array

    def mean_curve_peak_by_azimuth(self, distribution="lognormal",
                                   search_range_in_hz=(None, None),
                                   **find_peaks_kwargs):
        """Peak from each mean curve, one per azimuth.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of mean curve, default is ``"lognormal"``.
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
        tuple
            Of the form ``(peak_frequencies, peak_amplitudes)`` where
            each entry contains the peak of the mean curve, one per
            azimuth.

        """
        peak_frequencies = np.empty(self.n_azimuths)
        peak_amplitudes = np.empty(self.n_azimuths)
        for _idx, hvsr in enumerate(self.hvsrs):
            f_peak, a_peak = hvsr.mean_curve_peak(distribution=distribution,
                                                  search_range_in_hz=search_range_in_hz,
                                                  find_peak_kwargs=find_peaks_kwargs)
            peak_frequencies[_idx] = f_peak
            peak_amplitudes[_idx] = a_peak
        return (peak_frequencies, peak_amplitudes)

    def mean_curve(self, distribution="lognormal"):
        """Mean HVSR curve considering all valid HVSR curves across all azimuths.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of mean curve, default is ``"lognormal"``.

        Returns
        -------
        ndarray
            Mean HVSR curve considering all valid HVSR curves across all
            azimuths according to the distribution specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return mean_factory(distribution=distribution,
                            values=self.amplitude,
                            mean_kwargs=dict(axis=0))

    def std_curve(self, distribution="lognormal"):
        """Sample standard deviation associated with mean HVSR curve
        considering all valid HVSR curves across all azimuths.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of HVSR curve, default is
            ``"lognormal"``.

        Returns
        -------
        ndarray
            Sample standard deviation of HVSR curve considering all
            valid HVSR curves across all azimuths according to the
            distribution specified.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.amplitude,
                                 sum_kwargs=dict(axis=0))

    def nth_std_curve(self, n, distribution="lognormal"):
        """nth standard deviation on mean curve considering all valid
        windows across all azimuths."""
        return nth_std_factory(n=n,
                               distribution=distribution,
                               mean=self.mean_curve(distribution=distribution),
                               std=self.std_curve(distribution=distribution))

    def nth_std_fn_frequency(self, n, distribution="lognormal"):
        """nth standard deviation on frequency of ``fn`` considering all
        valid windows across all azimuths."""
        return nth_std_factory(n=n,
                               distribution=distribution,
                               mean=self.mean_fn_frequency(distribution=distribution),
                               std=self.std_fn_frequency(distribution=distribution))

    def nth_std_fn_amplitude(self, n, distribution="lognormal"):
        """nth standard deviation on amplitude of ``fn`` considering all
        valid windows across all azimuths."""
        return nth_std_factory(n=n,
                               distribution=distribution,
                               mean=self.mean_fn_amplitude(distribution=distribution),
                               std=self.std_fn_amplitude(distribution=distribution))

    def mean_curve_peak(self,
                        distribution="lognormal",
                        search_range_in_hz=(None, None),
                        **find_peaks_kwargs):
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
            see ``scipy`` documentation for details.    

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
