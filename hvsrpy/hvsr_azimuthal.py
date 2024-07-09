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
from .statistics import _nanmean_weighted, _nanstd_weighted, _flatten_list, _nth_std_factory

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

    def __init__(self, hvsrs, azimuths, meta=None):
        """``HvsrAzimuthal`` from iterable of ``HvsrTraditional`` objects.

        Parameters
        ----------
        hvsrs : iterable of HvsrTraditional
            Iterable of ``HvsrTraditional`` objects, one per azimuth.
        azimuths : float
            Rotation angles in degrees measured clockwise positive from
            north (i.e., 0 degrees), one per ``HvsrTraditional``.
        meta : dict, optional
            Meta information about the object, default is ``None``.

        Returns
        -------
        HvsrAzimuthal
            Instantiated ``HvsrAzimuthal`` object with single azimuth.

        """
        self.hvsrs = []
        self.azimuths = []
        ex_hvsr = hvsrs[0]
        for _idx, (hvsr, azimuth) in enumerate(zip(hvsrs, azimuths)):
            hvsr, azimuth = self._check_input(hvsr, azimuth)
            if not ex_hvsr.is_similar(hvsr):
                msg = "All HvsrTraditional must be similar; hvsrs[0] "
                msg += f"is not similar to hvsrs[{_idx}]"
                raise ValueError(msg)
            self.hvsrs.append(HvsrTraditional(hvsr.frequency, hvsr.amplitude,
                                              meta=hvsr.meta))
            self.azimuths.append(azimuth)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()
        self.update_peaks_bounded()

    @property
    def _search_range_in_hz(self):
        return self.hvsrs[0]._search_range_in_hz
    
    @property
    def _find_peaks_kwargs(self):
        return self.hvsrs[0]._find_peaks_kwargs

    def update_peaks_bounded(self, search_range_in_hz=(None, None), find_peaks_kwargs=None):
        """Update peaks associated with each HVSR curve, can be over bounded range.

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
        self.meta["search_range_in_hz"] = tuple(search_range_in_hz)
        self.meta["find_peaks_kwargs"] = None if find_peaks_kwargs is None else dict(find_peaks_kwargs) 
        for hvsr in self.hvsrs:
            hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz,
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

    def _compute_statistical_weights(self):
        """Compute weighting term following Cheng et al. (2020).

        .. warning::
            Private methods are subject to change without warning.

        """
        weights = []
        n_azimuths = len(self.azimuths)
        for hvsr in self.hvsrs:
            n_valid_peaks = int(np.sum(hvsr.valid_peak_boolean_mask))
            weights.extend([1/(n_azimuths*n_valid_peaks)]*n_valid_peaks)
        return np.array(weights)

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
        return _nanmean_weighted(distribution=distribution,
                                 weights=self._compute_statistical_weights(),
                                 values=np.array(_flatten_list(self.peak_frequencies)))

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
        return _nanmean_weighted(distribution=distribution,
                                 weights=self._compute_statistical_weights(),
                                 values=np.array(_flatten_list(self.peak_amplitudes)))

    def cov_fn(self, distribution="lognormal"):
        """Covariance of HVSR resonance across all valid HVSR curves and azimuths.

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
            resonance across all valid time windows and azimuths.

        Raises
        ------
        NotImplementedError
            If ``distribution`` does not match the available options.

        """
        distribution = DISTRIBUTION_MAP[distribution]

        frequencies = np.concatenate(self.peak_frequencies)
        amplitudes =  np.concatenate(self.peak_amplitudes)
        weights = self._compute_statistical_weights()

        if distribution == "normal":
            pass
        elif distribution == "lognormal":
            frequencies = np.log(frequencies)
            amplitudes = np.log(amplitudes)
        else: # pragma: no cover
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

        return np.cov(frequencies, amplitudes, aweights=weights)

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
        return _nanstd_weighted(distribution=distribution,
                                weights=self._compute_statistical_weights(),
                                values=np.array(_flatten_list(self.peak_frequencies)),
                                denominator="cheng")

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
        return _nanstd_weighted(distribution=distribution,
                                weights=self._compute_statistical_weights(),
                                values=np.array(_flatten_list(self.peak_amplitudes)),
                                denominator="cheng")

    @property
    def amplitude(self):
        return [hvsr.amplitude for hvsr in self.hvsrs]

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

    def mean_curve_peak_by_azimuth(self, distribution="lognormal"):
        """Peak from each mean curve, one per azimuth.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of mean curve, default is ``"lognormal"``.

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
            f_peak, a_peak = hvsr.mean_curve_peak(distribution=distribution)
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
        weights = self._compute_statistical_weights()
        mean_curve = np.empty_like(self.frequency)
        for _idx in range(len(mean_curve)):
            amplitude = [hvsr.amplitude[hvsr.valid_window_boolean_mask][:, _idx].tolist()
                         for hvsr in self.hvsrs]
            amplitude = _flatten_list(amplitude)
            mean_curve[_idx] = _nanmean_weighted(distribution=distribution,
                                                 values=np.array(amplitude),
                                                 weights=weights,
                                                 mean_kwargs=dict(axis=0))

        return mean_curve

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
        weights = self._compute_statistical_weights()
        std_curve = np.empty_like(self.frequency)
        for _idx in range(len(std_curve)):
            amplitude = [hvsr.amplitude[hvsr.valid_window_boolean_mask][:, _idx].tolist()
                         for hvsr in self.hvsrs]
            amplitude = _flatten_list(amplitude)
            std_curve[_idx] = _nanstd_weighted(distribution=distribution,
                                               values=np.array(amplitude),
                                               weights=weights,
                                               std_kwargs=dict(axis=0),
                                               denominator="cheng")

        return std_curve

    def nth_std_curve(self, n, distribution="lognormal"):
        """nth standard deviation on mean curve considering all valid
        windows across all azimuths."""
        return _nth_std_factory(n=n,
                               distribution=distribution,
                               mean=self.mean_curve(distribution=distribution),
                               std=self.std_curve(distribution=distribution))

    def nth_std_fn_frequency(self, n, distribution="lognormal"):
        """nth standard deviation on frequency of ``fn`` considering all
        valid windows across all azimuths."""
        return _nth_std_factory(n=n,
                               distribution=distribution,
                               mean=self.mean_fn_frequency(distribution=distribution),
                               std=self.std_fn_frequency(distribution=distribution))

    def nth_std_fn_amplitude(self, n, distribution="lognormal"):
        """nth standard deviation on amplitude of ``fn`` considering all
        valid windows across all azimuths."""
        return _nth_std_factory(n=n,
                                distribution=distribution,
                                mean=self.mean_fn_amplitude(distribution=distribution),
                                std=self.std_fn_amplitude(distribution=distribution))

    def mean_curve_peak(self, distribution="lognormal"):
        """Frequency and amplitude of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {"normal", "lognormal"}, optional
            Assumed distribution of HVSR curve, default is ``"lognormal"``.

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
                                                      search_range_in_hz=self._search_range_in_hz,
                                                      find_peaks_kwargs=self._find_peaks_kwargs)

        if f_peak is None or a_peak is None: # pragma: no cover
            msg = "Mean curve does not have a peak in the specified range."
            raise ValueError(msg)

        return (f_peak, a_peak)

    def is_similar(self, other):
        """Determine if ``other`` is similar to ``self``."""
        if not isinstance(other, HvsrAzimuthal):
            return False

        if len(self.hvsrs) != len(other.hvsrs):
            return False

        # note: do not need to check all because all must be similar.
        if not self.hvsrs[0].is_similar(other.hvsrs[0]):
            return False

        for self_azi, other_azi in zip(self.azimuths, other.azimuths):
            if abs(self_azi - other_azi) > 0.1:
                return False

        return True

    def __eq__(self, other):
        """Determine if ``other`` is equal to ``self``."""
        if not self.is_similar(other):
            return False

        for self_hvsr, other_hvsr in zip(self.hvsrs, other.hvsrs):
            if self_hvsr != other_hvsr:
                return False

        return True

    def __str__(self):
        """Human-readable representation of ``HvsrAzimuthal`` object."""
        return f"HvsrAzimuthal at {id(self)}"

    def __repr__(self):
        """Unambiguous representation of ``HvsrAzimuthal`` object."""
        return f"HvsrAzimuthal(hvsrs={self.hvsrs}, azimuths={self.azimuths}, meta={self.meta})"
