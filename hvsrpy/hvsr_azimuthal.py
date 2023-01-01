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

"""Class definition for HvsrRotated, a rotated Hvsr measurement."""

import logging

import numpy as np
from scipy.signal import find_peaks

from .hvsr_traditional import HvsrTraditional
from .metadata import __version__
from .constants import DISTRIBUTION_MAP

logger = logging.getLogger(__name__)

__all__ = ["HvsrAzimuthal"]


class HvsrAzimuthal():
    """Class definition for HVSR calculations across various azimuths.

    Attributes
    ----------
    hvsrs : list
        Container of `HvsrTraditional` objects, one per azimuth.
    azimuths : ndarray
        Vector of rotation azimuths correpsonding to `Hvsr` objects.

    """

    def __init__(self, hvsr, azimuth, meta=None):
        """Instantiate a `HvsrAzimuthal` object.

        Parameters
        ----------
        hvsr : Hvsr
            `Hvsr` object.
        azimuth : float
            Rotation angle in degrees measured clockwise positive from
            north (i.e., 0 degrees).
        meta : dict, optional
            Meta information about the object, default is `None`.

        Returns
        -------
        HvsrAzimuthal
            Instantiated `HvsrAzimuthal` object.

        """
        hvsr, azimuth = self._check_input(hvsr, azimuth)
        self.hvsrs = [hvsr]
        self.azimuths = [azimuth]
        self.meta = meta

    @staticmethod
    def _check_input(hvsr, az):
        """Check input, specifically:
            1. `hvsr` is an instance of `Hvsr`.
            2. `az` is `float`.
            3. `az` is greater than 0 and less than 180.

        """
        if not isinstance(hvsr, HvsrTraditional):
            raise TypeError("`hvsr` must be an instance of `HvsrTraditional`.")

        az = float(az)

        if (az < 0) or (az > 180):
            msg = f"`azimuth` is {az}; `azimuth` must be between 0 and 180."
            raise ValueError(msg)

        return (hvsr, az)

    def append(self, hvsr, azimuth):
        """Append `HvsrTraditional` object at a new azimuth.

        Parameters
        ----------
        hvsr : Hvsr
            `Hvsr` object.
        az : float
            Rotation angle in degrees measured clockwise from north
            (i.e., 0 degrees).

        Returns
        -------
        HvsrRotated
            Instantiated `HvsrRotated` object.

        """
        hvsr, azimuth = self._check_input(hvsr, azimuth)
        self.hvsrs.append(hvsr)
        self.azimuths.append(azimuth)

    @classmethod
    def from_iter(cls, hvsrs, azimuths, meta=None):
        """Create `HvsrAzimuthal` from iterable of `HvsrTraditional` objects."""
        obj = cls(hvsrs[0], azimuths[0], meta=meta)
        if len(azimuths) > 1:
            for hvsr, az in zip(hvsrs[1:], azimuths[1:]):
                obj.append(hvsr, az)
        return obj

    @property
    def peak_frequency(self):
        """Array of peak frequencies, one array per azimuth."""
        return [hvsr.peak_frequency for hvsr in self.hvsrs]

    @property
    def peak_amplitude(self):
        """Array of peak amplitudes, one array per azimuth."""
        return [hvsr.peak_amplitude for hvsr in self.hvsrs]

    @property
    def azimuth_count(self):
        return len(self.azimuths)

    def mean_f0_frq(self, distribution="lognormal"):
        """Mean `f0` from all valid timewindows and azimuths.

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
        return self._mean_factory(distribution=distribution,
                                  values=self.peak_frequency)

    def mean_f0_amp(self, distribution="lognormal"):
        """Mean `f0` amplitude from all valid timewindows and azimuths.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Mean amplitude of `f0` across all valid time windows and
            azimuths.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution=distribution,
                                  values=self.peak_amp)

    @staticmethod
    def _mean_factory(distribution, values, **kwargs):
        distribution = DISTRIBUTION_MAP[distribution]
        if distribution == "normal":
            mean = np.mean([np.mean(x, **kwargs) for x in values], **kwargs)
            return mean
        elif distribution == "lognormal":
            mean = np.mean([np.mean(np.log(x), **kwargs)
                            for x in values], **kwargs)
            return np.exp(mean)
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

    @staticmethod
    def _std_factory(distribution, values, **kwargs):
        distribution = DISTRIBUTION_MAP[distribution]
        n = len(values)
        mean = HvsrAzimuthal._mean_factory(distribution, values, **kwargs)
        num = 0
        wi2 = 0

        if distribution == "normal":
            def _diff(value, mean):
                return value - mean
        elif distribution == "lognormal":
            def _diff(value, mean):
                return np.log(value) - np.log(mean)

        for value in values:
            i = len(value)
            diff = _diff(value, mean)
            wi = 1/(n*i)
            num += np.sum(diff*diff*wi, **kwargs)
            wi2 += wi*wi*i

        return np.sqrt(num/(1-wi2))

    def std_f0_frq(self, distribution='lognormal'):
        """Sample standard deviation of `f0` for all valid time windows
        across all azimuths.

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
        return self._std_factory(distribution=distribution,
                                 values=self.peak_frq)

    def std_f0_amp(self, distribution='lognormal'):
        """Sample standard deviation of `f0` amplitude for all valid
        time windows across all azimuths.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of `f0` according
            to the distribution specified.

        Raises
        ------
        NotImplementedError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_amp)

    @property
    def amplitude(self):
        return [hvsr.amplitude[hvsr.valid_window_boolean_mask] for hvsr in self.hvsrs]

    @property
    def frequency(self):
        return self.hvsrs[0].frequency

    def mean_curves(self, distribution='lognormal'):
        """Mean curve for each azimuth

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of mean curve, default is 'lognormal'.

        Returns
        -------
        ndarray
            Each row corresponds to an azimuth and each column a
            frequency.

        """
        array = np.empty((self.azimuth_count, len(self.frq)))
        for az_cnt, hvsr in enumerate(self.hvsrs):
            array[az_cnt, :] = hvsr.mean_curve(distribution=distribution)
        return array

    def mean_curves_peak(self, distribution="lognormal"):
        """Peak from each mean curve, one per azimuth."""
        frqs, amps = np.empty(self.azimuth_count), np.empty(self.azimuth_count)
        for az_cnt, hvsr in enumerate(self.hvsrs):
            frqs[az_cnt], amps[az_cnt] = hvsr.mean_curve_peak(distribution=distribution)
        return (frqs, amps)

    def mean_curve(self, distribution='lognormal'):
        """Mean H/V curve considering all valid windows and azimuths.

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
        return self._mean_factory(distribution=distribution,
                                  values=self.amplitude,
                                  axis=0)

    def std_curve(self, distribution='lognormal'):
        """Sample standard deviation associated with mean HVSR curve.

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
        return self._std_factory(distribution=distribution,
                                 values=self.amp,
                                 axis=0)

    @staticmethod
    def _nth_std_factory(n, distribution, mean, std):
        return HvsrTraditional._nth_std_factory(n=n,
                                                distribution=distribution,
                                                mean=mean,
                                                std=std)

    def nstd_curve(self, n, distribution):
        """nth standard deviation on mean curve from all azimuths."""
        return self._nth_std_factory(n=n,
                                     distribution=distribution,
                                     mean=self.mean_curve(distribution=distribution),
                                     std=self.std_curve(distribution=distribution))

    def nstd_f0_frq(self, n, distribution):
        """nth standard deviation on `f0` from all azimuths"""
        return self._nth_std_factory(n=n,
                                     distribution=distribution,
                                     mean=self.mean_f0_frq(distribution=distribution),
                                     std=self.std_f0_frq(distribution=distribution))

    def mean_curve_peak(self,
                        distribution='lognormal',
                        search_range_in_hz=(None, None),
                        **find_peaks_kwargs):
        """Frequency of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of HVSR curve, default is 'lognormal'.

        Returns
        -------
        tuple
            Frequency and amplitude associated with the peak of the mean
            HVSR curve of the form
            `(mean_curve_peak_frequency, mean_curve_peak_amplitude)`.

        """
        #TODO(jpv): Finish documentation.
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
