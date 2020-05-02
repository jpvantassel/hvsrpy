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

"""Class definition for HvsrRotated, a rotated Hvsr measurement."""

import logging

import numpy as np

from hvsrpy import Hvsr

logger = logging.getLogger(__name__)

__all__ = ["HvsrRotated"]


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

    @staticmethod
    def _check_input(hvsr, az):
        """Check input, specifically:
            1. `hvsr` is an instance of `Hvsr`.
            2. Cast `az` to float (if it is not already).
            3. `az` is greater than 0.

        """
        if not isinstance(hvsr, Hvsr):
            raise TypeError("`hvsr` must be an instance of `Hvsr`.")

        az = float(az)

        if (az < 0) or (az > 180):
            raise ValueError(f"`azimuth` must be between 0 and 180, not {az}.")

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
        if len(azimuths) > 1:
            for hvsr, az in zip(hvsrs[1:], azimuths[1:]):
                obj.append(hvsr, az)
        return obj

    @property
    def peak_frq(self):
        """Array of peak frequencies, one array per azimuth."""
        return [hv.peak_frq for hv in self.hvsrs]

    @property
    def peak_amp(self):
        """Array of peak amplitudes, one array per azimuth."""
        return [hv.peak_amp for hv in self.hvsrs]

    # TODO (jpv): What if all windows get rejected on an azimuth?
    def reject_windows(self, **kwargs):
        for hv in self.hvsrs:
            hv.reject_windows(**kwargs)

    @property
    def azimuth_count(self):
        return len(self.azimuths)

    def mean_f0_frq(self, distribution="log-normal"):
        """Mean `f0` from all valid timewindows and azimuths.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}
            Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
        float
            Mean value of `f0` according to the distribution specified.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution=distribution,
                                  values=self.peak_frq)

    def mean_f0_amp(self, distribution="log-normal"):
        """Mean `f0` amplitude from all valid timewindows and azimuths.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}
            Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
        float
            Mean amplitude of `f0` across all valid time windows and
            azimuths.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution=distribution,
                                  values=self.peak_amp)

    @staticmethod
    def _mean_factory(distribution, values, **kwargs):
        if distribution == "normal":
            mean = np.mean([np.mean(x, **kwargs) for x in values], **kwargs)
            return mean
        elif distribution == "log-normal":
            mean = np.mean([np.mean(np.log(x), **kwargs)
                            for x in values], **kwargs)
            return np.exp(mean)
        else:
            msg = f"distribution type {distribution} not recognized."
            raise NotImplementedError(msg)

    @staticmethod
    def _std_factory(distribution, values, **kwargs):
        n = len(values)
        mean = HvsrRotated._mean_factory(distribution, values, **kwargs)
        num = 0
        wi2 = 0

        if distribution == "normal":
            def _diff(value, mean):
                return value - mean
        elif distribution == "log-normal":
            def _diff(value, mean):
                return np.log(value) - np.log(mean)

        for value in values:
            i = len(value)
            diff = _diff(value, mean)
            wi = 1/(n*i)
            num += np.sum(diff*diff*wi, **kwargs)
            wi2 += wi*wi*i
            
        return np.sqrt(num/(1-wi2))

    def std_f0_frq(self, distribution='log-normal'):
        """Sample standard deviation of `f0` of valid time windows.

        Parameters
        ----------
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
            Sample standard deviation of `f0` according to the
            distribution specified.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_frq)

    def std_f0_amp(self, distribution='log-normal'):
        """Sample standard deviation of `f0` amplitude of valid windows.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of f0 according
            to the distribution specified.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_amp)

    @property
    def amp(self):
        return [hv.amp[hv.valid_window_indices] for hv in self.hvsrs]

    @property
    def frq(self):
        return self.hvsrs[0].frq

    def mean_curves(self, distribution='log-normal'):
        """Mean curve for each azimuth row=azimuth, col=frq."""
        array = np.empty((self.azimuth_count, len(self.frq)))
        for az_cnt, hvsr in enumerate(self.hvsrs):
            array[az_cnt, :] = hvsr.mean_curve(distribution=distribution)
        return array

    def mean_curve(self, distribution='log-normal'):
        """Return mean H/V curve.

        Parameters
        ----------
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of mean curve, default is
                'log-normal'.

        Returns
        -------
        ndarray
            Mean H/V curve according to the distribution specified.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution=distribution,
                                  values=self.amp, axis=0)

    def std_curve(self, distribution='log-normal'):
        """Sample standard deviation associated with the mean H/V curve.

        Parameters
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of H/V curve, default is
                'log-normal'.

        Returns
        -------
        ndarray
            Sample standard deviation of H/V curve according to the
            distribution specified.

        Raises
        ------
        ValueError
            If only single time window is defined.
        KeyError
            If `distribution` does not match the available options.
        """
        return self._std_factory(distribution=distribution,
                                 values=self.amp, axis=0)

    @staticmethod
    def _nth_std_factory(n, distribution, mean, std):
        return Hvsr._nth_std_factory(n=n, distribution=distribution,
                                     mean=mean, std=std)

    def nstd_mean_curve(self, n, distribution):
        """Nth standard deviation on mean curve from all azimuths"""
        return self._nth_std_factory(n=n, distribution=distribution,
                                     mean=self.mean_curve(distribution=distribution),
                                     std=self.std_curve(distribution=distribution))

    def nstd_f0_frq(self, n, distribution):
        """Nth standard deviation on f0 from all azimuths"""
        return self._nth_std_factory(n=n, distribution=distribution,
                                     mean=self.mean_f0_frq(distribution=distribution),
                                     std=self.std_f0_amp(distribution=distribution))

    def mc_peak_amp(self, distribution='log-normal'):
        """Amplitude of the peak of the mean H/V curve.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        float
            Ampltiude associated with the peak of the mean H/V curve.

        """
        mc = self.mean_curve(distribution)
        return np.max(mc[Hvsr.find_peaks(mc)[0]])

    def mc_peak_frq(self, distribution='log-normal'):
        """Frequency of the peak of the mean H/V curve.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        float
            Frequency associated with the peak of the mean H/V curve.

        """
        mc = self.mean_curve(distribution)
        return float(self.frq[np.where(mc == np.max(mc[Hvsr.find_peaks(mc)[0]]))])


