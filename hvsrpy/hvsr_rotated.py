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
        """Mean `f0` considering all valid timewindows of all azimuths.

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
        return self._mean_factory(distribution=distribution, self.peak_frq)

    def mean_f0_amp(self, distribution="log-normal"):
        """Mean `f0` amplitude considering all valid timewindows and
        azimuths.

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
        return self._mean_factory(distribution=distribution, self.peak_amp)

    def _mean_factory(self, distribution, values):
        n = self.azimuth_count
        mean = 0
        if distribution == "normal":
            for value in values:
                mean += np.mean(value, axis=-1)
            return mean/n
        elif distribution == "log-normal":
            for value in values:
                mean += np.mean(np.log(value), axis=-1)
            return np.exp(mean/n)
        else:
            msg = f"distribution type {distribution} not recognized."
            raise KeyError(msg)

    def _std_factory(self, distribution, values):
        n = self.azimuth_count
        mean = _mean_factory(distribution, values)
        num = 0
        wi2 = 0
        if distribution == "normal":
            for value in values:
                diff = np.sum(value - mean, axis=-1)
                wi = 1/(n*len(value))
                num += diff*diff*wi
                wi2 += wi*wi
            return np.sqrt(num/(1-wi2))
        elif distribution == "log-normal":
            for value in values:
                diff = np.sum(np.log(value) - mean, axis=-1)
                wi = 1/(n*len(value))
                num += diff*diff*wi
                wi2 += wi*wi
            return np.sqrt(num/(1-wi2))
        else:
            msg = f"distribution type {distribution} not recognized."
            raise KeyError(msg)

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
        return self._std_factory(distribution=distribution, self.peak_frq)

    def std_f0_amp(self, distribution='log-normal'):
        """Sample standard deviation of amplitude of `f0` of valid
        time windows.

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
        return self._std_factory(distribution=distribution, self.peak_amp)

    @property
    def _valid_amps(self):
        return [hv.amp[hv.valid_window_indices] for hv in self.hvsrs]

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
        return self._mean_factory(distribution=distribution, values=self._valid_amps)

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
        return self._std_factory(distribution=distribution, values=self._valid_amps)
