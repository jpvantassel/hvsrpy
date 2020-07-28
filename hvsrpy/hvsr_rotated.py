# This file is part of hvsrpy, a Python package for horizontal-to-vertical
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
from pandas import DataFrame

from hvsrpy import Hvsr, __version__

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

    def __init__(self, hvsr, azimuth, meta=None):
        """Instantiate a `HvsrRotated` object.

        Parameters
        ----------
        hvsr : Hvsr
            `Hvsr` object.
        azimuth : float
            Rotation angle in degrees measured clockwise positive from
            north (i.e., 0 degrees).

        Returns
        -------
        HvsrRotated
            Instantiated `HvsrRotated` object.

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
            3. `az` is greater than 0.

        """
        if not isinstance(hvsr, Hvsr):
            raise TypeError("`hvsr` must be an instance of `Hvsr`.")

        az = float(az)

        if (az < 0) or (az > 180):
            raise ValueError(f"`azimuth` must be between 0 and 180, not {az}.")

        return (hvsr, az)

    def append(self, hvsr, azimuth):
        """Append `Hvsr` object at a new azimuth.

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
        """Create HvsrRotated from iterable of Hvsr objects."""
        obj = cls(hvsrs[0], azimuths[0], meta=meta)
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
        distribution : {'normal', 'log-normal'}, optional
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
        """Sample standard deviation of `f0` for all valid time windows
        across all azimuths.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
        float
            Sample standard deviation of `f0`.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution=distribution,
                                 values=self.peak_frq)

    def std_f0_amp(self, distribution='log-normal'):
        """Sample standard deviation of `f0` amplitude for all valid
        time windows across all azimuths.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of `f0`, default is 'log-normal'.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of `f0` according
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
        """Mean curve for each azimuth

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of mean curve, default is 'log-normal'.

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

    def mean_curves_peak(self, distribution="log-normal"):
        """Peak from each mean curve, one per azimuth."""
        frqs, amps = np.empty(self.azimuth_count), np.empty(self.azimuth_count)
        for az_cnt, hvsr in enumerate(self.hvsrs):
            frqs[az_cnt] = hvsr.mc_peak_frq(distribution=distribution)
            amps[az_cnt] = hvsr.mc_peak_amp(distribution=distribution)
        return (frqs, amps)

    def mean_curve(self, distribution='log-normal'):
        """Mean H/V curve considering all valid windows and azimuths.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of mean curve, default is 'log-normal'.

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
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Assumed distribution of H/V curve, default is 'log-normal'.

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

    def nstd_curve(self, n, distribution):
        """Nth standard deviation on mean curve from all azimuths."""
        return self._nth_std_factory(n=n, distribution=distribution,
                                     mean=self.mean_curve(distribution=distribution),
                                     std=self.std_curve(distribution=distribution))

    def nstd_f0_frq(self, n, distribution):
        """Nth standard deviation on `f0` from all azimuths"""
        return self._nth_std_factory(n=n, distribution=distribution,
                                     mean=self.mean_f0_frq(distribution=distribution),
                                     std=self.std_f0_frq(distribution=distribution))

    def mc_peak_amp(self, distribution='log-normal'):
        """Amplitude of the peak of the mean H/V curve.

        Parameters
        ----------
        distribution : {'normal', 'log-normal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        float
            Amplitude associated with the peak of the mean H/V curve.

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

    def _stats(self, distribution_f0):
        if distribution_f0 == "log-normal":
            columns = ["Lognormal Median", "Lognormal Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [1/self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)]])

        elif distribution_f0 == "normal":
            columns = ["Means", "Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [np.nan, np.nan]])
        else:
            msg = f"`distribution_f0` of {distribution_f0} is not implemented."
            raise NotImplementedError(msg)

        df = DataFrame(data=data, columns=columns,
                       index=["Fundamental Site Frequency, f0,AZ",
                              "Fundamental Site Period, T0,AZ"])
        return df

    def print_stats(self, distribution_f0, places=2):  # pragma: no cover
        """Print basic statistics of `Hvsr` instance."""
        display(self._stats(distribution_f0=distribution_f0).round(places))

    def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for hvsrpy-style file."""
        # `f0` from windows
        mean_f = self.mean_f0_frq(distribution_f0)
        sigm_f = self.std_f0_frq(distribution_f0)
        ci_68_lower_f = self.nstd_f0_frq(-1, distribution_f0)
        ci_68_upper_f = self.nstd_f0_frq(+1, distribution_f0)

        # mean curve
        mc = self.mean_curve(distribution_mc)
        mc_peak_frq = self.mc_peak_frq(distribution_mc)
        mc_peak_amp = self.mc_peak_amp(distribution_mc)
        _min = self.nstd_curve(-1, distribution_mc)
        _max = self.nstd_curve(+1, distribution_mc)

        rejection = "False" if self.meta.get('Performed Rejection') is None else "True"

        n_windows = self.hvsrs[0].n_windows
        n_accepted = sum([len(hvsr.valid_window_indices) for hvsr in self.hvsrs])
        n_rejected = self.azimuth_count*n_windows - n_accepted
        lines = [
            f"# hvsrpy output version {__version__}",
            f"# File Name (),{self.meta.get('File Name')}",
            f"# Window Length (s),{self.meta.get('Window Length')}",
            f"# Total Number of Windows per Azimuth (),{n_windows}",
            f"# Total Number of Azimuths (),{self.azimuth_count}",
            f"# Frequency Domain Window Rejection Performed (),{rejection}",
            f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
            f"# Number of Accepted Windows (),{n_accepted}",
            f"# Number of Rejected Windows (),{n_rejected}",
            f"# Distribution of f0 (),{distribution_f0}"]

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        if distribution_f0 == "log-normal":
            mean_t = 1/mean_f
            sigm_t = sigm_f
            ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
            ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

            lines += [
                f"# Median f0 (Hz) [LMf0,AZ],{fclean(mean_f)}",
                f"# Log-normal standard deviation f0 () [SigmaLNf0,AZ],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Median T0 (s) [LMT0,AZ],{fclean(mean_t)}",
                f"# Log-normal standard deviation T0 () [SigmaLNT0,AZ],{fclean(sigm_t)}",
                f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
            ]

        else:
            lines += [
                f"# Mean f0 (Hz) [f0,AZ],{fclean(mean_f)}",
                f"# Standard deviation f0 (Hz) [Sigmaf0,AZ],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Mean T0 (s) [LMT0,AZ],NAN",
                f"# Standard deviation T0 () [SigmaT0,AZ],NAN",
                f"# 68 % Confidence Interval T0 (s),NAN",
            ]

        c_type = "Median" if distribution_mc == "log-normal" else "Mean"
        lines += [
            f"# {c_type} Curve Distribution (),{distribution_mc}",
            f"# {c_type} Curve Peak Frequency (Hz) [f0mc,AZ],{fclean(mc_peak_frq)}",
            f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
            f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

        return _lines

    def to_file(self, fname, distribution_f0, distribution_mc):  # pragma: no cover
        """Save H/V data to file.

        Parameters
        ----------
        fname : str
            Name of file to save the results, may be the full or a
            relative path.
        distribution_f0 : {'log-normal', 'normal'}, optional
            Assumed distribution of `f0` from the time windows, the
            default is 'log-normal'.
        distribution_mc : {'log-normal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'log-normal'.

        Returns
        -------
        None
            Writes file to disk.

        """
        lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)

        with open(fname, "w") as f:
            for line in lines:
                f.write(line)
