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

"""Class definition for Hvsr object."""

import logging
import warnings

import numpy as np
import scipy.signal as sg
from pandas import DataFrame

from .meta import __version__

logger = logging.getLogger(name=__name__)

class Hvsr():
    """Class for creating and manipulating horizontal-to-vertical
    spectral ratio objects.

    Attributes
    ----------
    amp : ndarray
        Array of HVSR amplitudes. Each row represents an individual
        curve and each column a frequency.
    frq : ndarray
        Vector of frequencies corresponds to each column.
    n_windows : int
        Number of windows in `Hvsr` object.
    valid_window_indices : ndarray
        Array of indices indicating valid windows.

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
            Name of `value` to be checked, used solely for
            meaningful error messages.
        value : iterable
            Value to be checked.

        Returns
        -------
        ndarray
            `values` as ndarray of doubles.

        Raises
        ------
        TypeError
            If `value` is not castable to an ndarray of doubles.
        ValueError
            If `value` contains nan or a value less than or equal to
            zero.

        """
        try:
            value = np.array(value, dtype=np.double)
        except ValueError:
            msg = f"{name} must be castable to ndarray of doubles, not {type(value)}."
            raise TypeError(msg)

        if np.isnan(value).any():
            raise ValueError(f"{name} may not contain nan.")

        if np.sum(value < 0):
            raise ValueError(f"{name} must be > 0.")

        return value

    @staticmethod
    def correct_distribution(distribution):
        if distribution == "log-normal":
            msg = "distribution='log-normal' is deprecated use 'lognormal' instead."
            warnings.warn(msg, PendingDeprecationWarning)
            distribution = "lognormal"
        return distribution

    def __init__(self, amplitude, frequency, find_peaks=True, meta=None):
        """Create `Hvsr` from iterable of amplitude and frequency.

        Parameters
        ----------
        amplitude : ndarray
            Array of HVSR amplitudes. Each row represents an individual
            curve and each column a frequency.
        frequency : ndarray
            Vector of frequencies, corresponding to each column.
        find_peaks : bool, optional
            Indicates whether peaks of Hvsr will be found when created,
            default is `True`.
        meta : dict, optional
            Meta information about the object, default is `None`.

        Returns
        -------
        Hvsr
            Initialized with `amplitude` and `frequency`.
        """
        self.frq = self._check_input("frequency", frequency)
        nfrqs = len(self.frq)

        self.amp = self._check_input("amplitude", amplitude)
        try:
            self.amp = self.amp.reshape((self.amp.size//nfrqs, nfrqs))
        except ValueError as e:
            msg = f"Size of amplitude={self.amp.size} and frequency={self.frq.size} must be compatable."
            raise ValueError(msg) from e

        self.n_windows = self.amp.shape[0]
        self.valid_window_indices = np.ones(self.n_windows, dtype=bool)
        self._main_peak_frq = np.zeros(self.n_windows)
        self._main_peak_amp = np.zeros(self.n_windows)
        self._initialized_peaks = find_peaks
        if find_peaks:
            self.update_peaks()
        self.meta = meta

    @property
    def rejected_window_indices(self):
        """Rejected window indices."""
        return np.invert(self.valid_window_indices)
        # return np.array([cid for cid in range(self.n_windows) if cid not in self.valid_window_indices])

    @property
    def peak_frq(self):
        """Valid peak frequency vector."""
        if not self._initialized_peaks:
            self.update_peaks()
        return self._main_peak_frq[self.valid_window_indices]

    @property
    def peak_amp(self):
        """Valid peak amplitude vector."""
        if not self._initialized_peaks:
            self.update_peaks()
        return self._main_peak_amp[self.valid_window_indices]

    @staticmethod
    def find_peaks(amp, **kwargs):
        """Indices of all peaks in `amp`.

        Wrapper method for `scipy.signal.find_peaks` function.

        Parameters
        ----------
        amp : ndarray
            2D array of amplitudes. See `amp` attribute for details.
        **kwargs : dict
            Refer to
            `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            documentation.

        Returns
        -------
        Tuple
            Of the form `(peaks, settings)`. Where `peaks` is an
            `ndarray` or `list` of `ndarray` (one per window) of peak
            indices, and `settings` is `dict`, refer to
            `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            documentation for details.

        """
        peaks = []
        for c_amp in amp:
            peak, settings = sg.find_peaks(c_amp, **kwargs)
            peaks.append(peak.tolist())
        return (peaks, settings)

    def update_peaks(self, **kwargs):
        """Update with the lowest frequency, highest amplitude peak.

        Parameters
        ----------
        **kwargs : dict
            Refer to :meth:`find_peaks <Hvsr.find_peaks>` documentation.

        Returns
        -------
        None
            Updates `peaks` attribute.

        """
        if not self._initialized_peaks:
            self._initialized_peaks = True

        peak_indices, _ = self.find_peaks(self.amp[self.valid_window_indices],
                                          **kwargs)
        valid_indices = np.zeros_like(self.valid_window_indices, dtype=bool)
        for c_window, valid in enumerate(self.valid_window_indices):
            if not valid:
                continue

            c_window_peaks = peak_indices[c_window]

            try:
                c_index = np.nonzero(self.amp[c_window] == np.max(self.amp[c_window, c_window_peaks]))[0]
                # TODO (jpv): If a Hvsr curve has two peaks of equal
                # amplitude, arbitrarily take the first.
                if len(c_index) > 1:
                    c_index = c_index[0]
                self._main_peak_amp[c_window] = self.amp[c_window, c_index]
                self._main_peak_frq[c_window] = self.frq[c_index]
                valid_indices[c_window] = True
            except ValueError as e:
                if len(c_window_peaks) == 0:
                    logger.warning(f"No peak found in window #{c_window}.")
                else:
                    raise e
        self.valid_window_indices = np.array(valid_indices, dtype=bool)

    @staticmethod
    def _mean_factory(distribution, values, **kwargs):
        distribution = Hvsr.correct_distribution(distribution)

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
        KeyError
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
        KeyError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution, self.peak_amp)

    @staticmethod
    def _std_factory(distribution, values, **kwargs):
        distribution = Hvsr.correct_distribution(distribution)

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
        KeyError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution, self.peak_frq)

    def std_f0_amp(self, distribution='lognormal'):
        """Sample standard deviation of amplitude of `f0` of valid
        time windows.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Assumed distribution of `f0`, default is 'lognormal'.

        Returns
        -------
        float
            Sample standard deviation of the amplitude of f0.

        Raises
        ------
        KeyError
            If `distribution` does not match the available options.

        """
        return self._std_factory(distribution, self.peak_amp)

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
        KeyError
            If `distribution` does not match the available options.

        """
        return self._mean_factory(distribution, self.amp[self.valid_window_indices], axis=0)

    def std_curve(self, distribution='lognormal'):
        """Sample standard deviation associated with the mean HVSR curve.

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
        KeyError
            If `distribution` does not match the available options.

        """
        if self.n_windows == 1:
            msg = "The standard deviation of the mean curve is not defined for a single window."
            raise ValueError(msg)
        else:
            return self._std_factory(distribution, self.amp[self.valid_window_indices], axis=0)

    def mc_peak_frq(self, distribution='lognormal'):
        """Frequency of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        float
            Frequency associated with the peak of the mean HVSR curve.

        """
        mc = self.mean_curve(distribution)
        mc = mc.reshape((1, mc.size))
        return float(self.frq[np.where(mc[0] == np.max(mc[0, self.find_peaks(mc)[0]]))])

    def mc_peak_amp(self, distribution='lognormal'):
        """Amplitude of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : {'normal', 'lognormal'}, optional
            Refer to :meth:`mean_curve <Hvsr.mean_curve>` for details.

        Returns
        -------
        float
            Amplitude associated with the peak of the mean HVSR curve.

        """
        mc = self.mean_curve(distribution)
        mc = mc.reshape((1, mc.size))
        return np.max(mc[0, self.find_peaks(mc)[0]])

    def reject_windows(self, n=2, max_iterations=50,
                       distribution_f0='lognormal',
                       distribution_mc='lognormal'):
        """Perform rejection of spurious HVSR windows.

        Parameters
        ----------
        n : float, optional
            Number of standard deviations from the mean, default
            value is 2.
        max_iterations : int, optional
            Maximum number of rejection iterations, default value is
            50.
        distribution_f0 : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0` from time windows, the
            default is 'lognormal'.
        distribution_mc : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'lognormal'.

        Returns
        -------
        int
            Number of iterations required for convergence.

        """
        if isinstance(self.meta, dict):
            self.meta["n"] = n
            self.meta["Performed Rejection"] = "True"
        else:
            self.meta = {"n": n, "Performed Rejection": "True"}

        if not self._initialized_peaks:
            self.update_peaks()

        for c_iteration in range(1, max_iterations+1):

            logger.debug(f"c_iteration: {c_iteration}")
            logger.debug(f"valid_window_indices: {self.valid_window_indices}")

            mean_f0_before = self.mean_f0_frq(distribution_f0)
            std_f0_before = self.std_f0_frq(distribution_f0)
            mc_peak_frq_before = self.mc_peak_frq(distribution_mc)
            d_before = abs(mean_f0_before - mc_peak_frq_before)

            logger.debug(f"\tmean_f0_before: {mean_f0_before}")
            logger.debug(f"\tstd_f0_before: {std_f0_before}")
            logger.debug(f"\tmc_peak_frq_before: {mc_peak_frq_before}")
            logger.debug(f"\td_before: {d_before}")

            lower_bound = self.nstd_f0_frq(-n, distribution_f0)
            upper_bound = self.nstd_f0_frq(+n, distribution_f0)

            for c_window, (c_valid, c_peak) in enumerate(zip(self.valid_window_indices, self._main_peak_frq)):
                if not c_valid:
                    continue

                if c_peak > lower_bound and c_peak < upper_bound:
                    self.valid_window_indices[c_window] = True
                else:
                    self.valid_window_indices[c_window] = False

            mean_f0_after = self.mean_f0_frq(distribution_f0)
            std_f0_after = self.std_f0_frq(distribution_f0)
            mc_peak_frq_after = self.mc_peak_frq(distribution_mc)
            d_after = abs(mean_f0_after - mc_peak_frq_after)

            logger.debug(f"\tmean_f0_after: {mean_f0_after}")
            logger.debug(f"\tstd_f0_after: {std_f0_after}")
            logger.debug(f"\tmc_peak_frq_after: {mc_peak_frq_after}")
            logger.debug(f"\td_after: {d_after}")

            if d_before == 0 or std_f0_before == 0 or std_f0_after == 0:
                msg = f"Performed {c_iteration} iterations, returning b/c 0 values."
                logger.warning(msg)
                return c_iteration

            d_diff = abs(d_after - d_before)/d_before
            s_diff = abs(std_f0_after - std_f0_before)

            logger.debug(f"\td_diff: {d_diff}")
            logger.debug(f"\ts_diff: {s_diff}")

            if (d_diff < 0.01) and (s_diff < 0.01):
                # self.valid_window_indices = old_indices
                msg = f"Performed {c_iteration} iterations, returning b/c rejection converged."
                logger.info(msg)
                return c_iteration

    @staticmethod
    def _nth_std_factory(n, distribution, mean, std):
        distribution = Hvsr.correct_distribution(distribution)

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
            Number of standard deviations away from the mean `f0` from
            the valid time windows.
        distribution : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0`, the default is 'lognormal'.

        Returns
        -------
        float
            Value n standard deviations from mean `f0`.

        """
        return self._nth_std_factory(n, distribution,
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
        return self._nth_std_factory(n, distribution,
                                     self.mean_f0_amp(distribution),
                                     self.std_f0_amp(distribution))

    def nstd_curve(self, n, distribution="lognormal"):
        """nth standard deviation curve.

        Parameters
        ----------
        n : float
            Number of standard deviations away from the mean curve.
        distribution : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'lognormal'.

        Returns
        -------
        ndarray
            nth standard deviation curve.

        """
        return self._nth_std_factory(n, distribution,
                                     self.mean_curve(distribution),
                                     self.std_curve(distribution))

    def _stats(self, distribution_f0):
        distribution_f0 = self.correct_distribution(distribution_f0)

        if distribution_f0 == "lognormal":
            columns = ["Lognormal Median", "Lognormal Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [1/self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)]])

        elif distribution_f0 == "normal":
            columns = ["Mean", "Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [np.nan, np.nan]])
        else:
            msg = f"`distribution_f0` of {distribution_f0} is not implemented."
            raise NotImplementedError(msg)

        df = DataFrame(data=data, columns=columns,
                       index=["Fundamental Site Frequency, f0",
                              "Fundamental Site Period, T0"])
        return df

    def print_stats(self, distribution_f0, places=2):  # pragma: no cover
        """Print basic statistics of `Hvsr` instance."""
        display(self._stats(distribution_f0=distribution_f0).round(places))

    def _geopsy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for Geopsy-style file."""
        # f0 from windows
        mean = self.mean_f0_frq(distribution_f0)
        lower = self.nstd_f0_frq(-1, distribution_f0)
        upper = self.nstd_f0_frq(+1, distribution_f0)

        # mean curve
        mc = self.mean_curve(distribution_mc)
        mc_peak_frq = self.mc_peak_frq(distribution_mc)
        # mc_peak_amp = self.mc_peak_amp(distribution_mc)
        _min = self.nstd_curve(-1, distribution_mc)
        _max = self.nstd_curve(+1, distribution_mc)

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        lines = [
            f"# hvsrpy output version {__version__}",
            f"# Number of windows = {len(self.valid_window_indices)}",
            f"# f0 from average\t{fclean(mc_peak_frq)}",
            f"# Number of windows for f0 = {len(self.valid_window_indices)}",
            f"# f0 from windows\t{fclean(mean)}\t{fclean(lower)}\t{fclean(upper)}",
            f"# Peak amplitude\t{fclean(mc[np.where(self.frq == mc_peak_frq)][0])}",
            f"# Position\t{0} {0} {0}",
            f"# Category\tDefault",
            f"# Frequency\tAverage\tMin\tMax",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, a_i, n_i, x_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i}\t{a_i}\t{n_i}\t{x_i}\n")

        return _lines

    def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for hvsrpy-style file."""
        # f0 from windows
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

        n_rejected = self.n_windows - len(self.valid_window_indices)
        rejection = "False" if self.meta.get('Performed Rejection') is None else "True"
        lines = [
            f"# hvsrpy output version {__version__}",
            f"# File Name (),{self.meta.get('File Name')}",
            f"# Method (),{self.meta.get('method')}",
            f"# Azimuth (),{self.meta.get('azimuth')}",
            f"# Window Length (s),{self.meta.get('Window Length')}",
            f"# Total Number of Windows (),{self.n_windows}",
            f"# Frequency Domain Window Rejection Performed (),{rejection}",
            f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
            f"# Number of Accepted Windows (),{self.n_windows-n_rejected}",
            f"# Number of Rejected Windows (),{n_rejected}",
            f"# Distribution of f0 (),{distribution_f0}"]

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        distribution_f0 = self.correct_distribution(distribution_f0)
        distribution_mc = self.correct_distribution(distribution_mc)

        if distribution_f0 == "lognormal":
            mean_t = 1/mean_f
            sigm_t = sigm_f
            ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
            ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

            lines += [
                f"# Median f0 (Hz) [LMf0],{fclean(mean_f)}",
                f"# Lognormal standard deviation f0 () [SigmaLNf0],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Median T0 (s) [LMT0],{fclean(mean_t)}",
                f"# Lognormal standard deviation T0 () [SigmaLNT0],{fclean(sigm_t)}",
                f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
            ]

        else:
            lines += [
                f"# Mean f0 (Hz),{fclean(mean_f)}",
                f"# Standard deviation f0 (Hz) [Sigmaf0],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Mean T0 (s) [LMT0],NA",
                f"# Standard deviation T0 () [SigmaT0],NA",
                f"# 68 % Confidence Interval T0 (s),NA",
            ]

        c_type = "Median" if distribution_mc == "lognormal" else "Mean"
        lines += [
            f"# {c_type} Curve Distribution (),{distribution_mc}",
            f"# {c_type} Curve Peak Frequency (Hz) [f0mc],{fclean(mc_peak_frq)}",
            f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
            f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

        return _lines

    def to_file(self, fname, distribution_f0, distribution_mc, data_format="hvsrpy"):  # pragma: no cover
        """Save HVSR data to summary file.

        Parameters
        ----------
        fname : str
            Name of file to save the results, may be a full or
            relative path.
        distribution_f0 : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0` from the time windows, the
            default is 'lognormal'.
        distribution_mc : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'lognormal'.
        data_format : {'hvsrpy', 'geopsy'}, optional
            Format of output data file, default is 'hvsrpy'.

        Returns
        -------
        None
            Writes file to disk.

        """
        if data_format == "geopsy":
            lines = self._geopsy_style_lines(distribution_f0, distribution_mc)
        elif data_format == "hvsrpy":
            lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)
        else:
            raise NotImplementedError(f"data_format={data_format} is unknown.")

        with open(fname, "w") as f:
            for line in lines:
                f.write(line)
