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

"""TimeSeries class definition."""

import warnings
import logging

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt, detrend

from .windowed_timeseries import WindowedTimeSeries

logger = logging.getLogger("hvsrpy.timeseries")

__all__ = ['TimeSeries']


class TimeSeries():

    def __init__(self, amplitude, dt):
        """
        Initialize a `TimeSeries` object.

        Parameters
        ----------
        amplitude : iterable
            Amplitude of the time series at each time step.
        dt : float
            Time step between samples in seconds.

        Returns
        -------
        TimeSeries
            Instantiated with amplitude and time step information.

        Raises
        ------
        TypeError
            If `amplitude` is not castable to `ndarray`, refer to error
            message(s) for specific details.

        """
        try:
            self.amplitude = np.array(amplitude, dtype=np.double)
        except ValueError:
            msg = "`amplitude` must be convertable to numeric `ndarray`."
            raise TypeError(msg)

        if self.amplitude.ndim != 1:
            msg = f"`amplitude` must be 1-D, not {self.amplitude.ndim}-D."
            raise TypeError(msg)

        self.dt = float(dt)

    @property
    def nsamples(self):
        return len(self.amplitude)

    @property
    def fs(self):
        return 1/self.dt

    @property
    def fnyq(self):
        return 0.5*self.fs

    # @property
    # def df(self):
    #     return self.fs/self.nsamples

    @property
    def time(self):
        return np.arange(self.nsamples)*self.dt

    def trim(self, start_time, end_time):
        """Trim in the interval [`start_time`, `end_time`].

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds.

        Returns
        -------
        None
            Updates the attributes `amplitude` and `nsamples`.

        Raises
        ------
        IndexError
            If the `start_time` and/or `end_time` is illogical.
            For example, `start_time` is less than zero, `start_time` is
            after `end_time`, or `end_time` is after the end of the
            record.

        """
        current_time = self.time
        start = 0
        end = max(current_time)

        if start_time < start:
            msg = "Illogical `start_time` for trim; "
            msg += f"a `start_time` of {start_time} is before start of record."
            raise IndexError(msg)

        if start_time >= end_time:
            msg = "Illogical `start_time` for trim; "
            msg += f"`start_time` of {start_time} is greater than "
            msg += f"`end_time` of {end_time}."
            raise IndexError(msg)

        if end_time > end:
            msg = f"Illogical end_time for trim; "
            msg += f"`end_time` of {end_time} must be less than "
            msg += f"durection of the the timeseries of `{end:.2f}"
            raise IndexError(msg)

        start_index = np.argmin(np.absolute(current_time - start_time))
        end_index = np.argmin(np.absolute(current_time - end_time))

        self.amplitude = self.amplitude[start_index:end_index+1]

    def detrend(self, type="linear"):
        """Remove linear trend from `TimeSeries`.

        Parameters
        ----------
        type = {"constant", "linear"}, optional
            The type of detrending. If type == 'linear' (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If type == 'constant', only the mean of data is
            subtracted.

        Returns
        -------
        None
            Performs detrend on the `amplitude` attribute.

        """
        detrend(self.amplitude, type=type, inplace=True)

    def split(self, windowlength):
        """
        Split record into `n` series of length `windowlength`.

        Parameters
        ----------
        windowlength : float
            Duration of desired shorter series in seconds. If
            `windowlength` is not an integer multiple of `dt`, the
            window length is rounded to up to the next integer
            multiple of `dt`.

        Returns
        -------
        None
            Updates the object's internal attributes
            (e.g., `amplitude`).

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure an intuitive
            number of windows. Without this, for example, a 10-minute
            record could not be broken into 10 1-minute records.

        Examples
        --------
            >>> import numpy as np
            >>> from sigpropy import TimeSeries
            >>> amp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> tseries = TimeSeries(amp, dt=1)
            >>> wseries = tseries.split(2)
            >>> wseries.amplitude
            array([[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8]])
        """
        steps_per_win = int(windowlength/self.dt)
        nwindows = int((self.nsamples-1)/steps_per_win)
        rec_samples = (steps_per_win*nwindows)+1

        right_cols = np.reshape(self.amplitude[1:rec_samples],
                                (nwindows, steps_per_win))
        left_col = self.amplitude[:-steps_per_win:steps_per_win].T
        amplitudes = np.column_stack((left_col, right_cols))

        windows = [TimeSeries(amp, dt=self.dt) for amp in amplitudes]
        return WindowedTimeSeries(windows)

    def window(self, type="tukey", width=0.1):
        """Apply window to time series.
        
        Parameters
        ----------
        width : {0.-1.}
            Fraction of the time series to be windowed.
        type : {"tukey"}, optional

            `0` is equal to a rectangular and `1` a Hann window.

        Returns
        -------
        None
            Applies window to the `amplitude` attribute.

        """
        if type == "tukey":
            window = tukey(self.nsamples, alpha=width)
        else:
            raise NotImplementedError

        self.amplitude *= window 

    # def bandpassfilter(self, flow, fhigh, order=5):
    #     """
    #     Apply bandpass Butterworth filter to time series.
    #     Parameters
    #     ----------
    #     flow : float
    #         Low-cut frequency (content below `flow` is filtered).
    #     fhigh : float
    #         High-cut frequency (content above `fhigh` is filtered).
    #     order : int, optional
    #         Filter order, default is 5.
    #     Returns
    #     -------
    #     None
    #         Filters attribute `amplitude`.
    #     """
    #     fnyq = self.fnyq
    #     b, a = butter(order, [flow/fnyq, fhigh/fnyq], btype='bandpass')
    #     self._amp = filtfilt(b, a, self._amp, padlen=3*(max(len(b), len(a))-1))

    # @classmethod
    # def from_trace(cls, trace):
    #     """
    #     Initialize a `TimeSeries` object from a trace object.
    #     Parameters
    #     ----------
    #     trace : Trace
    #         Refer to
    #         `obspy documentation <https://github.com/obspy/obspy/wiki>`_
    #         for more information
    #     Returns
    #     -------
    #     TimeSeries
    #         Initialized with information from `trace`.
    #     """
    #     return cls(amplitude=trace.data, dt=trace.stats.delta)

    @classmethod
    def from_timeseries(cls, timeseries):
        """Copy constructor for `TimeSeries` object.

        Parameters
        ----------
        timeseries : TimeSeries
            `TimeSeries` to be copied.

        Returns
        -------
        TimeSeries
            Copy of the provided `TimeSeries` object.

        """
        return cls(timeseries.amplitude, timeseries.dt)

    def is_similar(self, other):
        """Check if `other` is similar to `self`."""
        if not isinstance(other, TimeSeries):
            return False

        if abs(other.dt - self.dt) < 1E-6:
            return False

        if other.nsamples != self.nsamples:
            return False

        return True

    def __eq__(self, other):
        """Check if `other` is equal to `self`."""
        for attr in ["nseries", "nsamples", "dt"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if not np.allclose(self.amplitude, other.amplitude):
            return False

        return True

    def __str__(self):
        """Human-readable representation of `TimeSeries`."""
        return f"TimeSeries with {self.nsamples} samples at {id(self)}."

    def __repr__(self):
        """Unambiguous representation of `TimeSeries`."""
        return f"TimeSeries(amplitude={self.amplitude}, dt={self.dt})"
