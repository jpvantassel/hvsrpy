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

"""TimeSeries class definition."""

import warnings
import logging

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, sosfiltfilt, detrend

logger = logging.getLogger(__name__)

__all__ = ["TimeSeries"]


class TimeSeries():

    def __init__(self, amplitude, dt_in_seconds):
        """Initialize a ``TimeSeries`` object.

        Parameters
        ----------
        amplitude : iterable
            Amplitude of the time series at each time step.
        dt_in_seconds : float
            Time step between samples in seconds.

        Returns
        -------
        TimeSeries
            Instantiated with amplitude and time step information.

        Raises
        ------
        TypeError
            If ``amplitude`` is not castable to ``ndarray``, refer to
            error message(s) for specific details.

        """
        try:
            self.amplitude = np.array(amplitude, dtype=np.double)
        except ValueError:
            msg = "``amplitude`` must be convertable to numeric ``ndarray``."
            raise TypeError(msg)

        if self.amplitude.ndim != 1:
            msg = f"``amplitude`` must be 1-D, not {self.amplitude.ndim}-D."
            raise TypeError(msg)

        self.dt_in_seconds = float(dt_in_seconds)
        logger.info(f"Created {self}.")

    @property
    def n_samples(self):
        return len(self.amplitude)

    @property
    def fs(self):
        return 1/self.dt_in_seconds

    @property
    def fnyq(self):
        return 0.5*self.fs

    # TODO (jpv): Consider adding absolute time information.
    def time(self):
        return np.arange(self.n_samples)*self.dt_in_seconds

    def trim(self, start_time, end_time):
        """Trim in the interval ``[start_time, end_time]``.

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds.

        Returns
        -------
        None
            Updates the attributes ``amplitude`` and ``n_samples``.

        Raises
        ------
        IndexError
            If the ``start_time`` and/or ``end_time`` is illogical. Checks
            include ``start_time`` is less than zero, ``start_time`` is
            after ``end_time``, or ``end_time`` is after the end of the
            record.

        """
        current_time = self.time()
        start = 0
        end = current_time[-1]

        if start_time < start:
            msg = "Illogical start_time for trim; "
            msg += f"a start_time of {start_time:.2f} is before start of record."
            raise IndexError(msg)

        if start_time >= end_time:
            msg = "Illogical start_time for trim; "
            msg += f"start_time of {start_time:.2f} is greater than "
            msg += f"end_time of {end_time:.2f}."
            raise IndexError(msg)

        if end_time > end:
            msg = "Illogical end_time for trim; "
            msg += f"end_time of {end_time:.2f} must be less than "
            msg += f"duration of the the time series of {end:.2f}"
            raise IndexError(msg)

        start_index = np.argmin(np.absolute(current_time - start_time))
        end_index = np.argmin(np.absolute(current_time - end_time))

        self.amplitude = self.amplitude[start_index:end_index+1]

    def detrend(self, type="linear"):
        """Remove trend from ``TimeSeries``.

        Parameters
        ----------
        type = {"constant", "linear"}, optional
            Type of detrend. If ``type == "linear"`` (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If ``type == "constant"``, only the mean of data
            is subtracted.

        Returns
        -------
        None
            Performs inplace detrend on the ``amplitude`` attribute.

        """
        self.amplitude = detrend(self.amplitude, type=type)

    # TODO (jpv): Consider adding the ability to overlap windows.
    def split(self, window_length_in_seconds):
        """Split record into set of records.

        Parameters
        ----------
        window_length_in_seconds : float
            Duration of each split in seconds.

        Returns
        -------
        list
            List of ``TimeSeries`` objects, one per split.

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure an intuitive
            number of windows. Without this, for example, a 10-minute
            record could not be broken into 10, 1-minute records.

        """
        samples_per_window = int(window_length_in_seconds/self.dt_in_seconds) + 1
        n_windows = int(self.n_samples / (samples_per_window-1))

        if n_windows < 1:
            msg = f"Window length of {window_length_in_seconds} s is larger "
            msg += f"than the record length of {(self.n_samples-1)*self.dt_in_seconds} s."
            raise ValueError(msg)

        start_idx = 0
        windows = []
        for _ in range(n_windows):
            end_idx = start_idx + samples_per_window
            tseries = TimeSeries(self.amplitude[start_idx:end_idx], self.dt_in_seconds)
            windows.append(tseries)
            start_idx = end_idx - 1
        return windows

    def window(self, type="tukey", width=0.1):
        """Apply window to time series.

        Parameters
        ----------
        width : {0.-1.}
            Fraction of the time series to be windowed.
        type : {"tukey"}, optional
            If ``type="tukey"``, a width of ``0`` is a rectangular window
            and ``1`` is a Hann window, default is ``0.1`` indicating
            a 5% taper off of both ends of the time series.

        Returns
        -------
        None
            Applies window to the ``amplitude`` attribute in-place.

        """
        if type == "tukey":
            window = tukey(self.n_samples, alpha=width)
        else:
            msg = f"Window type {type} not recognized, try ['tukey',]."
            raise NotImplementedError(msg)

        self.amplitude *= window

    def butterworth_filter(self, fcs_in_hz, order=5):
        """Apply Butterworth filter.

        Parameters
        ----------
        fcs_in_hz : tuple
            Butterworth filter's corner frequencies in Hz. ``None`` can
            be used to specify a one-sided filter. For example a high
            pass filter at 3 Hz would be specified as
            ``fcs_in_hz=(3, None)``.
        order : int, optional
            Butterworth filter order, default is ``5``.

        Returns
        -------
        None
            Filters ``amplitude`` attribute in-place.

        """
        fc_low, fc_high = fcs_in_hz
        if fc_low is None and fc_high is not None:
            btype = "lowpass"
            wn = fc_high
        elif fc_low is not None and fc_high is None:
            btype = "highpass"
            wn = fc_low
        elif fc_low is not None and fc_high is not None:
            btype = "bandpass"
            wn = [fc_low, fc_high]
        else:
            msg = "No corner frequencies provided; no filtering performed."
            warnings.warn(msg)
            return None

        sos = butter(order, wn, btype, fs=self.fs, output='sos')
        self.amplitude = sosfiltfilt(sos, self.amplitude)

    @classmethod
    def from_trace(cls, trace):
        """Initialize a ``TimeSeries`` object from ``obspy`` ``Trace``."""
        return cls(trace.data, trace.stats.delta)

    @classmethod
    def from_timeseries(cls, timeseries):
        """Copy constructor for ``TimeSeries`` object.

        Parameters
        ----------
        timeseries : TimeSeries
            ``TimeSeries`` to be copied.

        Returns
        -------
        TimeSeries
            Copy of the provided ``TimeSeries`` object.

        """
        return cls(timeseries.amplitude, timeseries.dt_in_seconds)

    def is_similar(self, other):
        """Check if ``other`` is similar to ``self``."""
        if not isinstance(other, TimeSeries):
            return False

        if abs(other.dt_in_seconds - self.dt_in_seconds) > 1E-8:
            return False

        if other.n_samples != self.n_samples:
            return False

        return True

    def __eq__(self, other):
        """Check if ``other`` is equal to ``self``."""
        if not self.is_similar(other):
            return False

        if not np.allclose(self.amplitude, other.amplitude):
            return False

        return True

    def __str__(self):
        """Human-readable representation of ``TimeSeries``."""
        return f"TimeSeries with {self.n_samples} samples at {id(self)}."

    def __repr__(self):
        """Unambiguous representation of ``TimeSeries``."""
        return f"TimeSeries(amplitude={self.amplitude}, dt_in_seconds={self.dt_in_seconds})"
