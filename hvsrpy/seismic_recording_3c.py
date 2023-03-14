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

"""Class definition of SeismicRecording3C, a 3-component seismic record."""

import json

import numpy as np

from .timeseries import TimeSeries

__all__ = ["SeismicRecording3C"]


class SeismicRecording3C():
    """Class for creating and manipulating 3-component seismic records.

    Attributes
    ----------
    ns : TimeSeries
        North-south component, time domain.
    ew : TimeSeries
        East-west component, time domain.
    vt : TimeSeries
        Vertical component, time domain.

    """

    def __init__(self, ns, ew, vt, degrees_from_north=0., meta=None):
        """Initialize a 3-component seismic recording object.

        Parameters
        ----------
        ns, ew, vt : TimeSeries
            ``TimeSeries`` object for each component.
        degrees_from_north : float, optional
            Orientation of the ``ns`` component (i.e., station north)
            relative to magnetic north measured in decimal degrees
            (clockwise positive). The default value is ``0``. indicating
            station north and magnetic north are aligned.
        meta : dict, optional
            Meta information for object, default is ``None``.

        Returns
        -------
        SeismicRecording3C
            Initialized 3-component sensor object.

        """
        tseries = []
        for name, component in zip(["ns", "ew", "vt"], [ns, ew, vt]):
            if not ns.is_similar(component):
                msg = f"Component {name} is not similar to component ns; "
                msg += "all components must be similar."
                raise ValueError(msg)
            tseries.append(TimeSeries.from_timeseries(component))
        self.ns, self.ew, self.vt = tseries

        # ensure less than 360 degrees
        self.degrees_from_north = float(degrees_from_north - 360*(degrees_from_north // 360))

        meta = {} if meta is None else meta
        self.meta = {"file name(s)": "seismic recording was not created from file",
                     "deployed degrees from north": self.degrees_from_north,
                     "current degrees from north": self.degrees_from_north,
                     **meta}

    def trim(self, start_time, end_time):
        """Trim component ``TimeSeries``.

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
        self.meta["trim"] = (start_time, end_time) 
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).trim(start_time=start_time,
                                          end_time=end_time)

    def detrend(self, type="linear"):
        """Remove trend from component ``TimeSeries``.

        Parameters
        ----------
        type : {"constant", "linear"}, optional
            Type of detrending. If ``type == "linear"`` (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If ``type == "constant"``, only the mean of data
            is subtracted.

        Returns
        -------
        None
            Performs inplace detrend on the ``amplitude`` attribute.

        """
        self.meta["detrend"] = type
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).detrend(type=type)

    def split(self, window_length_in_seconds):
        """Split component ``TimeSeries`` into time windows.

        Parameters
        ----------
        window_length_in_seconds : float
            Duration of each split in seconds.

        Returns
        -------
        list
            List of ``SeismicRecording3C`` objects, one per split.

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure an intuitive
            number of windows. Without this, for example, a 10-minute
            record could not be broken into 10, 1-minute records.

        """
        self.meta["split"] = window_length_in_seconds
        split_recordings = []
        for (_ns, _ew, _vt) in zip(self.ns.split(window_length_in_seconds),
                                   self.ew.split(window_length_in_seconds),
                                   self.vt.split(window_length_in_seconds)):
            split_recordings.append(SeismicRecording3C(_ns, _ew, _vt,
                                                       degrees_from_north=self.degrees_from_north,
                                                       meta=self.meta))
        return split_recordings

    def window(self, type="tukey", width=0.1):
        """Window component ``TimeSeries``.

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
        self.meta["window_type_and_width"] = (type, width)
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).window(type=type, width=width)

    def butterworth_filter(self, fcs_in_hz, order=5):
        """Butterworth filter component ``TimeSeries``.

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
        self.meta["butterworth_filter"] = fcs_in_hz
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).butterworth_filter(fcs_in_hz=fcs_in_hz,
                                                        order=order)

    def orient_sensor_to(self, degrees_from_north):
        """Orient sensor's horizontal components.

        Parameters
        ----------
        degrees_from_north : float
            New sensor orientation in degrees from north
            (clockwise positive). The sensor's north component will be
            oriented such that it is aligned with the defined
            orientation.

        Returns
        -------
        None
            Modifies the objects internal state.

        """
        angle_diff_degrees = degrees_from_north - self.degrees_from_north
        angle_diff_radians = np.radians(angle_diff_degrees)
        c = np.cos(angle_diff_radians)
        s = np.sin(angle_diff_radians)

        ew = self.ew.amplitude
        ns = self.ns.amplitude

        self.ew.amplitude = ew*c - ns*s
        self.ns.amplitude = ew*s + ns*c

        self.degrees_from_north = degrees_from_north
        self.meta["current degrees from north"] = degrees_from_north

    def _to_dict(self):
        return dict(dt_in_seconds=self.ns.dt_in_seconds,
                    ns_amplitude=self.ns.amplitude.tolist(),
                    ew_amplitude=self.ew.amplitude.tolist(),
                    vt_amplitude=self.vt.amplitude.tolist(),
                    degrees_from_north=self.degrees_from_north,
                    meta=self.meta)

    @classmethod
    def _from_dict(cls, data):
        ns = TimeSeries(data["ns_amplitude"], data["dt_in_seconds"])
        ew = TimeSeries(data["ew_amplitude"], data["dt_in_seconds"])
        vt = TimeSeries(data["vt_amplitude"], data["dt_in_seconds"])
        degrees_from_north = data["degrees_from_north"]
        meta = data["meta"]
        return cls(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)

    def save(self, fname):
        with open(fname, "w") as f:
            json.dump(self._to_dict(), f)

    @classmethod
    def load(cls, fname):
        with open(fname, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def from_seismic_recording_3c(cls, seismic_recording_3c):
        """Copy constructor for ``SeismicRecording3C`` object.

        Parameters
        ----------
        seismic_recording_3c : SeismicRecording3C
            ``SeismicRecording3C`` to be copied.

        Returns
        -------
        SeismicRecording3C
            Copy of the provided ``SeismicRecording3C`` object.

        """
        new_components = []
        for component in ["ns", "ew", "vt"]:
            tseries = getattr(seismic_recording_3c, component)
            new_components.append(tseries.from_timeseries(tseries))
        return cls(*new_components,
                   degrees_from_north=seismic_recording_3c.degrees_from_north,
                   meta=seismic_recording_3c.meta)

    def is_similar(self, other):
        """Check if ``other`` is similar to ``self``."""
        if not isinstance(other, SeismicRecording3C):
            return False

        for attr in ["ns", "ew", "vt"]:
            if not getattr(self, attr).is_similar(getattr(other, attr)):
                return False

        return True

    def __eq__(self, other):
        """Check if ``other`` is equal to ``self``."""
        if not self.is_similar(other):
            return False

        for attr in ["ns", "ew", "vt", "meta"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for attr, tol in [("degrees_from_north", 0.1)]:
            if abs(getattr(self, attr) - getattr(other, attr)) > tol:
                return False

        return True

    def __str__(self):
        """Human-readable representation of ``SeismicRecording3C`` object."""
        return f"SeismicRecording3C at {id(self)}"

    def __repr__(self):
        """Unambiguous representation of ``SeismicRecording3C`` object."""
        return f"SeismicRecording3C(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"
