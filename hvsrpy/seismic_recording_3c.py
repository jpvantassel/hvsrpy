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

"""Class definition of SeismicRecording3C, a 3-component seismic record."""


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
            `TimeSeries` object for each component.
        degrees_from_north : float, optional
            Orientation of the `ns` component (i.e., station north)
            relative to magnetic north measured in decimal degrees
            (clockwise positive). The default value is `0.` indicating
            station north and magnetic north are aligned.
        meta : dict, optional
            Meta information for object, default is `None`.

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
            tseries.append(ns.from_timeseries(component))
        self.ns, self.ew, self.vt = tseries

        # TODO (jpv): Add degrees from north and associated rotation ability
        # throughout hvsrpy workflow.
        self.degrees_from_north = float(degrees_from_north)

        meta = {} if meta is None else meta
        self.meta = {"File Name(s)": "Was not created from file", **meta}

    def trim(self, start_time, end_time):
        """Trim component `TimeSeries`."""
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).trim(start_time=start_time,
                                          end_time=end_time)

    def detrend(self, type="linear"):
        """Remove trend from component `TimeSeries`."""
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).detrend(type=type)

    def split(self, window_length_in_seconds):
        """Split component `TimeSeries`."""
        split_recordings = []
        for (_ns, _ew, _vt) in zip(self.ns.split(window_length_in_seconds),
                                   self.ew.split(window_length_in_seconds),
                                   self.vt.split(window_length_in_seconds)):
            split_recordings.append(SeismicRecording3C(_ns, _ew, _vt))
        return split_recordings

    def window(self, type="tukey", width=0.1):
        """Window component `TimeSeries`."""
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).window(type=type, width=width)

    def butterworth_filter(self, fcs, order=5):
        """Butterworth filter component `TimeSeries`."""
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).butterworth_filter(fcs=fcs, order=order)

    # TODO(jpv): Include full docstrings for SeismicRecording3C methods.
    # Can likely adopt these straight from TimeSeries.

    @classmethod
    def from_seismic_recording_3c(cls, seismic_recording_3c):
        # TODO(jpv): Add docstring.
        original = seismic_recording_3c
        new_components = []
        for component in ["ns", "ew", "vt"]:
            tseries = getattr(original, component)
            new_components.append(tseries.from_timeseries(tseries))
        return cls(*new_components,
                   degrees_from_north=original.degrees_from_north,
                   meta=original.meta)
