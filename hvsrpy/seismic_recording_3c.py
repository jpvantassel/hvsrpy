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

"""Class definition of SeismicRecording3C, 3-component seismic record."""

from .timeseries import TimeSeries

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

    def __init__(self, ns, ew, vt, meta=None):
        """Initialize a 3-component seismic recording object.

        Parameters
        ----------
        ns, ew, vt : TimeSeries
            `TimeSeries` object for each component.
        meta : dict, optional
            Meta information for object, default is `None`.

        Returns
        -------
        SeismicRecording3C
            Initialized 3-component sensor object.

        """
        tseries = []
        for component in [ns, ew, vt]:
            if not ns.is_similar(component):
                msg = "All components must be similar."
                raise ValueError(msg)
            tseries.append(TimeSeries.from_timeseries(component))
        self.ns, self.ew, self.vt = tseries

        meta = {} if meta is None else meta
        self.meta = {"File Name(s)": "Was not created from file", **meta}
    