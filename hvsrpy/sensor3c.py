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

"""Class definition for Sensor3c, a 3-component sensor."""

import math
import logging
import json
import warnings

import numpy as np
import obspy
from sigpropy import TimeSeries, FourierTransform, WindowedTimeSeries, FourierTransformSuite

from hvsrpy import Hvsr, HvsrRotated

logger = logging.getLogger(__name__)

__all__ = ["Sensor3c"]


class Sensor3c():
    """Class for creating and manipulating 3-component sensor objects.

    Attributes
    ----------
    ns : TimeSeries
        North-south component, time domain.
    ew : TimeSeries
        East-west component, time domain.
    vt : TimeSeries
        Vertical component, time domain.

    """

    @staticmethod
    def _check_input(values_dict):
        """Perform checks on inputs.

        Specifically:
        1. Ensure all components are `TimeSeries` objects.
        2. Ensure all components have equal `dt`.
        3. Ensure all components have same `nsamples`. If not trim
        components to the common length.

        Parameters
        ----------
        values_dict : dict
            Key is human readable component name {'ns', 'ew', 'vt'}.
            Value is corresponding `TimeSeries` object.

        Returns
        -------
        Tuple
            Containing checked components.

        """
        ns = values_dict["ns"]
        if not isinstance(ns, TimeSeries):
            msg = f"'ns' must be a `TimeSeries`, not {type(ns)}."
            raise TypeError(msg)
        dt = ns.dt
        nsamples = ns.nsamples
        flag_cut = False
        for key, value in values_dict.items():
            if key == "ns":
                continue
            if not isinstance(value, TimeSeries):
                msg = f"`{key}`` must be a `TimeSeries`, not {type(value)}."
                raise TypeError(msg)
            if value.dt != dt:
                msg = "All components must have equal `dt`."
                raise ValueError(msg)
            if value.nsamples != nsamples:
                logging.info("Components are not of the same length.")
                flag_cut = True

        if flag_cut:
            min_time = 0
            max_time = np.inf
            for value in values_dict.values():
                min_time = max(min_time, min(value.time))
                max_time = min(max_time, max(value.time))
            logging.info(f"Trimming between {min_time} and {max_time}.")
            for value in values_dict.values():
                value.trim(min_time, max_time)

        return (values_dict["ns"], values_dict["ew"], values_dict["vt"])

    def __init__(self, ns, ew, vt, meta=None):
        """Initialize a 3-component sensor (Sensor3c) object.

        Parameters
        ----------
        ns, ew, vt : TimeSeries
            `TimeSeries` object for each component.
        meta : dict, optional
            Meta information for object, default is None.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.

        """
        self.ns, self.ew, self.vt = self._check_input({"ns": ns,
                                                       "ew": ew,
                                                       "vt": vt})
        self.meta = {} if meta is None else meta

    @property
    def normalization_factor(self):
        """Time history normalization factor across all components."""
        factor = 1E-6
        for attr in ["ns", "ew", "vt"]:
            cmax = np.max(np.abs(getattr(self, attr).amp.flatten()))
            factor = cmax if cmax > factor else factor
        return factor

    @classmethod
    def from_mseed(cls, fname=None, fnames_1c=None):
        """Create 3-component sensor (Sensor3c) object from .mseed file.

        Parameters
        ----------
        fname : str, optional
            Name of miniseed file, full path may be used if desired.
            The file should contain three traces with the
            appropriate channel names. Refer to the `SEED` Manual
            `here <https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.
            for specifics, default is `None`.
        fnames_1c : dict, optional
            Some data acquisition systems supply three separate miniSEED
            files rather than a single combined file. To use those types
            of files, simply specify the three files in a `dict` of
            the form `{'e':'east.mseed', 'n':'north.mseed',
            'z':'vertical.mseed'}`, default is `None`.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.

        Raises
        ------
        ValueError
            If both `fname` and `fname_verbose` are `None`.

        """
        if fnames_1c is None and fname is None:
            msg = "`fnames_1c` and `fname` cannot both be `None`."
            raise ValueError(msg)
        if fnames_1c is not None:
            trace_list = []
            for key in ["e", "n", "z"]:
                stream = obspy.read(fnames_1c[key], format="MSEED")
                if len(stream) > 1:
                    msg = f"File {fnames_1c[key]} contained {len(stream)}"
                    msg += "traces, rather than 1 as was expected."
                    raise IndexError(msg)
                trace = stream[0]
                if trace.meta.channel[-1] != key.capitalize():
                    msg = "Component indicated in the header of "
                    msg += f"{fnames_1c[key]} is {trace.meta.channel[-1]} "
                    msg += f"which does not match the key {key} specified. "
                    msg += "Ignore this warning only if you know "
                    msg += "your digitizer's header is incorrect."
                    warnings.warn(msg)
                    trace.meta.channel = trace.meta.channel[:-1] + \
                        key.capitalize()
                trace_list.append(trace)
            traces = obspy.Stream(trace_list)
        else:
            traces = obspy.read(fname, format="MSEED")

        if len(traces) != 3:
            msg = f"miniseed file {fname} has {len(traces)} traces, but should have 3."
            raise ValueError(msg)

        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel.endswith("E") and not found_ew:
                ew = TimeSeries.from_trace(trace)
                found_ew = True
            elif trace.meta.channel.endswith("N") and not found_ns:
                ns = TimeSeries.from_trace(trace)
                found_ns = True
            elif trace.meta.channel.endswith("Z") and not found_vt:
                vt = TimeSeries.from_trace(trace)
                found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components. See documentation."
                raise ValueError(msg)

        meta = {"File Name": fname}
        return cls(ns, ew, vt, meta)

    def to_dict(self):
        """Dictionary representation of `Sensor3c` object.

        Returns
        -------
        dict
            With all of the components of the `Sensor3c`.

        """
        dictionary = {}
        for name in ["ns", "ew", "vt"]:
            value = getattr(self, name).to_dict()
            dictionary[name] = value
        dictionary["meta"] = self.meta
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        """Create `Sensor3c` object from dictionary representation.

        Parameters
        ---------
        dictionary : dict
            Must contain keys "ns", "ew", "vt", and may also contain
            the optional key "meta". "ns", "ew", and "vt" must be
            dictionary representations of `TimeSeries` objects, see
            `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
            documentation for details.

        Returns
        -------
        Sensor3c
            Instantiated `Sensor3c` object.

        """
        components = []
        for comp in ["ns", "ew", "vt"]:
            components.append(TimeSeries.from_dict(dictionary[comp]))
        return cls(*components, meta=dictionary.get("meta"))

    def to_json(self):
        """Json string representation of `Sensor3c` object.

        Returns
        -------
        str
            With all of the components of the `Sensor3c`.

        """
        dictionary = self.to_dict()
        return json.dumps(dictionary)

    @classmethod
    def from_json(cls, json_str):
        """Create `Sensor3c` object from Json-string representation.

        Parameters
        ---------
        json_str : str
            Json-style string, which must contain keys "ns", "ew", and
            "vt", and may also contain the optional key "meta". "ns",
            "ew", and "vt" must be Json-style string representations of
            `TimeSeries` objects, see
            `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
            documentation for details.

        Returns
        -------
        Sensor3c
            Instantiated `Sensor3c` object.

        """
        dictionary = json.loads(json_str)
        return cls.from_dict(dictionary)

    def split(self, windowlength):
        """Split component `TimeSeries` into `WindowedTimeSeries`.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for attr in ["ew", "ns", "vt"]:
            wtseries = WindowedTimeSeries.from_timeseries(getattr(self, attr),
                                                          windowlength)
            setattr(self, attr, wtseries)

    def detrend(self):
        """Detrend components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()

    def bandpassfilter(self, flow, fhigh, order):  # pragma: no cover
        """Bandpassfilter components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)

    def cosine_taper(self, width):
        """Cosine taper components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)

    def transform(self, **kwargs):
        """Perform Fourier transform on components.

        Returns
        -------
        dict
            With `FourierTransform`-like objects, one for for each
            component, indicated by the key 'ew','ns', 'vt'.

        """
        ffts = {}
        for attr in ["ew", "ns", "vt"]:
            tseries = getattr(self, attr)
            if isinstance(tseries, WindowedTimeSeries):
                fft = FourierTransformSuite.from_timeseries(tseries, **kwargs)
            elif isinstance(tseries, TimeSeries):
                fft = FourierTransform.from_timeseries(tseries, **kwargs)
            else:
                raise NotImplementedError
            ffts[attr] = fft
        return ffts

    def combine_horizontals(self, method, horizontals, azimuth=None):
        """Combine two horizontal components (`ns` and `ew`).

        Parameters
        ----------
        method : {'squared-average', 'geometric-mean', 'single-azimuth', 'multiple-azimuths'}
            Defines how the two horizontal components are combined
            to represent a single horizontal component.
        horizontals : dict
            If combination is done in the frequency-domain (i.e.,
            `method in ['squared-average', 'geometric-mean']`)
            horizontals is a `dict` of `FourierTransform` objects,
            see :meth:`transform <Sensor3c.transform>` for details. If
            combination is done in the time-domain
            (i.e., `method in ['single-azimuth', 'multiple-azimuths']`)
            horizontals is a `dict` of `TimeSeries` objects.
        azimuth : float, optional
            Valid only if `method` is `single-azimuth` in which case an
            azimuth (clockwise positive) from North (i.e., 0 degrees) is
            required.

        Returns
        -------
        TimeSeries or FourierTransform
            Depending upon the specified `method` requires the
            combination to happen in the time or frequency domain.

        """
        if method in ["squared-average", "geometric-mean"]:
            return self._combine_horizontal_fd(method, horizontals)
        elif method in ["azimuth", "single-azimuth"]:
            return self._combine_horizontal_td(method, horizontals, azimuth=azimuth)
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

    @staticmethod
    def _combine_horizontal_fd(method, horizontals, **kwargs):
        ns = horizontals["ns"]
        ew = horizontals["ew"]

        if method == "squared-average":
            horizontal = np.sqrt((ns.mag*ns.mag + ew.mag*ew.mag)/2)
        elif method == "geometric-mean":
            horizontal = np.sqrt(ns.mag * ew.mag)
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

        if isinstance(ns, FourierTransformSuite):
            return FourierTransformSuite(horizontal, ns.frq)
        elif isinstance(ns, FourierTransform):
            return FourierTransform(horizontal, ns.frq)
        else:
            raise NotImplementedError

    def _combine_horizontal_td(self, method, horizontals, azimuth):
        az_rad = math.radians(azimuth)
        ns = horizontals["ns"]
        ew = horizontals["ew"]

        if method in ["azimuth", "single-azimuth"]:
            horizontal = ns.amp*math.cos(az_rad) + ew.amp*math.sin(az_rad)
        else:
            msg = f"method={method} has not been implemented."
            raise NotImplementedError(msg)

        if isinstance(ns, WindowedTimeSeries):
            return WindowedTimeSeries(horizontal, ns.dt)
        elif isinstance(ns, TimeSeries):
            return TimeSeries(horizontal, ns.dt)
        else:
            raise NotImplementedError

    def hv(self, windowlength, bp_filter, taper_width, bandwidth,
           resampling, method, f_low=None, f_high=None, azimuth=None):
        """Prepare time series and Fourier transforms then compute H/V.

        Parameters
        ----------
        windowlength : float
            Length of time windows in seconds.
        bp_filter : dict
            Bandpass filter settings, of the form
            `{'flag':bool, 'flow':float, 'fhigh':float, 'order':int}`.
        taper_width : float
            Width of cosine taper, value between `0.` and `1.`.
        bandwidth : float
            Bandwidth (b) of the Konno and Ohmachi (1998) smoothing
            window.
        resampling : dict
            Resampling settings, of the form
            `{'minf':float, 'maxf':float, 'nf':int, 'res_type':str}`.
        method : {'squared-averge', 'geometric-mean', 'single-azimuth', 'multiple-azimuths'}
            Refer to :meth:`combine_horizontals <Sensor3c.combine_horizontals>`
            for details.
        f_low, f_high : float, optional
            Upper and lower frequency limits to restrict peak selection,
            default is `None` meaning search range will not be
            restricted.
        azimuth : float, optional
            Refer to
            :meth:`combine_horizontals <Sensor3c.combine_horizontals>`
            for details.

        Returns
        -------
        Hvsr
            Instantiated `Hvsr` object.

        """
        if bp_filter["flag"]:
            self.bandpassfilter(flow=bp_filter["flow"],
                                fhigh=bp_filter["fhigh"],
                                order=bp_filter["order"])
        self.split(windowlength)
        self.detrend()
        self.cosine_taper(width=taper_width)

        if method in ["squared-average", "geometric-mean", "azimuth", "single-azimuth"]:
            if method == "azimuth":
                msg = "method='azimuth' is deprecated, replace with the more descriptive 'single-azimuth'."
                warnings.warn(msg, DeprecationWarning)
                method = "single-azimuth"
            return self._make_hvsr(method=method, resampling=resampling,
                                   bandwidth=bandwidth, f_low=f_low,
                                   f_high=f_high, azimuth=azimuth)

        elif method in ["rotate", "multiple-azimuths"]:
            if method == "rotate":
                msg = "method='rotate' is deprecated, replace with the more descriptive 'multiple-azimuths'."
                warnings.warn(msg, DeprecationWarning)
                method = "multiple-azimuths"
            hvsrs = np.empty(len(azimuth), dtype=object)
            for index, az in enumerate(azimuth):
                hvsrs[index] = self._make_hvsr(method="single-azimuth",
                                               resampling=resampling,
                                               bandwidth=bandwidth,
                                               f_low=f_low,
                                               f_high=f_high,
                                               azimuth=az)
            return HvsrRotated.from_iter(hvsrs, azimuth, meta=self.meta)

        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

    def _make_hvsr(self, method, resampling, bandwidth, f_low=None, f_high=None, azimuth=None):
        if method in ["squared-average", "geometric-mean"]:
            ffts = self.transform()
            hor = self.combine_horizontals(method=method, horizontals=ffts)
            ver = ffts["vt"]
            del ffts
        elif method == "single-azimuth":
            hor = self.combine_horizontals(method=method,
                                           horizontals={"ew": self.ew,
                                                        "ns": self.ns},
                                           azimuth=azimuth)
            if isinstance(hor, WindowedTimeSeries):
                hor = FourierTransformSuite.from_timeseries(hor)
                ver = FourierTransformSuite.from_timeseries(self.vt)
            elif isinstance(hor, TimeSeries):
                hor = FourierTransform.from_timeseries(hor)
                ver = FourierTransform.from_timeseries(self.vt)
            else:
                raise NotImplementedError
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

        self.meta["method"] = method
        self.meta["azimuth"] = azimuth

        if resampling["res_type"] == "linear":
            frq = np.linspace(resampling["minf"],
                              resampling["maxf"],
                              resampling["nf"])
        elif resampling["res_type"] == "log":
            frq = np.geomspace(resampling["minf"],
                               resampling["maxf"],
                               resampling["nf"])
        else:
            raise NotImplementedError

        hor.smooth_konno_ohmachi_fast(frq, bandwidth)
        ver.smooth_konno_ohmachi_fast(frq, bandwidth)
        hor.amp /= ver.amp
        hvsr = hor
        del ver

        if self.ns.n_windows == 1:
            window_length = max(self.ns.time)
        else:
            window_length = max(self.ns.time[0])

        self.meta["Window Length"] = window_length

        return Hvsr(hvsr.amp, hvsr.frq, find_peaks=False, f_low=f_low, f_high=f_high, meta=self.meta)

    def __iter__(self):
        """Iterable representation of a Sensor3c object."""
        return iter((self.ns, self.ew, self.vt))

    def __str__(self):
        """Human-readable representation of `Sensor3c` object."""
        return "Sensor3c"

    def __repr__(self):
        """Unambiguous representation of `Sensor3c` object."""
        return f"Sensor3c(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"
