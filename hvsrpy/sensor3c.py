# This file is part of hvsrpy a Python module for horizontal-to-vertical 
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

"""This file contains the 3-component sensor (Sensor3c) class."""

import numpy as np
from hvsrpy import Hvsr
from sigpropy import TimeSeries, FourierTransform
import obspy
import logging
logging.getLogger()


class Sensor3c():
    """Class for creating and manipulating 3-component sensor objects.

    Attributes
    ----------
    ns, ew, vt : Timeseries
        `TimeSeries` object for each component.
    ns_f, ew_f, vt_f : FourierTransform
        `FourierTransform` object for each component.
    normalization_factor : float
        Maximum value of `ns`, `ew`, and `vt` amplitude used for
        normalization when plotting.
    """

    @staticmethod
    def _check_input(values_dict):
        """Perform checks on inputs

        Specifically:
            1. Ensure all components are `TimeSeries` objects.
            2. Ensure all components have equal `dt`.
            3. Ensure all components have same `n_samples`. If not trim
            components to a common length.

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
        if not isinstance(values_dict["ns"], TimeSeries):
            msg = f"'ns' must be a `TimeSeries`, not {type(values_dict['ns'])}."
            raise TypeError(msg)
        dt = values_dict["ns"].dt
        delay = values_dict["ns"].delay
        n_samples = values_dict["ns"].n_samples
        flag_cut = False
        for key, value in values_dict.items():
            if not isinstance(value, TimeSeries):
                msg = f"{key} must be a `TimeSeries`, not {type(value)}."
                raise TypeError(msg)
            if value.dt != dt:
                msg = f"All components must have equal `dt`."
                raise ValueError(msg)
            if value.delay != delay:
                msg = f"All components must have equal `dt`."
                raise ValueError(msg)

            if value.n_samples != n_samples:
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

    def __init__(self, ns, ew, vt):
        """Initalize a 3-component sensor (Sensor3c) object.

        Parameters
        ----------
        ns, ew, vt : timeseries
            `TimeSeries` object for each component.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.
        """
        self.ns, self.ew, self.vt = self._check_input({"ns": ns,
                                                       "ew": ew,
                                                       "vt": vt})
        self.ns_f = None
        self.ew_f = None
        self.vt_f = None

    @property
    def normalization_factor(self):
        """Return sensor time history normalization factor."""
        return max(max(self.ns.amp.flatten()), max(self.ew.amp.flatten()), max(self.vt.amp.flatten()))
        
    @classmethod
    def from_mseed(cls, fname):
        """Initialize a 3-component sensor (Sensor3c) object from a
        .miniseed file.

        Parameters
        ----------
        fname : str
            Name of miniseed file, full path may be used if desired.
            The file should contain three traces with the 
            appropriate channel names. Refer to the `SEED` Manual 
            `here <https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.
            for specifics.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.
        """
        traces = obspy.read(fname)

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
                msg = f"Missing, duplicate, or incorrectly named components. See documentation."
                raise ValueError(msg)

        return cls(ns, ew, vt)

    def split(self, windowlength):
        """Split component `TimeSeries`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.split(windowlength)

    def detrend(self):
        """Detrend component `TimeSeries`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()

    def bandpassfilter(self, flow, fhigh, order):
        """Bandpassfilter component `TimeSeries`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)

    def cosine_taper(self, width):
        """Cosine taper component `TimeSeries`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)

    def transform(self):
        """Perform Fourier transform on component `TimeSeries`.

        Returns
        -------
        None
            Redefines attributes `ew_f`, `ns_f`, and `vt_f` as 
            `FourierTransform` objects for each component.
        """
        self.ew_f = FourierTransform.from_timeseries(self.ew)
        self.ns_f = FourierTransform.from_timeseries(self.ns)
        self.vt_f = FourierTransform.from_timeseries(self.vt)

    def smooth(self, bandwidth):
        """Smooth component `FourierTransforms`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.smooth_konno_ohmachi(bandwidth)

    def resample(self, fmin, fmax, fn, res_type, inplace):
        """Resample component `FourierTransforms`.

        Refer to `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_ documentation for details.
        """
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.resample(fmin, fmax, fn, res_type, inplace)

    def combine_horizontals(self, method='squared-average'):
        """Combine two horizontal components (`ns` and `ew`).

        Parameters
        ----------
        method : {'squared-averge', 'geometric-mean'}, optional
            Defines how the two horizontal components are combined 
            to represent a single horizontal component, the default
            is 'squared-average'.

        Returns
        -------
        FourierTransform
            Representing the combined horizontal components.
        """
        if method == 'squared-average':
            horizontal = np.sqrt(
                (self.ns_f.amp*self.ns_f.amp + self.ew_f.amp*self.ew_f.amp)/2)
        elif method == 'geometric-mean':
            horizontal = np.sqrt(self.ns_f.amp * self.ew_f.amp)
        else:
            msg = f"ratio_type {method} has not been implemented."
            raise NotImplementedError(msg)
        return FourierTransform(horizontal, self.ns_f.frq)

    def hv(self, windowlength, bp_filter, taper_width, bandwidth, resampling, method):
        """Prepare time series and Fourier transforms then compute H/V.

        Parameters
        ----------
        windowlength : float
            Length of time windows in seconds.
        bp_filter : dict
            Bandpass filter settings, of the form 
            {'flag':`bool`, 'flow':`float`, 'fhigh':`float`,
            'order':`int`}.
        taper_width : float
            Width of cosine taper.
        bandwidth : float
            Bandwidth of the Konno and Ohmachi smoothing window.
        resampling : dict
            Resampling settings, of the form 
            {'minf':`float`, 'maxf':`float`, 'nf':`int`, 
            'res_type':`str`}.
        method : {'squared-averge', 'geometric-mean'}
            Refer to :meth:`combine_horizontals <Sensor3c.combine_horizontals>` for details.

        Returns
        -------
        Hvsr

        Notes
        -----
        More information for the above arguements can be found in
        the documenation of `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_.
        """
        # Time Domain Effects
        # Split
        self.split(windowlength)

        # Detrend
        self.detrend()

        # Filter
        if bp_filter["flag"]:
            self.bandpassfilter(flow=bp_filter["flow"],
                                fhigh=bp_filter["fhigh"],
                                order=bp_filter["order"])
        
        # Cosine Taper
        self.cosine_taper(width=taper_width)

        # Frequency Domain Effects
        self.transform()

        for comp in [self.ns_f, self.ew_f, self.vt_f]:
            comp.amp = comp.mag

        # H/V Effects
        hor = self.combine_horizontals(method=method)
        hor.smooth_konno_ohmachi(bandwidth)
        self.vt_f.smooth_konno_ohmachi(bandwidth)

        # H/V
        hvsr = FourierTransform(hor.amp/self.vt_f.amp, hor.frq)
        hvsr.resample(minf=resampling["minf"],
                      maxf=resampling["maxf"],
                      nf=resampling["nf"],
                      res_type=resampling["res_type"],
                      inplace=True)

        return Hvsr(hvsr.amp, hvsr.frq, find_peaks=False)
