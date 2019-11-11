"""This file contains a class for creating and manipulating 3-component
sensor objects (Sensor3c)."""

import numpy as np
from hvsrpy import Hvsr
from sigpropy import TimeSeries, FourierTransform
import obspy
import logging
logging.getLogger()


class Sensor3c():
    """Class for creating and manipulating 3-component sensor objects.

    Attributes:
        ns, ew, vt : Timeseries
            Timeseries object for each component.
        ns_f, ew_f, vt_f : FourierTransform
            FourierTransform object for each component.
    """

    @staticmethod
    def _check_input(values_dict):
        """Perform checks on inputs

        Specifically:
            1. Ensure all components are TimeSeries objects.
            2. Ensure all components have equal `dt`.
            3. Ensure all components have same `n_samples`. If not trim
            components to a common length.

        Args:
            values_dict : dict
                Key is human readable component name {'ns', 'ew', 'vt'}.
                Value is corresponding TimeSeries object.

        Returns:
            Tuple of checked components.
        """
        if not isinstance(values_dict["ns"], TimeSeries):
            msg = f"'ns' must be a TimeSeries, not {type(values_dict['ns'])}."
            raise TypeError(msg)
        dt = values_dict["ns"].dt
        delay = values_dict["ns"].delay
        n_samples = values_dict["ns"].n_samples
        flag_cut = False
        for key, value in values_dict.items():
            if not isinstance(value, TimeSeries):
                msg = f"{key} must be a TimeSeries, not {type(value)}."
                raise TypeError(msg)
            if value.dt != dt:
                msg = f"All components must have equal dt."
                raise ValueError(msg)
            if value.delay != delay:
                msg = f"All components must have equal dt."
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

        Args:
            ns, ew, vt : timeseries
                Timeseries object for each component.

        Returns:
            Initialized 3-component sensor (Sensor3c) object.
        """
        self.ns, self.ew, self.vt = self._check_input(
            {"ns": ns, "ew": ew, "vt": vt})
        self.ns_f = None
        self.ew_f = None
        self.vt_f = None

    @classmethod
    def from_mseed(cls, fname):
        """Initialize a 3-component sensor (Sensor3c) object from a
        miniseed file.

        Args:
            fname : str
                Name of miniseed file, full path may be used if desired.
                The file should contain three traces with the 
                appropriate channel names. Refer to SEED Manual for 
                specifics (https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf).

        Returns:
            Initialized 3-component sensor (Sensor3c) object.
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
                msg = f"Missing, duplicate, or incorrectly named component. See documentation."
                raise ValueError(msg)

        return cls(ns, ew, vt)

    def split(self, windowlength):
        """Split component TimeSeries.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.split(windowlength)

    def detrend(self):
        """Detrend component TimeSeries.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()

    def bandpassfilter(self, flow, fhigh, order):
        """Bandpassfilter component TimeSeries.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)

    def cosine_taper(self, width):
        """Cosine taper component TimeSeries.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)

    def transform(self):
        """Perform Fourier transform on components.

        Returns:
            `None`, redefines attributes `ew_f`, `ns_f`, and `vt_f` as 
            FourierTransform objects for each component.
        """
        self.ew_f = FourierTransform.from_timeseries(self.ew)
        self.ns_f = FourierTransform.from_timeseries(self.ns)
        self.vt_f = FourierTransform.from_timeseries(self.vt)

    def smooth(self, bandwidth):
        """Smooth component FourierTransforms.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.smooth_konno_ohmachi(bandwidth)

    def resample(self, fmin, fmax, fn, res_type, inplace):
        """Resample component FourierTransforms.

        Refer to `sigpropy` documentation for details.
        """
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.resample(fmin, fmax, fn, res_type, inplace)

    def combine_horizontals(self, method='squared-average'):
        """Combine two horizontal components (ns and ew).

        Args:
            ratio_type : {'squared-averge', 'geometric-mean'}, optional
                Defines how the two horizontal components are combined 
                to represent a single horizontal component. By default
                the 'square-average' approach is used.

        Return:
            A FourierTransform object representing the combined
            horizontal component.
        """
        if method == 'squared-average':
            horizontal = np.sqrt(
                (self.ns_f.amp*self.ns_f.amp + self.ew_f.amp*self.ew_f.amp)/2)
        elif method == 'geometric-mean':
            horizontal = np.sqrt(self.ns_f.amp * self.ew_f.amp)
        else:
            raise NotImplementedError(
                f"ratio_type {method} has not been implemented.")
        return FourierTransform(horizontal, self.ns_f.frq)

    # TODO (jpv): Refactor hv methods
    def hv(self, windowlength, bp_filter, taper_width, bandwidth, resampling, method, find_peaks=False):
        """Prepare time series and fourier transform and compute H/V.

        Args:
            windowlength : float
                Length of time windows in seconds.
            bp_filter : dict
                Bandpass filter settings, `dict` is of the form 
                {'flag':`bool`, 'flow':`float`, 'fhigh':`float`, 
                'order':`int`} refer to `SigProPy` documentation for
                details.
            taper_width : float
                Width of cosine taper.
            bandwidth : float
                Width of Konno and Ohmachi Smoothing window.
            resampling : dict
                Resampling settings, `dict` is of the form 
                {'minf':`float`, 'maxf':`float`, 'nf':`int`, 
                'res_type':`str`} refer to `SigProPy` documentation for
                details.
            method : {'squared-averge', 'geometric-mean'}
                Refer to method `combine_horizontals` for details.
            find_peaks : bool
                Refer to method `calc_hv` for details.

        Returns:
            Initialized Hvsr object.

        Notes:
            More information for the above arguements can be found in
            the documenation of `SigProPy`.
        """
        self.split(windowlength)
        self.detrend()
        if bp_filter["flag"]:
            self.bandpassfilter(flow=bp_filter["flow"],
                                fhigh=bp_filter["fhigh"],
                                order=bp_filter["order"])
        self.cosine_taper(width=taper_width)
        self.transform()

        for comp in [self.ns_f, self.ew_f, self.vt_f]:
            comp.amp = comp.mag

        # Horizontal Component
        hor = self.combine_horizontals(method=method)
        hor.smooth_konno_ohmachi(bandwidth)

        # Vertical Component
        self.vt_f.smooth_konno_ohmachi(bandwidth)

        # H/V
        hvsr = FourierTransform(hor.amp/self.vt_f.amp, hor.frq)
        hvsr.resample(minf=resampling["minf"],
                     maxf=resampling["maxf"],
                     nf=resampling["nf"],
                     res_type=resampling["res_type"],
                     inplace=True)

        return Hvsr(hvsr.amp, hvsr.frq, find_peaks=find_peaks)
