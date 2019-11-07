"""This file contains a class for creating and manipulating 3-component
sensor objects (Sensor3c)."""

import numpy as np
from hvsrpy import Sensor, Hvsr
from sigpropy import TimeSeries, FourierTransform
import obspy
import logging
logging.getLogger()


class Sensor3c(Sensor):
    """Derived class for creating and manipulating 3-component sensor 
    objects.

    Attributes:
        ns, ew, vt : timeseries
            Timeseries object for each component.

    """

    @staticmethod
    def check_input():
        pass

    def __init__(self, ns, ew, vt):
        """Initalize a 3-component sensor (Sensor3c) object.

        Args:
            ns, ew, vt : timeseries
                See class attributes for details.

        Returns:
            Initialized 3-component sensor (Sensor3c) object.
        """
        # TODO (jpv): Write checks on input.
        self.ns = ns
        self.ew = ew
        self.vt = vt
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
                appropriate channel labels of 'BHE', 'BHZ', and 'BHN'
                for each orthogonal component.

        Returns:
            Initialized 3-component sensor (Sensor3c) object.
        """

        traces = obspy.read(fname)

        if len(traces) != 3:
            raise ValueError(
                f"miniseed file {fname} has {len(traces)} number of traces, but should have 3.")

        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel == "BHE" and not found_ew:
                ew = TimeSeries.from_trace(trace)
                found_ew = True
            elif trace.meta.channel == "BHN" and not found_ns:
                ns = TimeSeries.from_trace(trace)
                found_ns = True
            elif trace.meta.channel == "BHZ" and not found_vt:
                vt = TimeSeries.from_trace(trace)
                found_vt = True
            else:
                raise ValueError(
                    f"Missing, duplicate, or incorrectly named component. See documentation.")

        return cls(ns, ew, vt)

    def split(self, windowlength):
        """Split components."""
        for comp in [self.ew, self.ns, self.vt]:
            comp.split(windowlength)

    def bandpassfilter(self, flow, fhigh, order):
        """Bandpassfilter components."""
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)

    def cosine_taper(self, width):
        """Cosine taper components."""
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)

    def transform(self):
        """Perform Fourier Transform on Signals."""
        self.ew_f = FourierTransform.from_timeseries(self.ew)
        self.ns_f = FourierTransform.from_timeseries(self.ns)
        self.vt_f = FourierTransform.from_timeseries(self.vt)
        # TODO (jpv): Add checking frequency vector are same.

    def smooth(self, bandwidth):
        """Konno and Ohmachi smooth components."""
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.smooth_konno_ohmachi(bandwidth)

    def resample(self, fmin, fmax, fn, res_type, inplace):
        for comp in [self.ew_f, self.ns_f, self.vt_f]:
            comp.resample(fmin, fmax, fn, res_type, inplace)

    def calc_hv(self, ratio_type='squared-average', find_peaks=False):
        """Calculate H/V ratio.

        Args:
            ratio_type : {'squared-averge', 'geometric-mean'}
        """
        if ratio_type == 'squared-average':
            horizontal = np.sqrt((self.ns_f.amp**2 + self.ew_f.amp**2)/2)
        elif ratio_type == 'geometric-mean':
            horizontal = np.sqrt(self.ns_f.amp * self.ew_f.amp)
        else:
            raise NotImplementedError(f"ratio_type {ratio_type} has not been implemented.")
        return Hvsr(horizontal/self.vt_f.amp, self.vt_f.frq, find_peaks=find_peaks)

    def hv(self, windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type, find_peaks=False):
        """Calculate Hvsr"""
        self.split(windowlength)
        self.bandpassfilter(flow, fhigh, forder)
        self.cosine_taper(width)
        self.transform()
        self.smooth(bandwidth)
        self.resample(fmin, fmax, fn, res_type, inplace=True)
        return self.calc_hv(ratio_type, find_peaks)

    def hv_reject(self, windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type, n, max_iter):
        """Perform Hvsr calculation with rejection."""
        hvsr = self.hv(windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type, find_peaks=True)
        hvsr.reject_windows(n, max_iter)
        return hvsr
