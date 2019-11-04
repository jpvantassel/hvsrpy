"""This file contains a class for creating and manipulating 3-component
sensor objects (Sensor3c)."""

from hvsrpy import Sensor
from sigpropy import TimeSeries
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

    @classmethod
    def from_mseed(cls, fname):
        """Initialize a 3-component sensor (Sensor3c) object from a
        miniseed file.

        Args:
            fname : str
                Name of miniseed file, full path may be used if desired.
                The file should contain three traces labeled 'BHE', 
                'BHZ', and 'BHN' for each orthogonal component.
            
        Returns:
            Initialized 3-component sensor (Sensor3c) object.
        """

        traces = obspy.read(fname)

        if len(traces) != 3:
            raise ValueError(f"miniseed file {fname} has {len(traces)} number of traces, but should have 3.")

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
                raise ValueError(f"Missing, duplicate, or incorrectly named component. See documentation.")
            
        return cls(ns, ew, vt)

    def hv(self):
        pass

    def hv_reject(self):
        pass
