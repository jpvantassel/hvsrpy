"""This file contains a class for creating and manipulating 3-component
sensor objects (Sensor3c)."""

from hvpy import Sensor
from sigpropy import TimeSeries, FourierTransform 

class Sensor3c(Sensor):
    """Derived class for creating and manipulating 3-component sensor 
    objects."""
    
    def __init__(self):
        pass

    @classmethod
    def from_mseed(cls):
        pass

    def hv(self):
        pass

    def hv_reject(self):
        pass
