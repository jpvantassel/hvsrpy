# This file is part of hvsrpy a Python package for horizontal-to-vertical
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

"""Definition of Settings class and its descendents."""

import json

import numpy as np

from .metadata import __version__


class Settings():

    def __init__(self):
        self.attrs = ["hvsrpy_version"]
        self.hvsrpy_version = __version__

    @property
    def attr_dict(self):
        return {name: getattr(self, name) for name in (self.attrs)}

    def extend_attributes(self, attributes_with_defaults):
        self.attrs.extend([attributes_with_defaults.keys()])
        for attr, value in attributes_with_defaults.items():
            setattr(self, attr, value)

    def save(self, fname):
        with open(fname, "w") as f:
            json.dump(self.attr_dict, f)

    def load(self, fname):
        with open(fname, "r") as f:
            attr_dict = json.load(f)
        self.custom_initialization(attr_dict)

    def custom_initialization(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __str__(self):
        return f"{type(self).__name__} with {len(self.attrs)} attributes."

    def __repr__(self):
        kwargs = ", ".join([f"{k}={v}" for k, v in self.attr_dict.items()])
        return f"{type(self).__name__}.custom_initialization({kwargs})"


class HvsrPreProcessingSettings(Settings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "detrend": "linear",
            "orient_to_degrees_from_north": 0.,
            "window_length_in_seconds": 60,
            "filter_corner_frequencies_in_hz": (None, None)
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrProcessingSettings(Settings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "window_type_and_width": ("tukey", 0.1),
            "smoothing_type_and_bandwidth": ("konno_and_ohmachi", 40),
            "frequency_resampling_in_hz": np.geomspace(0.1, 50, 200),
            "fft_settings": None
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrTraditionalProcessingSettings(HvsrProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "processing_method": "traditional",
        }
        self.extend_attributes(attributes_with_defaults)

class HvsrTraditionalFrequencyDomainProcessingSettings(HvsrTraditionalProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "method_to_combine_horizontals": "geometric_mean",
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrTraditionalSingleAzimuthProcessingSettings(HvsrTraditionalProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "method_to_combine_horizontals": "single_azimuth",
            "azimuth_in_degrees": 20
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrTraditionalRotDnProcessingSettings(HvsrTraditionalProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "method_to_combine_horizontals": "rotdn",
            "nth_percentile_for_rotd_computation":50,
            "azimuths_in_degrees": np.arange(0, 180, 5)
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrAzimuthalProcessingSettings(HvsrProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "processing_method": "azimuthal",
            "azimuths_in_degrees": np.arange(0, 180, 5)
        }
        self.extend_attributes(attributes_with_defaults)


class HvsrDiffuseFieldProcessingSettings(HvsrProcessingSettings):

    def __init__(self):
        super().__iter__()
        attributes_with_defaults = {
            "processing_method": "diffuse_field",
        }
        self.extend_attributes(attributes_with_defaults)
