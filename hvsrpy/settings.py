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

"""Definition of Settings class and its descendants."""

from abc import ABC

import json

import numpy as np

from .metadata import __version__


class Settings(ABC):

    def __init__(self, hvsrpy_version=__version__):
        """Initialize abstract ``Settings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object.

        Notes
        -----
        .. note::
            The ``Settings`` class is an abstract base class; it cannot
            be instantiated directly.

        """
        self.attrs = ["hvsrpy_version"]
        self.hvsrpy_version = hvsrpy_version

    @property
    def attr_dict(self):
        """Dictionary of ``Settings`` object's attributes."""
        attr_dict = {}
        for name in self.attrs:
            attr = getattr(self, name)
            try:
                attr = attr.tolist()
            except:
                pass
            attr_dict[name] = attr
        return attr_dict

    def save(self, fname):
        """Save ``Settings`` object to file on disk.

        Parameters
        ----------
        fname : str
            Name of file to which the ``Settings`` object will be saved.
            Can be a relative or the full path.

        Returns
        -------
        None
            Instead saves ``Settings`` object information to file on
            disk.                

        """
        with open(fname, "w") as f:
            json.dump(self.attr_dict, f)

    @classmethod
    def load(cls, fname):
        """Load ``Settings`` object from file on disk.

        Parameters
        ----------
        fname : str
            Name of file from which the ``Settings`` object information
            will be loaded. Can be a relative or the full path.

        Returns
        -------
        Settings
            Creates ``Settings`` object from information stored in a
            file on disk.                

        """
        with open(fname, "r") as f:
            attr_dict = json.load(f)
        return cls(**attr_dict)

    def __str__(self):
        """String representation of ``Settings`` object."""
        return f"{type(self).__name__} with {len(self.attrs)} attributes."

    def __repr__(self):
        """Unambiguous representation of ``Settings`` object."""
        kwargs = ", ".join([f"{k}={v}" for k, v in self.attr_dict.items()])
        return f"{type(self).__name__}({kwargs})"


class HvsrPreProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=(None, None),
                 window_length_in_seconds=60.,
                 detrend="linear"):

        """Initialize ``HvsrPreProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object.
        orient_to_degrees_from_north : float, optional
            New sensor orientation in degrees from north
            (clockwise positive). The sensor's north component will be
            oriented such that it is aligned with the defined
            orientation.
        filter_corner_frequencies_in_hz : tuple of float or None, optional
            Butterworth filter's corner frequencies in Hz. ``None`` can
            be used to specify a one-sided filter. For example a high
            pass filter at 3 Hz would be specified as
            ``(3, None)``. Default is ``(None, None)`` indicating no
            filtering will be performed.
        window_length_in_seconds : float or None, optional
            Duration length of each split, default is ``60.`` indicating
            all records will be split into 60-second windows. Use
            ``None`` to skip splitting records into windows during
            pre-processing.
        detrend : {"constant", "linear"}, optional
            Type of detrending. If ``type == "linear"`` (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If ``type == "constant"``, only the mean of data
            is subtracted.

        Returns
        -------
        HvsrPreProcessingSettings
            Object contains all user-defined settings to control
            preprocessing of microtremor or earthquake recordings
            in preparation for HVSR processing.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         )
        self.attrs.extend(["orient_to_degrees_from_north",
                           "filter_corner_frequencies_in_hz",
                           "window_length_in_seconds",
                           "detrend"])
        self.orient_to_degrees_from_north = orient_to_degrees_from_north
        self.filter_corner_frequencies_in_hz = filter_corner_frequencies_in_hz
        self.window_length_in_seconds = window_length_in_seconds
        self.detrend = detrend


# TODO(jpv): Finish documenting settings module.
class HvsrProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None):
        """Initialize ``HvsrProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object.
        window_type_and_width : tuple, optional
            A tuple with entries like 
            ("tukey", 0.1),
        smoothing_operator_and_bandwidth : tuple, optional
            ("konno_and_ohmachi", 40,),
        frequency_resampling_in_hz : ndarray, optional
            np.geomspace(0.1, 50, 200),
        fft_settings : dict or None, optional
            fft_settings=None):

        """
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["window_type_and_width",
                           "smoothing_operator_and_bandwidth",
                           "frequency_resampling_in_hz",
                           "fft_settings"])
        self.window_type_and_width = window_type_and_width
        self.smoothing_operator_and_bandwidth = smoothing_operator_and_bandwidth
        self.frequency_resampling_in_hz = np.array(frequency_resampling_in_hz)
        self.fft_settings = fft_settings


class HvsrTraditionalProcessingSettingsBase(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="traditional"):

        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings)
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method


class HvsrTraditionalProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="traditional",
                 method_to_combine_horizontals="geometric_mean"):

        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals"])
        self.method_to_combine_horizontals = method_to_combine_horizontals


class HvsrTraditionalSingleAzimuthProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="traditional",
                 method_to_combine_horizontals="single_azimuth",
                 azimuth_in_degrees=20.):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals", "azimuth_in_degrees"])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.azimuth_in_degrees = azimuth_in_degrees


class HvsrTraditionalRotDppProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="traditional",
                 method_to_combine_horizontals="rotdpp",
                 ppth_percentile_for_rotdpp_computation=50.,
                 azimuths_in_degrees=np.arange(0, 180, 5)
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals",
                           "ppth_percentile_for_rotdpp_computation",
                           "azimuths_in_degrees"])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.ppth_percentile_for_rotdpp_computation = ppth_percentile_for_rotdpp_computation
        self.azimuths_in_degrees = np.array(azimuths_in_degrees)


class HvsrAzimuthalProcessingSettings(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="azimuthal",
                 azimuths_in_degrees=np.arange(0, 180, 5)):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings)
        self.attrs.extend(["processing_method",
                           "azimuths_in_degrees"])
        self.processing_method = processing_method
        self.azimuths_in_degrees = azimuths_in_degrees


class HvsrDiffuseFieldProcessingSettings(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=("tukey", 0.1),
                 smoothing_operator_and_bandwidth=("konno_and_ohmachi", 40,),
                 frequency_resampling_in_hz=np.geomspace(0.1, 50, 200),
                 fft_settings=None,
                 processing_method="diffuse_field"):

        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing_operator_and_bandwidth=smoothing_operator_and_bandwidth,
                         frequency_resampling_in_hz=frequency_resampling_in_hz,
                         fft_settings=fft_settings)
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method
