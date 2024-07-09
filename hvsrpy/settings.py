# This file is part of hvsrpy a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

__all__ = [
    "HvsrPreProcessingSettings",
    "PsdPreProcessingSettings",
    "PsdProcessingSettings",
    "HvsrTraditionalProcessingSettings",
    "HvsrTraditionalSingleAzimuthProcessingSettings",
    "HvsrTraditionalRotDppProcessingSettings",
    "HvsrAzimuthalProcessingSettings",
    "HvsrDiffuseFieldProcessingSettings",
]


class Settings(ABC):

    def __init__(self, hvsrpy_version=__version__):
        """Initialize abstract ``Settings`` object.

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

        def to_serializable(entry):
            try:
                entry = entry.tolist()
            except:
                pass
            return entry

        for name in self.attrs:
            attr = getattr(self, name)

            if isinstance(attr, dict):
                attr = {k: to_serializable(v) for k, v in attr.items()}
            else:
                attr = to_serializable(attr)

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

    def load(self, fname):
        """Load ``Settings`` object from file on disk.

        Parameters
        ----------
        fname : str
            Name of file from which the ``Settings`` object information
            will be loaded. Can be a relative or the full path.

        Returns
        -------
        None
            Updates ``Settings`` object from information stored in a
            file on disk.                

        """
        with open(fname, "r") as f:
            attr_dict = json.load(f)
        for key, value in attr_dict.items():
            setattr(self, key, value)

    def psummary(self):
        "Pretty summary of information in ``Settings`` object."
        for key, value in self.attr_dict.items():
            if isinstance(value, dict):
                print(f"{key: <40} :")
                for key, value in value.items():
                    if len(str(value)) > 40:
                        value = f"{str(value)[0:20]} ... {str(value)[-20:]}"
                    print(f"     {key: <35} : {value}")
            else:
                print(f"{key: <40} : {value}")

    def __eq__(self, other):
        if self.attr_dict == other.attr_dict:
            return True
        return False

    def __str__(self):
        """String representation of ``Settings`` object."""
        return f"{type(self).__name__} with {len(self.attrs)} attributes."

    def __repr__(self):
        """Unambiguous representation of ``Settings`` object."""
        kwargs = ", ".join([f"{k}={v}" for k, v in self.attr_dict.items()])
        return f"{type(self).__name__}({kwargs})"


class PreProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 ):
        """Base class for preprocessing.

        Notes
        -----
        .. note::
            The ``PreProcessingSettings`` class is a base class; it
            should not be instantiated directly. Instead use
            ``HvsrPreProcessingSettings`` or
            ``PsdPreProcessingSettings``.

        """
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["orient_to_degrees_from_north",
                           "filter_corner_frequencies_in_hz",
                           "window_length_in_seconds",
                           "detrend",
                           "ignore_dissimilar_time_step_warning",
                           ])
        self.orient_to_degrees_from_north = orient_to_degrees_from_north
        self.filter_corner_frequencies_in_hz = filter_corner_frequencies_in_hz
        self.window_length_in_seconds = window_length_in_seconds
        self.detrend = detrend
        self.ignore_dissimilar_time_step_warning = ignore_dissimilar_time_step_warning


class HvsrPreProcessingSettings(PreProcessingSettings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 preprocessing_method="hvsr",
                 ):
        """Initialize ``HvsrPreProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        orient_to_degrees_from_north : float, optional
            New sensor orientation in degrees from north
            (clockwise positive). The sensor's north component will be
            oriented such that it is aligned with the defined
            orientation.
        filter_corner_frequencies_in_hz : list of float or None, optional
            Butterworth filter's corner frequencies in Hz. ``None`` can
            be used to specify a one-sided filter. For example a high
            pass filter at 3 Hz would be specified as
            ``[3, None]``. Default is ``[None, None]`` indicating no
            filtering will be performed.
        window_length_in_seconds : float or None, optional
            Duration length of each split, default is ``60.`` indicating
            all records will be split into 60-second windows. Use
            ``None`` to skip splitting records into windows during
            pre-processing.
        detrend : {"linear", "constant", "none"}, optional
            Type of detrending. If ``detrend == "linear"`` (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If ``detrend == "constant"``, only the mean of data
            is subtracted. If ``detrend == "none"``, no detrend is
            performed. Detrend is done on a window-by-window basis.
        ignore_dissimilar_time_step_warning : bool, optional
            If ``True`` a warning will not be raised if records have
            different time steps, default is ``False`` (i.e., a warning
            will be raised).
        preprocessing_method : str, optional
            Defines pre-processing for later reference, default is
            ``'hvsr'``. Should not be changed.

        Returns
        -------
        HvsrPreProcessingSettings
            Object contains all user-defined settings to control
            preprocessing of microtremor or earthquake recordings
            in preparation for HVSR processing.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         orient_to_degrees_from_north=orient_to_degrees_from_north,
                         filter_corner_frequencies_in_hz=filter_corner_frequencies_in_hz,
                         window_length_in_seconds=window_length_in_seconds,
                         detrend=detrend,
                         ignore_dissimilar_time_step_warning=ignore_dissimilar_time_step_warning)
        self.attrs.extend(["preprocessing_method"])
        self.preprocessing_method = preprocessing_method


class PsdPreProcessingSettings(PreProcessingSettings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 window_type_and_width=["tukey", 0.1],
                 fft_settings=None,
                 instrument_transfer_function=None,
                 differentiate=False,
                 preprocessing_method="psd",
                 ):
        """Initialize ``PsdPreProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        orient_to_degrees_from_north : float, optional
            New sensor orientation in degrees from north
            (clockwise positive). The sensor's north component will be
            oriented such that it is aligned with the defined
            orientation.
        filter_corner_frequencies_in_hz : list of float or None, optional
            Butterworth filter's corner frequencies in Hz. ``None`` can
            be used to specify a one-sided filter. For example a high
            pass filter at 3 Hz would be specified as
            ``[3, None]``. Default is ``[None, None]`` indicating no
            filtering will be performed.
        window_length_in_seconds : float or None, optional
            Duration length of each split, default is ``60.`` indicating
            all records will be split into 60-second windows. Use
            ``None`` to skip splitting records into windows during
            pre-processing.
        detrend : {"linear", "constant", "none"}, optional
            Type of detrending. If ``detrend == "linear"`` (default), the
            result of a linear least-squares fit to data is subtracted
            from data. If ``detrend == "constant"``, only the mean of data
            is subtracted. If ``detrend == "none"``, no detrend is
            performed. Detrend is done on a window-by-window basis.
        ignore_dissimilar_time_step_warning : bool, optional
            If ``True`` a warning will not be raised if records have
            different time steps, default is ``False`` (i.e., a warning
            will be raised).
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft``, default is ``None``
            indicating ``hvsrpy`` defaults will be used.
        instrument_transfer_function : InstrumentTransferFunction, optional
            If the sensor's frequency response is provided it will be
            removed, default is ``None`` meaning no instrument
            correction is performed.
        differentiate : bool, optional
            If ``True`` the provided signal will be differentiated,
            default is ``False``.
        preprocessing_method : str, optional
            Defines pre-processing for later reference, default is
            ``'psd'``. Should not be changed.

        Returns
        -------
        PsdPreProcessingSettings
            Object contains all user-defined settings to control
            preprocessing of microtremor or earthquake recordings
            in preparation for PSD processing.

        """
        # Need window_type_and_width in PSD preprocessing for use
        # with differentiation and/or instrument_transfer_function.
        super().__init__(hvsrpy_version=hvsrpy_version,
                         orient_to_degrees_from_north=orient_to_degrees_from_north,
                         filter_corner_frequencies_in_hz=filter_corner_frequencies_in_hz,
                         window_length_in_seconds=window_length_in_seconds,
                         detrend=detrend,
                         ignore_dissimilar_time_step_warning=ignore_dissimilar_time_step_warning)
        self.attrs.extend(["window_type_and_width",
                           "fft_settings",
                           "instrument_transfer_function",
                           "differentiate",
                           "preprocessing_method",
                           ])
        self.window_type_and_width = window_type_and_width
        self.fft_settings = fft_settings
        self.instrument_transfer_function = instrument_transfer_function
        self.differentiate = differentiate
        self.preprocessing_method = preprocessing_method


class PsdProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="keeping_majority_time_step",
                 processing_method="psd",
                 ):
        """Initialize ``PsdProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft``, default is ``None``
            indicating ``hvsrpy`` defaults will be used.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"keeping_majority_time_step"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'psd'``. Should not be changed.

        Returns
        -------
        PsdProcessingSettings
            Object contains all user-defined settings to control
            PSD processing of microtremor or earthquake recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["window_type_and_width",
                           "smoothing",
                           "fft_settings",
                           "handle_dissimilar_time_steps_by",
                           "processing_method"
                           ])
        self.window_type_and_width = window_type_and_width
        self.fft_settings = fft_settings
        self.smoothing = dict(smoothing)
        self.handle_dissimilar_time_steps_by = handle_dissimilar_time_steps_by
        self.processing_method = processing_method


class HvsrProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 ):
        """Base class for HVSR processing settings.

        Notes
        -----
        .. note::
            The ``HvsrProcessingSettings`` class is a base class; it
            should not be instantiated directly. Instead use
            ``HvsrTraditionalProcessingSettings``,
            ``HvsrTraditionalSingleAzimuthProcessingSettings``,
            ``HvsrTraditionalRotDppProcessingSettings``,
            ``HvsrAzimuthalProcessingSettings``, or
            ``HvsrDiffuseFieldProcessingSettings``.

        """
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["window_type_and_width",
                           "smoothing",
                           "fft_settings",
                           "handle_dissimilar_time_steps_by",
                           ])
        self.window_type_and_width = window_type_and_width
        self.smoothing = dict(smoothing)
        self.fft_settings = fft_settings
        self.handle_dissimilar_time_steps_by = handle_dissimilar_time_steps_by


class HvsrTraditionalProcessingSettingsBase(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 fft_settings=None,
                 processing_method="traditional",
                 ):
        """Base class for traditional HVSR processing settings.

        Notes
        -----
        .. note::
            The ``HvsrTraditionalProcessingSettingsBase`` class is a
            base class; it should not be instantiated directly. Instead
            use
            ``HvsrTraditionalProcessingSettings``,
            ``HvsrTraditionalSingleAzimuthProcessingSettings``, or
            ``HvsrTraditionalRotDppProcessingSettings``.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings)
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method


class HvsrTraditionalProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="traditional",
                 method_to_combine_horizontals="geometric_mean",
                 ):
        """Initialize ``HvsrTraditionalProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft`` default is ``None``.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"frequency_domain_resampling"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'traditional'``. Should not be changed.
        method_to_combine_horizontals : str, optional
            Defines method for combining the two horizontal components
            options include: "arithmetic_mean", "squared_average",
            "quadratic_mean", "geometric_mean", "total_horizontal_energy",
            "vector_summation", and "maximum_horizontal_value", default
            is "geometric_mean".

        Returns
        -------
        HvsrTraditionalProcessingSettings
            Object contains all user-defined settings to control
            HVSR processing of microtremor or earthquake recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals"])
        self.method_to_combine_horizontals = method_to_combine_horizontals


class HvsrTraditionalSingleAzimuthProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 fft_settings=None,
                 processing_method="traditional",
                 method_to_combine_horizontals="single_azimuth",
                 azimuth_in_degrees=20.,
                 ):
        """Initialize ``HvsrTraditionalSingleAzimuthProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft`` default is ``None``.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"frequency_domain_resampling"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'traditional'``. Should not be changed.
        method_to_combine_horizontals : str, optional
            Defines method for combining the two horizontal components
            "single_azimuth". Do not change.
        azimuth_in_degrees : float, optional
            Azimuth at which to compute the single azimuth HVSR,
            measured from north in degrees (clockwise positive). Default
            is 20 degrees (i.e., 20 degrees to the east from north).

        Returns
        -------
        HvsrTraditionalSingleAzimuthProcessingSettings
            Object contains all user-defined settings to control
            HVSR processing of microtremor or earthquake recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals",
                           "azimuth_in_degrees",
                           ])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.azimuth_in_degrees = azimuth_in_degrees


class HvsrTraditionalRotDppProcessingSettings(HvsrTraditionalProcessingSettingsBase):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="traditional",
                 method_to_combine_horizontals="rotdpp",
                 ppth_percentile_for_rotdpp_computation=50.,
                 azimuths_in_degrees=np.arange(0, 180, 5)
                 ):
        """Initialize ``HvsrTraditionalRotDppProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft`` default is ``None``.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"frequency_domain_resampling"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'traditional'``. Should not be changed.
        method_to_combine_horizontals : str, optional
            Defines method for combining the two horizontal components
            "rotdpp". Do not change.
        ppth_percentile_for_rotdpp_computation : float, optional
            The frequency-by-frequency percentile to be selected
            from the HVSR curves computed, default is 50. which is
            consistent with RotD50 from ground motion processing.
        azimuths_in_degrees : iterable of float, optional
            Azimuths measured from north in degrees (clockwise positive)
            at which to compute single azimuth HVSRs, to then select
            the frequency-by-frequency ppth precentile.

        Returns
        -------
        HvsrTraditionalRotDppProcessingSettings
            Object contains all user-defined settings to control
            HVSR processing of microtremor or earthquake recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method,
                         )
        self.attrs.extend(["method_to_combine_horizontals",
                           "ppth_percentile_for_rotdpp_computation",
                           "azimuths_in_degrees"])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.ppth_percentile_for_rotdpp_computation = ppth_percentile_for_rotdpp_computation
        self.azimuths_in_degrees = np.array(azimuths_in_degrees)


class HvsrAzimuthalProcessingSettings(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="azimuthal",
                 azimuths_in_degrees=np.arange(0, 180, 5)):
        """Initialize ``HvsrAzimuthalProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft`` default is ``None``.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"frequency_domain_resampling"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'azimuthal'``. Should not be changed.
        azimuths_in_degrees : iterable of float, optional
            Azimuths measured from north in degrees (clockwise positive)
            at which to compute single azimuth HVSRs.

        Returns
        -------
        HvsrAzimuthalProcessingSettings
            Object contains all user-defined settings to control
            azimuthal HVSR processing of microtremor or earthquake
            recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         fft_settings=fft_settings,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         )
        self.attrs.extend(["processing_method",
                           "azimuths_in_degrees"])
        self.processing_method = processing_method
        self.azimuths_in_degrees = azimuths_in_degrees


class HvsrDiffuseFieldProcessingSettings(HvsrProcessingSettings):

    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=np.geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="keeping_majority_time_step",
                 processing_method="diffuse_field"):
        """Initialize ``HvsrDiffuseFieldProcessingSettings`` object.

        Parameters
        ----------
        hvsrpy_version : str
            Denotes the version of ``hvsrpy`` used to create the
            ``Settings`` object. Should not be changed.
        window_type_and_width : list, optional
            A list with entries like ``["tukey", 0.1]`` that control the
            window type and width, respectively.
        smoothing : dict, optional
            Smoothing information like ``dict(operator="konno_and_ohmachi",
            bandwidth=40, center_frequencies_in_hz=np.geomspace(0.1, 50, 200))``.
        fft_settings : dict or None, optional
            Custom settings for ``np.fft.rfft`` default is ``None``.
        handle_dissimilar_time_steps_by : {"frequency_domain_resampling", "keeping_smallest_time_step", "keeping_majority_time_step"}, optional
            Method to resolve multiple records with a different
            time step, default is ``"frequency_domain_resampling"``.
        processing_method : str, optional
            Defines processing_method for later reference, default is
            ``'diffuse_field'``. Should not be changed.

        Returns
        -------
        HvsrDiffuseFieldProcessingSettings
            Object contains all user-defined settings to control
            diffuse-field HVSR processing of microtremor or earthquake
            recordings.

        """
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         fft_settings=fft_settings,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         )
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method
