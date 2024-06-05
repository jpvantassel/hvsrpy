# This file is part of hvsrpy, a Python package for horizontal-to-vertical
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

import warnings

import numpy as np

from .seismic_recording_3c import SeismicRecording3C
from .instrument_response import _remove_instrument_response, _differentiate
from .processing import prepare_fft_settings


def hvsr_preprocess(records, settings):

    preprocessed_records = []

    if isinstance(records, SeismicRecording3C):
        records = [records]

    ex_dt = records[0].vt.dt_in_seconds
    for idx, srecord3c in enumerate(records):

        # check if all records have same dt.
        if np.abs(srecord3c.vt.dt_in_seconds - ex_dt) > 1E-5 and not settings.ignore_dissimilar_time_step_warning:  # pragma: no cover
            msg = f"The dt_in_seconds of all records are not equal, "
            msg += f"dt_in_seconds of record {idx} is "
            msg += f"{srecord3c.vt.dt_in_seconds} which does not match "
            msg += f"dt_in_seconds of record 0 of {ex_dt}."
            warnings.warn(msg)

        # orient receiver to north.
        if settings.orient_to_degrees_from_north is not None:
            srecord3c.orient_sensor_to(settings.orient_to_degrees_from_north)

        # time-domain filter raw signal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        # divide raw signal into time windows.
        if settings.window_length_in_seconds is not None:
            windows = srecord3c.split(settings.window_length_in_seconds)
        else:
            windows = [srecord3c]

        # detrend each time window individually.
        if (settings.detrend is not None) and (settings.detrend != "none"):
            for window in windows:
                window.detrend(type=settings.detrend)

        preprocessed_records.extend(windows)

    return preprocessed_records

def psd_preprocess(records, settings):

    prepare_fft_settings(records, settings)

    preprocessed_records = []

    if isinstance(records, SeismicRecording3C):
        records = [records]

    ex_dt = records[0].vt.dt_in_seconds
    for idx, srecord3c in enumerate(records):

        # check if all records have same dt.
        if np.abs(srecord3c.vt.dt_in_seconds - ex_dt) > 1E-5 and not settings.ignore_dissimilar_time_step_warning:  # pragma: no cover
            msg = f"The dt_in_seconds of all records are not equal, "
            msg += f"dt_in_seconds of record {idx} is "
            msg += f"{srecord3c.vt.dt_in_seconds} which does not match "
            msg += f"dt_in_seconds of record 0 of {ex_dt}."
            warnings.warn(msg)

        # orient receiver to north.
        if settings.orient_to_degrees_from_north is not None:
            srecord3c.orient_sensor_to(settings.orient_to_degrees_from_north)

        # time-domain filter raw signal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        # window full signal
        if settings.instrument_transfer_function is not None or settings.differentiate:
            srecord3c.detrend(type="constant")
            srecord3c.window(*settings.window_type_and_width)

        # remove instrument response.
        if settings.instrument_transfer_function is not None:
            for component in ["ns", "ew", "vt"]:
                new_tseries = _remove_instrument_response(getattr(srecord3c, component),
                                                          settings.instrument_transfer_function,
                                                          settings.fft_settings)
                setattr(srecord3c, component, new_tseries)

            # repeat filter after removing instrument response.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        # differentiate raw signal.
        if settings.differentiate:
            for component in ["ns", "ew", "vt"]:
                new_tseries = _differentiate(getattr(srecord3c, component),
                                             settings.fft_settings)
                setattr(srecord3c, component, new_tseries)

        # divide raw signal into time windows.
        if settings.window_length_in_seconds is not None:
            windows = srecord3c.split(settings.window_length_in_seconds)
        else:
            windows = [srecord3c]

        # detrend each time window individually.
        if (settings.detrend is not None) and (settings.detrend != "none"):
            for window in windows:
                window.detrend(type=settings.detrend)

        preprocessed_records.extend(windows)

    return preprocessed_records

PREPROCESSING_METHODS = {
    "hvsr": hvsr_preprocess,
    "psd": psd_preprocess,
}


def preprocess(records, settings):
    """Preprocess time domain data before performing processing.

    records: SeismicRecording3C or iterable of SeismicRecording3C
        Time-domain data as an ``SeismicRecording3C`` object or iterable
        object containing ``SeismicRecording3C`` objects. This is the
        data that will be preprocessed.
    settings : PreProcessingSettings
        ``PreProcessingSettings`` object that controls how the
        time-domain data will be preprocessed.

    Returns
    -------
    List of SeismicRecording3C
        Seismic records that have been preprocessed and are ready for
        processing. 

    """
    return PREPROCESSING_METHODS[settings.preprocessing_method](records, settings)

