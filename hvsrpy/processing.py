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

from .smoothing import SMOOTHING_OPERATORS
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .hvsr_diffuse_field import HvsrDiffuseField
from .timeseries import TimeSeries
from .settings import HvsrTraditionalSingleAzimuthProcessingSettings


def preprocess(records, settings):
    """Preprocess time domain data before performing HVSR calculations.

    records: iterable of SeismicRecording3C
        Time-domain data in the form of an iterable object containing
        ``SeismicRecording3C`` objects. This is the data that will be
        preprocessed.
    settings : HvsrPreProcessingSettings
        ``HvsrPreProcessingSettings`` object that controls how the
        time-domain data will be preprocessed.

    Returns
    -------
    List of SeismicRecording3C
        Seismic records that have been preprocessed and are ready for
        HVSR processing. 

    """
    preprocessed_records = []

    ex_dt = records[0].vt.dt_in_seconds
    for idx, srecord3c in enumerate(records):

        # check all records have some dt; required later for fft.
        if np.abs(srecord3c.vt.dt_in_seconds - ex_dt) > 10E-6: #pragma: no cover
            msg = f"The dt_in_seconds of all records must be equal, "
            msg += f"dt_in_seconds of record {idx} is "
            msg += f"{srecord3c.vt.dt_in_seconds} which does not match "
            msg += f"dt_in_seconds of record 0 of {ex_dt}."
            raise ValueError(msg)

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

        for window in windows:

            # detrend each time window individually.
            if settings.detrend is not None:
                window.detrend(type=settings.detrend)

        preprocessed_records.extend(windows)

    return preprocessed_records


def arithmetic_mean(ns, ew, settings=None):
    """Computes the arithmetic mean of two vectors."""
    return (ns+ew) / 2


def squared_average(ns, ew, settings=None):
    """Computes the squared average (aka quadratic mean) of two vectors."""
    return np.sqrt((ns*ns + ew*ew)/2)


def geometric_mean(ns, ew, settings=None):
    """Computes the geometric mean of two vectors."""
    return np.sqrt(ns * ew)


def total_horizontal_energy(ns, ew, settings=None):
    """Computes the magnitude of sum of two orthoginal vectors."""
    return np.sqrt((ns*ns + ew*ew))


def maximum_horizontal_value(ns, ew, settings=None):
    """Computes the entry-by-entry maximum of the vectors provided."""
    return np.where(ns > ew, ns, ew)


def single_azimuth(ns, ew, degrees_from_north):
    """Computes the magnitude of two vectors along a specifc azimuth."""
    radians_from_north = np.radians(degrees_from_north)
    return ns*np.cos(radians_from_north) + ew*np.sin(radians_from_north)


COMBINE_HORIZONTAL_REGISTER = {
    "arithmetic_mean": arithmetic_mean,
    "squared_average": squared_average,
    "quadratic_mean": squared_average,
    "geometric_mean": geometric_mean,
    "total_horizontal_energy": total_horizontal_energy,
    "vector_summation": total_horizontal_energy,
    "maximum_horizontal_value": maximum_horizontal_value,
}


def rfft(amplitude, **kwargs):
    n = kwargs.get("n", len(amplitude))
    rfft = np.fft.rfft(amplitude, **kwargs)
    rfft *= 2/min(len(amplitude), n)
    return rfft


def nextpow2(n, minimum_power_of_two=2**15): # 2**15 = 32768
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two *= 2

def prepare_fft_setttings(records, settings):
    # To accelerate smoothing, need consistent value of n.
    # Will round to nearest power of 2 to accelerate FFT.
    max_n_samples = 0
    for record in records:
        if record.vt.n_samples > max_n_samples:
            max_n_samples = record.vt.n_samples
    good_n = nextpow2(max_n_samples)

    if settings.fft_settings is None:
        settings.fft_settings = dict(n=good_n)
    else:
        user_n = settings.fft_settings.get("n", max_n_samples)
        settings.fft_settings["n"] = good_n if good_n > user_n else user_n


def traditional_hvsr_processing(records, settings):
    prepare_fft_setttings(records, settings)

    h_idx = 0
    v_idx = len(records)
    fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], records[0].ns.dt_in_seconds)
    spectra = np.empty((len(records)*2, len(fft_frq)))
    for record in records:
        # window time series to mitigate frequency-domain artifacts.
        record.window(*settings.window_type_and_width)

        # compute fft of horizontals and combine.
        fft_ns = np.abs(rfft(record.ns.amplitude, **settings.fft_settings))
        fft_ew = np.abs(rfft(record.ew.amplitude, **settings.fft_settings))
        method = COMBINE_HORIZONTAL_REGISTER[settings.method_to_combine_horizontals]
        h = method(fft_ns, fft_ew, settings)

        # compute fft of vertical.
        v = np.abs(rfft(record.vt.amplitude, **settings.fft_settings))

        # store
        spectra[h_idx] = h
        spectra[v_idx] = v

        # increment
        h_idx += 1
        v_idx += 1

    # smooth.
    smoothing_operator, bandwidth = settings.smoothing_operator_and_bandwidth
    frq = settings.frequency_resampling_in_hz
    smooth_spectra = SMOOTHING_OPERATORS[smoothing_operator](fft_frq, spectra, frq, bandwidth)

    # compute hvsr
    hvsr_spectra = smooth_spectra[:len(records)] / smooth_spectra[len(records):]

    return HvsrTraditional(frq, hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


def traditional_single_azimuth_hvsr_processing(records, settings):
    prepare_fft_setttings(records, settings)

    h_idx = 0
    v_idx = len(records)
    fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], records[0].ns.dt_in_seconds)
    spectra = np.empty((len(records)*2, len(fft_frq)))
    for record in records:
        # combine horizontal components in the time domain.
        h = single_azimuth(record.ns.amplitude,
                           record.ew.amplitude,
                           settings.azimuth_in_degrees)
        h = TimeSeries(h, record.ns.dt_in_seconds)

        # window time series to mitigate frequency-domain artifacts.
        h.window(*settings.window_type_and_width)
        v = record.vt
        v.window(*settings.window_type_and_width)

        # compute fourier transform.
        spectra[h_idx] = np.abs(rfft(h.amplitude, **settings.fft_settings))
        spectra[v_idx] = np.abs(rfft(v.amplitude, **settings.fft_settings))

        # increment.
        h_idx += 1
        v_idx += 1

    # smooth all at once to boost performance.
    smoothing_operator, bandwidth = settings.smoothing_operator_and_bandwidth
    frq = settings.frequency_resampling_in_hz
    smooth_spectra = SMOOTHING_OPERATORS[smoothing_operator](fft_frq, spectra, frq, bandwidth)

    # compute hvsr
    hvsr_spectra = smooth_spectra[:len(records)] / smooth_spectra[len(records):]

    return HvsrTraditional(frq, hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


def traditional_rotdpp_hvsr_processing(records, settings):
    prepare_fft_setttings(records, settings)

    frq = settings.frequency_resampling_in_hz
    fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], records[0].vt.dt_in_seconds)
    spectra_per_record = np.empty((len(settings.azimuths_in_degrees)+1, len(fft_frq)))
    rotdpp_hvsr_spectra = []
    for record in records:
        # prepare vertical component only once per record.
        v = record.vt
        v.window(*settings.window_type_and_width)
        fft_v = np.abs(rfft(v.amplitude, **settings.fft_settings))
        spectra_per_record[-1] = fft_v

        # rotate horizontals through defined azimuths.
        for idx, azimuth in enumerate(settings.azimuths_in_degrees):
            h = single_azimuth(record.ns.amplitude,
                               record.ew.amplitude,
                               azimuth)
            h = TimeSeries(h, dt_in_seconds=record.ns.dt_in_seconds)
            h.window(*settings.window_type_and_width)
            fft_h = np.abs(rfft(h.amplitude, **settings.fft_settings))
            spectra_per_record[idx] = fft_h

        # smooth.
        smoothing_operator, bandwidth = settings.smoothing_operator_and_bandwidth
        frq = settings.frequency_resampling_in_hz
        smooth_spectra = SMOOTHING_OPERATORS[smoothing_operator](fft_frq, spectra_per_record, frq, bandwidth)

        smooth_h = np.percentile(smooth_spectra[:-1],
                                 settings.ppth_percentile_for_rotdpp_computation,
                                 axis=0)
        smooth_v = smooth_spectra[-1]
        rotdpp_hvsr_spectra.append(smooth_h / smooth_v)

    return HvsrTraditional(frq, rotdpp_hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


TRADITIONAL_PROCESSING_REGISTER = {
    "single_azimuth": traditional_single_azimuth_hvsr_processing,
    "directional_energy": traditional_single_azimuth_hvsr_processing,
    "rotdpp": traditional_rotdpp_hvsr_processing,
    "arithmetic_mean": traditional_hvsr_processing,
    "squared_average": traditional_hvsr_processing,
    "quadratic_mean": traditional_hvsr_processing,
    "geometric_mean": traditional_hvsr_processing,
    "total_horizontal_energy": traditional_hvsr_processing,
    "vector_summation": traditional_hvsr_processing,
    "maximum_horizontal_value": traditional_hvsr_processing,
}


def traditional_hvsr_processing_base(records, settings):
    # need to handle this seperately b/c time domain technique
    method = TRADITIONAL_PROCESSING_REGISTER[settings.method_to_combine_horizontals]
    return method(records, settings)


def azimuthal_hvsr_processing(records, settings):
    prepare_fft_setttings(records, settings)
    single_azimuth_settings = HvsrTraditionalSingleAzimuthProcessingSettings()
    hvsr_per_azimuth = []
    for azimuth in settings.azimuths_in_degrees:
        single_azimuth_settings.azimuth_in_degrees = azimuth
        hvsr = traditional_single_azimuth_hvsr_processing(records,
                                                          single_azimuth_settings)
        hvsr_per_azimuth.append(hvsr)
    return HvsrAzimuthal(hvsr_per_azimuth, settings.azimuths_in_degrees, meta={**records[0].meta, **settings.attr_dict})


def diffuse_field_hvsr_processing(records, settings):
    prepare_fft_setttings(records, settings)

    fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], records[0].vt.dt_in_seconds)
    psd_ns = np.zeros(len(fft_frq))
    psd_ew = np.zeros(len(fft_frq))
    psd_vt = np.zeros(len(fft_frq))
    df = (1/records[0].ns.dt_in_seconds) / settings.fft_settings["n"]
    for record in records:
        # window time series to mitigate frequency-domain artifacts.
        record.window(*settings.window_type_and_width)

        # TODO(jpv): Double check PSD calculation.

        # compute fourier transform.
        fft_ns = np.abs(rfft(record.ns.amplitude, **settings.fft_settings))
        fft_ew = np.abs(rfft(record.ew.amplitude, **settings.fft_settings))
        fft_vt = np.abs(rfft(record.vt.amplitude, **settings.fft_settings))

        # compute psd with appropriate normalization for completeness;
        # note normalization is not technically needed for this application.
        psd_ns += (fft_ns * fft_ns) / (2*df)
        psd_ew += (fft_ew * fft_ew) / (2*df)
        psd_vt += (fft_vt * fft_vt) / (2*df)

    # compute average psd over all records (i.e., windows).
    psd_ns /= len(records)
    psd_ew /= len(records)
    psd_vt /= len(records)

    # smooth.
    smoothing_operator, bandwidth = settings.smoothing_operator_and_bandwidth
    frq = settings.frequency_resampling_in_hz

    spectra = np.array([psd_ns + psd_ew, psd_vt])
    smooth_spectra = SMOOTHING_OPERATORS[smoothing_operator](fft_frq, spectra, frq, bandwidth)
    hor = smooth_spectra[0]
    ver = smooth_spectra[1]

    # compute hvsr
    return HvsrDiffuseField(frq, np.sqrt((hor)/ver), meta={**records[0].meta, **settings.attr_dict})


PROCESSING_METHODS = {
    "traditional": traditional_hvsr_processing_base,
    "azimuthal": azimuthal_hvsr_processing,
    "diffuse_field": diffuse_field_hvsr_processing
}


def process(records, settings):
    """Process time domain domain data.

    records: iterable of SeismicRecording3C
        Time-domain data in the form of iterable object containing
        SeismicRecording3C objects. This is the data that will be
        processed.
    settings : HvsrProcessingSettings
        HvsrProcessingSettings object that controls how the
        time-domain data will be processed.

    Returns
    -------
    HvsrTraditional, HvsrAzimuthal, HvsrDiffuseField
        Instantiated HVSR object according to the processing settings
        selected.

    """
    return PROCESSING_METHODS[settings.processing_method](records, settings)
