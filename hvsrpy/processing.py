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

import numpy as np

from .smoothing import konno_ohmachi_1d
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .hvsr_diffuse_field import HvsrDiffuseField
from .timeseries import TimeSeries


def preprocess(records, settings):
    """Preprocess time domain data before performing HVSR calculations.

    records: iterable of SeismicRecording3C
        Time-domain data in the form of an interable object containing
        SeismicRecording3C objects. This is the data that will be
        preprocessed.
    settings : HvsrPreProcessingSettings
        HvsrPreProcessingSettings object that controls how the
        time-domain data will be preprocessed.

    Returns
    -------
    List of SeismicRecording3C
        Seismic records that have been preprocessed and are ready for
        HVSR processing. 

    """
    preprocessed_records = []

    for srecord3c in records:

        # orient receiver to north.
        if settings.orient_to_degrees_from_north is not None:
            srecord3c.orient_sensor_to(settings.orient_to_degrees_from_north)

        # bandpass filter raw signal.
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


def traditional_single_azimuth_hvsr_processing(records, settings):
    hvsr_spectra = []
    for record in records:
        # combine horizontal components in the time domain.
        h = single_azimuth(record.ns.amplitude,
                           record.ew.amplitude,
                           settings.azimuth_in_degrees)
        h = TimeSeries(h, record.ns.dt)

        # window time series to mitigate frequency-domain artifacts.
        h.window(*settings.window_type_and_width)
        v = record.vt
        v.window(*settings.window_type_and_width)

        # compute fourier transform.
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=h.nsamples)
        fft_h = np.abs(rfft(h.amplitude, **settings.fft_settings))
        fft_v = np.abs(rfft(v.amplitude, **settings.fft_settings))
        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], h.dt)

        # smoothing
        frq = settings.frequency_resampling_in_hz
        smooth_v = konno_ohmachi_1d(fft_frq, fft_v, frq)
        smooth_h = konno_ohmachi_1d(fft_frq, fft_h, frq)

        # compute hvsr
        hvsr_spectra.append(smooth_h / smooth_v)

    return HvsrTraditional(hvsr_spectra, frq)
    # TODO(jpv): Add metadata HvsrTraditional.


def traditional_rotdpp_hvsr_processing(records, settings):
    frq = settings.frequency_resampling_in_hz
    rotdn_hvsr_spectra = []
    for record in records:
        # prepare vertical component only once per record.
        v = record.vt
        v.window(*settings.window_type_and_width)
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=v.nsamples)
        fft_v = np.abs(rfft(v.amplitude, **settings.fft_settings))
        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], record.ns.dt)
        smooth_v = konno_ohmachi_1d(fft_frq, fft_v, frq)

        # rotate horizontals through defined azimuths.
        rotated_hvsr_spectra = np.empty((len(settings.azimuths_in_degrees), len(frq)))
        for idx, azimuth in enumerate(settings.azimuths_in_degrees):
            h = single_azimuth(record.ns.amplitude,
                               record.ew.amplitude,
                               azimuth)
            h = TimeSeries(h, dt=record.ns.dt)
            h.window(*settings.window_type_and_width)
            fft_h = np.abs(rfft(h.amplitude, **settings.fft_settings))
            smooth_h = konno_ohmachi_1d(fft_frq, fft_h, frq)
            rotated_hvsr_spectra[idx] = smooth_h

        smooth_h = np.percentile(rotated_hvsr_spectra,
                                 settings.ppth_percentile_for_rotdpp_computation,
                                 axis=0)
        rotdn_hvsr_spectra.append(smooth_h / smooth_v)

    return HvsrTraditional(rotdn_hvsr_spectra, frq)


def traditional_hvsr_processing(records, settings):
    hvsr_spectra = []
    for record in records:
        # window time series to mitigate frequency-domain artifacts.
        record.window(*settings.window_type_and_width)

        # compute fourier transform.
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=record.ns.nsamples)
        fft_ns = np.abs(rfft(record.ns.amplitude, **settings.fft_settings))
        fft_ew = np.abs(rfft(record.ew.amplitude, **settings.fft_settings))
        fft_vt = np.abs(rfft(record.vt.amplitude, **settings.fft_settings))
        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], record.ns.dt)

        # combine horizontals
        method = COMBINE_HORIZONTAL_REGISTER[settings.method_to_combine_horizontals]
        h = method(fft_ns, fft_ew, settings)

        # smoothing
        frq = settings.frequency_resampling_in_hz
        smooth_v = konno_ohmachi_1d(fft_frq, fft_vt, frq)
        smooth_h = konno_ohmachi_1d(fft_frq, h, frq)

        # compute hvsr
        hvsr_spectra.append(smooth_h / smooth_v)

    return HvsrTraditional(hvsr_spectra, frq)
    # TODO(jpv): Add metadata HvsrTraditional.


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
    # allocate memory for vertical spectra.
    frq = settings.frequency_resampling_in_hz
    v_spectra = np.empty((len(records), len(frq)))

    # prepare verticals once at start.
    for idx, record in enumerate(records):
        v = record.vt
        v.window(*settings.window_type_and_width)
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=record.ns.nsamples)
        fft_v = np.abs(rfft(v.amplitude, **settings.fft_settings))
        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], record.ns.dt)
        smooth_v = konno_ohmachi_1d(fft_frq, fft_v, frq)
        v_spectra[idx] = smooth_v

    # rotate horizontals through defined azimuths.
    hvsr_per_azimuth = []
    for azimuth in settings.azimuths_in_degrees:
        hvsr_spectra = np.empty((len(records), len(frq)))
        for idx, (record, smooth_v) in enumerate(zip(records, v_spectra)):
            h = single_azimuth(record.ns.amplitude,
                               record.ew.amplitude,
                               azimuth)
            h = TimeSeries(h, dt=record.ns.dt)
            h.window(*settings.window_type_and_width)
            fft_h = np.abs(rfft(h.amplitude, **settings.fft_settings))
            smooth_h = konno_ohmachi_1d(fft_frq, fft_h, frq)
            hvsr_spectra[idx] = smooth_h / smooth_v
        hvsr_per_azimuth.append(HvsrTraditional(hvsr_spectra, frq))

    return HvsrAzimuthal.from_iter(hvsr_per_azimuth, settings.azimuths_in_degrees)
    # TODO(jpv): Add metadata HvsrAzimuthal and associated HvsrTraditional.


def diffuse_field_hvsr_processing(records, settings):
    # records must be of the same length for psd stacking.
    _nsamples = records[0].ns.nsamples
    for idx, record in enumerate(records):
        if _nsamples != record.ns.nsamples:
            msg = f"The number of samples in records[0] of {_nsamples}"
            msg += f" does not match the number of samples in records[{idx}]"
            msg += f" of {record.ns.nsamples}; number of samples must be"
            msg += " equal for processing under the diffuse field assumption."
            raise IndexError(msg)

    nfrqs = _nsamples//2 if (_nsamples % 2) == 0 else _nsamples//2 + 1
    psd_ns = np.zeros(nfrqs)
    psd_ew = np.zeros(nfrqs)
    psd_vt = np.zeros(nfrqs)
    df = (1/records[0].ns.dt) / _nsamples
    for record in records:
        # window time series to mitigate frequency-domain artifacts.
        record.window(*settings.window_type_and_width)

        # compute fourier transform.
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=record.ns.nsamples)
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

    # compute hvsr
    fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], record.ns.dt)
    return HvsrDiffuseField(np.sqrt((psd_ns + psd_ew)/psd_vt), fft_frq)


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
    Hvsr
        # TODO(jpv): Finish docstring.

    """
    return PROCESSING_METHODS[settings.processing_method](records, settings)
