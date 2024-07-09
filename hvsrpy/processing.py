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
from numpy.fft import rfft

from .smoothing import SMOOTHING_OPERATORS
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .hvsr_diffuse_field import HvsrDiffuseField
from .timeseries import TimeSeries
from .settings import HvsrTraditionalSingleAzimuthProcessingSettings
from .psd import Psd


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
    "root_mean_square": squared_average,
    "effective_amplitude_spectrum" : squared_average,
    "geometric_mean": geometric_mean,
    "total_horizontal_energy": total_horizontal_energy,
    "vector_summation": total_horizontal_energy,
    "maximum_horizontal_value": maximum_horizontal_value,
}


def nextpow2(n, minimum_power_of_two=2**15):  # 2**15 = 32768
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two *= 2


def prepare_fft_settings(records, settings):
    # to accelerate smoothing, need consistent value of n.
    # will round to nearest power of 2 to accelerate FFT.
    max_n_samples = 0
    for record in records:
        if record.vt.n_samples > max_n_samples:
            max_n_samples = record.vt.n_samples
    good_n = nextpow2(max_n_samples)

    if settings.fft_settings is None:
        settings.fft_settings = dict(n=good_n)
    else:
        user_n = settings.fft_settings.get("n", max_n_samples)
        if user_n is None:
            settings.fft_settings["n"] = max_n_samples
        else:
            settings.fft_settings["n"] = good_n if good_n > user_n else user_n


def prepare_records_with_inconsistent_dt(records, settings):
    # identify dt of records provided and track count.
    dt_with_count = dict()
    for record in records:
        _dt = record.ns.dt_in_seconds
        try:
            dt_with_count[_dt] += 1
        except KeyError:
            dt_with_count[_dt] = 1

    if settings.handle_dissimilar_time_steps_by == "frequency_domain_resampling":
        return records, dt_with_count

    elif settings.handle_dissimilar_time_steps_by == "keeping_smallest_time_step":
        smallest_dt = min(dt_with_count.keys())
        abbr_records = []
        count = 0
        for record in records:
            if record.ns.dt_in_seconds == smallest_dt:
                abbr_records.append(record)
                count += 1
                if count == dt_with_count[smallest_dt]:
                    break

        if (len(records) - len(abbr_records)) > 0:
            msg = "Keeping the smallest time step resulting in the removal of "
            msg += f"{len(records) - len(abbr_records)} of {len(records)} records. "
            msg += f"{len(abbr_records)} records remain."
            warnings.warn(msg)

        return abbr_records, {smallest_dt: count}

    elif settings.handle_dissimilar_time_steps_by == "keeping_majority_time_step":
        majority_count = 0
        for potential_dt, potential_count in dt_with_count.items():
            if potential_count > majority_count:
                majority_dt = potential_dt
                majority_count = potential_count

        abbr_records = []
        count = 0
        for record in records:
            if record.ns.dt_in_seconds == majority_dt:
                abbr_records.append(record)
                count += 1
                if count == majority_count:
                    break

        if (len(records) - len(abbr_records)) > 0:
            msg = "Keeping the majority time step resulting in the removal of "
            msg += f"{len(records) - len(abbr_records)} of {len(records)} records. "
            msg += f"{len(abbr_records)} records remain."
            warnings.warn(msg)

        return abbr_records, {majority_dt: majority_count}


def check_nyquist_frequency(dt, fcs):
    # check resampling does not violate the Nyquist.
    fnyq = 1/(2*dt)
    if max(fcs) > fnyq:
        msg = f"The maximum resampling frequency of {np.max(fcs):.2f} Hz "
        msg += f"exceeds the records Nyquist frequency of {fnyq:.2f} Hz"
        raise ValueError(msg)


def traditional_hvsr_processing(records, settings):
    prepare_fft_settings(records, settings)

    records, dt_with_count = prepare_records_with_inconsistent_dt(
        records, settings)

    # allocate array for hvsr results.
    fcs = np.array(settings.smoothing["center_frequencies_in_hz"])
    hvsr_spectra = np.empty((len(records), len(fcs)))
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    # process in groups of constant dt for efficiency.
    hvsr_idx = 0
    cur_idx = 0
    hvsr_indices_to_order = np.empty(len(records), dtype=int)
    for dt, count in dt_with_count.items():

        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], dt)
        raw_spectra = np.empty((count*2, len(fft_frq)))
        hor_idx = 0
        ver_idx = count
        for org_idx, record in enumerate(records):
            # only examine records with the current dt.
            if record.ns.dt_in_seconds != dt:
                continue

            # track original position for later reorder.
            hvsr_indices_to_order[org_idx] = cur_idx
            cur_idx += 1

            # window time series to mitigate frequency-domain artifacts.
            record.window(*settings.window_type_and_width)

            # compute fft of horizontals and combine.
            fft_ns = np.abs(rfft(record.ns.amplitude, **settings.fft_settings))
            fft_ew = np.abs(rfft(record.ew.amplitude, **settings.fft_settings))
            method = COMBINE_HORIZONTAL_REGISTER[settings.method_to_combine_horizontals]
            h = method(fft_ns, fft_ew, settings)

            # compute fft of vertical.
            v = np.abs(rfft(record.vt.amplitude, **settings.fft_settings))

            # store.
            raw_spectra[hor_idx] = h
            raw_spectra[ver_idx] = v
            hor_idx += 1
            ver_idx += 1

        # smooth each dt group at once to boost performance.
        operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
        smooth_spectra = SMOOTHING_OPERATORS[operator](
            fft_frq, raw_spectra, fcs, bandwidth)

        # compute hvsr.
        hvsr_spectra[hvsr_idx:hvsr_idx +
                     count] = smooth_spectra[:count] / smooth_spectra[count:]
        hvsr_idx += count

    # reorder hvsr spectra to follow original order.
    hvsr_spectra = hvsr_spectra[hvsr_indices_to_order]

    if np.isnan(hvsr_spectra).any():
        for idx, spectra in enumerate(hvsr_spectra):
            if np.isnan(spectra).any():
                print(f"{idx} - {spectra}")

    return HvsrTraditional(fcs, hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


def traditional_single_azimuth_hvsr_processing(records, settings):
    prepare_fft_settings(records, settings)

    records, dt_with_count = prepare_records_with_inconsistent_dt(
        records, settings)

    # allocate array for hvsr results.
    fcs = np.array(settings.smoothing["center_frequencies_in_hz"])
    hvsr_spectra = np.empty((len(records), len(fcs)))
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    # process in groups of constant dt for efficiency.
    hvsr_idx = 0
    cur_idx = 0
    hvsr_indices_to_order = np.empty(len(records), dtype=int)
    for dt, count in dt_with_count.items():

        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], dt)
        raw_spectra = np.empty((count*2, len(fft_frq)))
        hor_idx = 0
        ver_idx = count
        for org_idx, record in enumerate(records):
            # only examine records with defined dt.
            if record.ns.dt_in_seconds != dt:
                continue

            # track original position for later reorder.
            hvsr_indices_to_order[org_idx] = cur_idx
            cur_idx += 1

            # combine horizontal components in the time domain.
            h = single_azimuth(record.ns.amplitude,
                               record.ew.amplitude,
                               settings.azimuth_in_degrees)
            h = TimeSeries(h, record.ns.dt_in_seconds)

            # window time series to mitigate frequency-domain artifacts.
            h.window(*settings.window_type_and_width)
            v = TimeSeries.from_timeseries(record.vt)
            v.window(*settings.window_type_and_width)

            # compute fourier transform.
            raw_spectra[hor_idx] = np.abs(
                rfft(h.amplitude, **settings.fft_settings))
            raw_spectra[ver_idx] = np.abs(
                rfft(v.amplitude, **settings.fft_settings))
            hor_idx += 1
            ver_idx += 1

        # smooth each dt group at once to boost performance.
        operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
        smooth_spectra = SMOOTHING_OPERATORS[operator](
            fft_frq, raw_spectra, fcs, bandwidth)

        # compute hvsr.
        hvsr_spectra[hvsr_idx:hvsr_idx +
                     count] = smooth_spectra[:count] / smooth_spectra[count:]
        hvsr_idx += count

    # reorder hvsr spectra to follow original order.
    hvsr_spectra = hvsr_spectra[hvsr_indices_to_order]

    return HvsrTraditional(fcs, hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


def traditional_rotdpp_hvsr_processing(records, settings):
    prepare_fft_settings(records, settings)

    records, dt_with_count = prepare_records_with_inconsistent_dt(
        records, settings)

    # allocate array for hvsr results.
    fcs = np.array(settings.smoothing["center_frequencies_in_hz"])
    hvsr_spectra = np.empty((len(records), len(fcs)))
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    # process in groups of constant dt for efficiency.
    hvsr_idx = 0
    cur_idx = 0
    hvsr_indices_to_order = np.empty(len(records), dtype=int)
    # TODO (jpv): Accelerate computation by smoothing all azimuth and all dt simultaneously.
    for dt, _ in dt_with_count.items():

        fft_frq = np.fft.rfftfreq(settings.fft_settings["n"], dt)
        raw_spectra_per_record = np.empty(
            (len(settings.azimuths_in_degrees)+1, len(fft_frq)))
        for org_idx, record in enumerate(records):
            # only examine records with defined dt.
            if record.ns.dt_in_seconds != dt:
                continue

            # track original position for later reorder.
            hvsr_indices_to_order[org_idx] = cur_idx
            cur_idx += 1

            # prepare vertical component only once per record.
            v = TimeSeries.from_timeseries(record.vt)
            v.window(*settings.window_type_and_width)
            fft_v = np.abs(rfft(v.amplitude, **settings.fft_settings))
            raw_spectra_per_record[-1] = fft_v

            # rotate horizontals through defined azimuths.
            for idx, azimuth in enumerate(settings.azimuths_in_degrees):
                h = single_azimuth(record.ns.amplitude,
                                   record.ew.amplitude,
                                   azimuth
                                   )
                h = TimeSeries(h, dt_in_seconds=record.ns.dt_in_seconds)
                h.window(*settings.window_type_and_width)
                fft_h = np.abs(rfft(h.amplitude, **settings.fft_settings))
                raw_spectra_per_record[idx] = fft_h

            # smooth.
            operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
            smooth_spectra = SMOOTHING_OPERATORS[operator](
                fft_frq, raw_spectra_per_record, fcs, bandwidth)

            # select ppth percentile.
            smooth_h = np.percentile(smooth_spectra[:-1],
                                     settings.ppth_percentile_for_rotdpp_computation,
                                     axis=0)
            smooth_v = smooth_spectra[-1]

            # compute hvsr.
            hvsr_spectra[hvsr_idx] = smooth_h / smooth_v
            hvsr_idx += 1

    # reorder hvsr spectra to follow original order.
    hvsr_spectra = hvsr_spectra[hvsr_indices_to_order]

    return HvsrTraditional(fcs, hvsr_spectra, meta={**records[0].meta, **settings.attr_dict})


TRADITIONAL_PROCESSING_REGISTER = {
    "single_azimuth": traditional_single_azimuth_hvsr_processing,
    "directional_energy": traditional_single_azimuth_hvsr_processing,
    "rotdpp": traditional_rotdpp_hvsr_processing,
    "arithmetic_mean": traditional_hvsr_processing,
    "squared_average": traditional_hvsr_processing,
    "quadratic_mean": traditional_hvsr_processing,
    "root_mean_square": traditional_hvsr_processing,
    "effective_amplitude_spectrum" : traditional_hvsr_processing,
    "geometric_mean": traditional_hvsr_processing,
    "total_horizontal_energy": traditional_hvsr_processing,
    "vector_summation": traditional_hvsr_processing,
    "maximum_horizontal_value": traditional_hvsr_processing,
}


def traditional_hvsr_processing_base(records, settings):
    # need to handle this separately b/c time domain technique
    method = TRADITIONAL_PROCESSING_REGISTER[settings.method_to_combine_horizontals]
    return method(records, settings)


def azimuthal_hvsr_processing(records, settings):
    prepare_fft_settings(records, settings)
    single_azimuth_settings = HvsrTraditionalSingleAzimuthProcessingSettings(
        window_type_and_width=settings.window_type_and_width,
        smoothing=settings.smoothing,
        handle_dissimilar_time_steps_by=settings.handle_dissimilar_time_steps_by,
        fft_settings=settings.fft_settings,
    )
    hvsr_per_azimuth = []
    for azimuth in settings.azimuths_in_degrees:
        single_azimuth_settings.azimuth_in_degrees = azimuth
        hvsr = traditional_single_azimuth_hvsr_processing(records,
                                                          single_azimuth_settings)
        hvsr_per_azimuth.append(hvsr)
    return HvsrAzimuthal(hvsr_per_azimuth, settings.azimuths_in_degrees, meta={**records[0].meta, **settings.attr_dict})


def _rpds_single_component(timeseries, settings):
    """Compute power spectral density of real-valued, time-domain data.

    timeseries : iterable of TimeSeries
        Time-domain data in the form of iterable object containing
        ``TimeSeries`` objects. These data are assumed to have already
        been preprocessed. This is the data that will be processed.
    settings : PsdProcessingSettings
        ``PsdProcessingSettings`` object that controls how the
        time-domain data will be processed.

    Returns
    -------
    Psd
        Instantiated PSD object according to the processing settings
        selected.

    """
    n = settings.fft_settings.get("n", timeseries[0].n_samples)
    psd = np.zeros(n/2 if n % 2 else int(n/2)+1)
    for tseries in timeseries:

        # window time series to mitigate frequency-domain artifacts.
        tseries.window(*settings.window_type_and_width)

        # compute fourier transform.
        fft = rfft(tseries.amplitude, **settings.fft_settings)

        # compute square of fourier transform
        # (not quite the psd; need the normalizations).
        psd += np.real(np.conjugate(fft) * fft)

    # scale according to window; see Welch (1967)
    window = TimeSeries(amplitude=np.ones_like(tseries.amplitude),
                        dt_in_seconds=tseries.dt_in_seconds)
    window.window(*settings.window_type_and_width)
    window_scaling_factor = np.mean(window.amplitude**2)
    psd /= window_scaling_factor

    # scale by number of samples;
    # Welch (1967) shows (1/L)**2 * L scaling for I_{k} simplify to 1/L.
    psd /= tseries.n_samples

    # scaly by number of samples; (add for power spectrum).
    # psd /= tseries.n_samples * tseries.n_samples

    # compute power spectrum density (remove for power spectrum).
    psd /= tseries.fs

    # scale by two b/c only looking at positive frequencies.
    psd *= 2

    # compute average psd over all records (i.e., windows).
    psd /= len(timeseries)

    return psd


def rpsd(records, settings):
    """Compute power spectral density of real-valued, time-domain data.

    records : iterable of SeismicRecording3C
        Time-domain data in the form of iterable object containing
        ``SeismicRecording3C`` objects. This is the data that will be
        processed.
    settings : PsdProcessingSettings
        ``PsdProcessingSettings`` object that controls how the
        time-domain data will be processed.

    Returns
    -------
    Psd
        Instantiated PSD object according to the processing settings
        selected.

    """
    prepare_fft_settings(records, settings)
    fft_frq = np.fft.rfftfreq(
        settings.fft_settings["n"], records[0].vt.dt_in_seconds)
    psd_ns = _rpds_single_component(
        [record.ns for record in records], settings)
    psd_ew = _rpds_single_component(
        [record.ew for record in records], settings)
    psd_vt = _rpds_single_component(
        [record.vt for record in records], settings)

    if settings.smoothing is not None:
        operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
        fcs = np.array(settings.smoothing["center_frequencies_in_hz"])
        spectra = np.empty((3, len(fft_frq)))
        spectra[0] = psd_ns
        spectra[1] = psd_ew
        spectra[2] = psd_vt
        smooth_spectra = SMOOTHING_OPERATORS[operator](
            fft_frq, spectra, fcs, bandwidth)
        fft_frq = fcs
        psd_ns = smooth_spectra[0]
        psd_ew = smooth_spectra[1]
        psd_vt = smooth_spectra[2]

    return dict(ns=Psd(fft_frq, psd_ns),
                ew=Psd(fft_frq, psd_ew),
                vt=Psd(fft_frq, psd_vt))


def diffuse_field_hvsr_processing(records, settings):
    prepare_fft_settings(records, settings)

    records, dt_with_count = prepare_records_with_inconsistent_dt(
        records, settings)

    if len(dt_with_count.keys()) > 1:
        msg = "You cannot use diffuse field processing with records with "
        msg += "dissimilar time steps. Try setting "
        msg += "'handle_dissimilar_time_steps_by' to "
        msg += "'keeping_smallest_time_step' or 'keeping_majority_time_step' "
        msg += "to only process those records with similar time steps."
        raise ValueError(msg)

    # allocate array for hvsr results.
    fcs = np.array(settings.smoothing["center_frequencies_in_hz"])
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    # compute psd.
    fft_frq = np.fft.rfftfreq(
        settings.fft_settings["n"], records[0].vt.dt_in_seconds)
    psd_ns = _rpds_single_component(
        [record.ns for record in records], settings)
    psd_ew = _rpds_single_component(
        [record.ew for record in records], settings)
    psd_vt = _rpds_single_component(
        [record.vt for record in records], settings)

    # smooth.
    operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
    spectra = np.array([psd_ns + psd_ew, psd_vt])
    smooth_spectra = SMOOTHING_OPERATORS[operator](
        fft_frq, spectra, fcs, bandwidth)
    hor = smooth_spectra[0]
    ver = smooth_spectra[1]

    # compute hvsr
    return HvsrDiffuseField(fcs, np.sqrt(hor/ver), meta={**records[0].meta, **settings.attr_dict})


PROCESSING_METHODS = {
    "traditional": traditional_hvsr_processing_base,
    "azimuthal": azimuthal_hvsr_processing,
    "diffuse_field": diffuse_field_hvsr_processing,
    "psd": rpsd,
}

def process(records, settings):
    """Process time domain data.

    records : iterable of SeismicRecording3C
        Time-domain data in the form of iterable object containing
        SeismicRecording3C objects. This is the data that will be
        processed.
    settings : ProcessingSettings
        ``ProcessingSettings`` object that controls how the
        time-domain data will be preprocessed.

    Returns
    -------
    HvsrTraditional, HvsrAzimuthal, HvsrDiffuseField, Psd
        Instantiated object according to the processing settings
        selected.

    """
    return PROCESSING_METHODS[settings.processing_method](records, settings)
