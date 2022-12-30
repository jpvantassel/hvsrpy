# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2022 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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
from numpy import fft

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

    for timeseries in records:

        # TODO(jpv): Add orient to north functionality.

        # bandpass filter raw signal.
        timeseries.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        # divide raw signal into time windows
        if settings.window_length_in_seconds is not None:
            windows = timeseries.split(settings.window_length_in_seconds)
        else:
            windows = [timeseries]

        for window in windows:

            # detrend each time window individually.
            if settings.detrend is not None:
                window.detrend(type=settings.detrend)

        preprocessed_records.extend(windows)

    return preprocessed_records

def arithmetic_mean(ns, ew):
    """Computes the arithmetic mean of two vectors."""
    return (ns+ew) / 2

def squared_average(ns, ew):
    """Computes the squared average (aka quadratic mean) of two vectors."""
    return np.sqrt((ns*ns + ew*ew)/2)

def geometric_mean(ns, ew):
    """Computes the geometric mean of two vectors."""
    return np.sqrt(ns * ew)

def total_horizontal_energy(ns, ew):
    """Computes the magnitude of sum of two orthoginal vectors."""
    return np.sqrt((ns*ns + ew*ew))

def maximum_horizontal_value(ns, ew):
    """Computes the entry-by-entry maximum of the vecotrs provided."""
    return np.where(ns>ew, ns, ew)

METHODS_TO_COMBINE_HORIZONTAL_COMPONENTS = {
    "arithmetic_mean" :arithmetic_mean,
    "squared_average": squared_average,
    "quadratic_mean" : squared_average,
    "geometric_mean": geometric_mean,
    "total_horizontal_energy" : total_horizontal_energy,
    "maximum_horizontal_value" : maximum_horizontal_value
}

# PREPARATION_METHODS = {
#     "arithmetic_mean" : prepare_in_frequency_domain,
#     "squared_average": prepare_in_frequency_domain,
#     "quadratic_mean" : prepare_in_frequency_domain,
#     "geometric_mean": prepare_in_frequency_domain,
#     "total_horizontal_energy" : prepare_in_frequency_domain,
#     "maximum_horizontal_value" : prepare_in_frequency_domain,
#     "single_azimuth": prepare_in_time_domain,
#     "multiple_azimuth": prepare_in_time_domain,
#     "rotd50" : prepare_in_time_domain,
# }

# def single_azimuth(ns, ew):
#     """Computes the magnitude of two vectors along a specifc azimuth."""
#     radians_from_north = np.radians(degrees_from_north)
#     return ns*np.cos(radians_from_north) + ew*np.sin(radians_from_north)

# def prepare_in_time_domain(record, settings):
#     pass

# def prepare_in_frequency_domain(record, settings):
#     pass

# def hvsr_process_typical(recordings, settings):

#     return HvsrTypical


# def hvsr_process_azimuthal(recordings, settings):

#     hvsrs = np.empty(len(azimuth), dtype=object)
#     for index, az in enumerate(azimuth):
#         hvsrs[index] = self._make_hvsr(method="single-azimuth",
#                                         resampling=resampling,
#                                         bandwidth=bandwidth,
#                                         f_low=f_low,
#                                         f_high=f_high,
#                                         azimuth=az)
#     return HvsrRotated.from_iter(hvsrs, azimuth, meta=self.meta)

def traditional_hvsr_processing(records, settings):
    hvsr_spectra = []
    for record in records:
        # window time series to mitigate frequency-domain artifacts.
        record.window(*settings.window_type_and_width)

        # compute fourier transform.
        if settings.fft_settings is None:
            settings.fft_settings = dict(n=records.ns.nsamples)
        ns = np.abs(fft.fft(record.ns.amplitude, **settings.fft_settings)
        ew = np.abs(fft.fft(record.ew.amplitude, **settings.fft_settings)
        vt = np.abs(fft.fft(record.vt.amplitude, **settings.fft_settings)
        # TODO (jpv): Finish HVSR processing calculations.

        # prepare frequency vector
        frq = fft.fftfreq(n, d=records.ns.dt)

        # 



        method = PREPARATION_METHODS[settings.method_to_combine_horizontals]
        h, v = method(record, settings)

        # smoothing

    return Hvsr



def azimuthal_hvsr_processing(records, settings):
    pass

def diffuse_field_hvsr_processing(records, settings):
    pass


PROCESSING_METHODS = {
    "traditional" : traditional_hvsr_processing,
    "azimuthal" : azimuthal_hvsr_processing,
    "diffuse_field" : diffuse_field_hvsr_processing
}



def process(records, settings):
    """Process time domain domain data.

    records: iterable of SeismicRecording3C
        Time-domain data in the form of interable object containing
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



# def _make_hvsr(self, method, resampling, bandwidth, f_low=None, f_high=None, azimuth=None):
#     if method in ["squared-average", "geometric-mean"]:
#         ffts = self.transform()
#         hor = self._combine_horizontal_fd(
#             method=method, ew=ffts["ew"], ns=ffts["ns"])
#         ver = ffts["vt"]
#         del ffts
#     elif method == "single-azimuth":
#         hor = self._combine_horizontal_td(method=method,
#                                             azimuth=azimuth)
#         hor = FourierTransform.from_timeseries(hor)
#         ver = FourierTransform.from_timeseries(self.vt)
#     else:
#         msg = f"`method`={method} has not been implemented."
#         raise NotImplementedError(msg)

#     self.meta["method"] = method
#     self.meta["azimuth"] = azimuth

#     # TODO (jpv): Move these sampling out of the make method
#     if isinstance(resampling, dict):
#         if resampling["res_type"] == "linear":
#             frq = np.linspace(resampling["minf"],
#                             resampling["maxf"],
#                             resampling["nf"])
#         elif resampling["res_type"] == "log":
#             frq = np.geomspace(resampling["minf"],
#                             resampling["maxf"],
#                             resampling["nf"])
#         else:
#             msg = f"`res_type`={resampling['res_type']} has not been implemented."
#             raise NotImplementedError(msg)
#     else:
#         frq = np.array(resampling)

#     hor.smooth_konno_ohmachi_fast(frq, bandwidth)
#     ver.smooth_konno_ohmachi_fast(frq, bandwidth)
#     hor._amp /= ver._amp
#     hvsr = hor
#     del ver

#     if self.ns.nseries == 1:
#         window_length = max(self.ns.time)
#     else:
#         window_length = max(self.ns.time[0])

#     self.meta["Window Length"] = window_length

#     return Hvsr(hvsr.amplitude, hvsr.frequency, find_peaks=False,
#                 f_low=f_low, f_high=f_high, meta=self.meta)

# def transform(self, **kwargs):
#     """Perform Fourier transform on components.

#     Returns
#     -------
#     dict
#         With `FourierTransform`-like objects, one for for each
#         component, indicated by the key 'ew','ns', 'vt'.

#     """
#     ffts = {}
#     for attr in ["ew", "ns", "vt"]:
#         tseries = getattr(self, attr)
#         fft = FourierTransform.from_timeseries(tseries, **kwargs)
#         ffts[attr] = fft
#     return ffts

