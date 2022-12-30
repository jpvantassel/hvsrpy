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

def preprocess(recordings, settings):
    """Preprocess time domain data before performing HVSR calculations.

    recordings: iterable of SeismicRecording3C
        Time-domain data in the form of an interable object containing
        SeismicRecording3C objects. This is the data that will be
        preprocessed.
    settings : HvsrPreProcessingSettings
        HvsrPreProcessingSettings object that controls how the
        time-domain data will be preprocessed.

    Returns
    -------
    List of SeismicRecording3C
        Seismic recordings that have been preprocessed and are ready for
        HVSR processing. 

    """
    preprocessed_recordings = []

    for timeseries in recordings:

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

        preprocessed_recordings.extend(windows)

    return preprocessed_recordings


def process(recordings, settings):
    """Process time domain domain data.

    recordings: iterable of SeismicRecording3C
        Time-domain data in the form of interable object containing
        SeismicRecording3C objects. This is the data that will be
        processed.
    settings : HvsrProcessingSettings
        HvsrProcessingSettings object that controls how the
        time-domain data will be processed.

    Returns
    -------
    Hvsr
        Seismic recordings that have been preprocessed and are ready for
        HVSR processing. 

    """
    for recording in recordings:

        # window time series to mitigate frequency-domain artifacts
        recording.window(*settings.window_type_and_width)

        


    if settings.processing_method in ["squared-average", "geometric-mean", "single-azimuth"]:

        # def hvsr_process_typical(recordings, settings):

        #     # Cosine taper each window individually.
        #     recording.cosine_taper(settings.tukey_window_size)

        #     return self._make_hvsr(method=method, resampling=resampling,
        #                             bandwidth=bandwidth, f_low=f_low,
        #                             f_high=f_high, azimuth=azimuth)
        #     return HvsrTypical

        # def hvsr_process_azimuthal(recordings, settings):

        #         # # Combine horizontal components directly.

        #     # # Rotate horizontal components through a variety of azimuths.
        #     # elif method in ["rotate", "multiple-azimuths"]:

        #     # Deprecate `rotate` in favor of the more descriptive `multiple-azimuths`.
        #     if method == "rotate":
        #         msg = "method='rotate' is deprecated, replace with the more descriptive 'multiple-azimuths'."
        #         warnings.warn(msg, DeprecationWarning)

        #     hvsrs = np.empty(len(azimuth), dtype=object)
        #     for index, az in enumerate(azimuth):
        #         hvsrs[index] = self._make_hvsr(method="single-azimuth",
        #                                         resampling=resampling,
        #                                         bandwidth=bandwidth,
        #                                         f_low=f_low,
        #                                         f_high=f_high,
        #                                         azimuth=az)
        #     return HvsrRotated.from_iter(hvsrs, azimuth, meta=self.meta)

        # else:
        #     msg = f"`method`={method} has not been implemented."
        #     raise NotImplementedError(msg)

        # return HvsrAzimuthal

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

        #     def transform(self, **kwargs):
        #         """Perform Fourier transform on components.

        #         Returns
        #         -------
        #         dict
        #             With `FourierTransform`-like objects, one for for each
        #             component, indicated by the key 'ew','ns', 'vt'.

        #         """
        #         ffts = {}
        #         for attr in ["ew", "ns", "vt"]:
        #             tseries = getattr(self, attr)
        #             fft = FourierTransform.from_timeseries(tseries, **kwargs)
        #             ffts[attr] = fft
        #         return ffts

        #     @staticmethod
        #     def _combine_horizontal_fd(method, ns, ew):
        #         """Combine horizontal components in the frequency domain.

        #         Parameters
        #         ----------
        #         method : {'squared-average', 'geometric-mean'}
        #             Defines how the two horizontal components are combined.
        #         ns, ew : FourierTransform
        #             Frequency domain representation of each component.

        #         Returns
        #         -------
        #         FourierTransform
        #             Representing the combined horizontal components.

        #         """
        #         if method == "squared-average":
        #             horizontal = np.sqrt((ns.mag*ns.mag + ew.mag*ew.mag)/2)
        #         elif method == "geometric-mean":
        #             horizontal = np.sqrt(ns.mag * ew.mag)
        #         else:
        #             msg = f"`method`={method} has not been implemented."
        #             raise NotImplementedError(msg)

        #         return FourierTransform(horizontal, ns.frequency, dtype=float)

        #     def _combine_horizontal_td(self, method, azimuth):
        #         """Combine horizontal components in the time domain.

        #         azimuth : float, optional
        #             Azimuth (clockwise positive) from North (i.e., 0 degrees).

        #         Returns
        #         -------
        #         TimeSeries
        #             Representing the combined horizontal components.

        #         """
        #         az_rad = math.radians(azimuth)

        #         if method in ["azimuth", "single-azimuth"]:
        #             horizontal = self.ns._amp * \
        #                 math.cos(az_rad) + self.ew._amp*math.sin(az_rad)
        #         else:
        #             msg = f"method={method} has not been implemented."
        #             raise NotImplementedError(msg)

        #         return TimeSeries(horizontal, self.ns.dt)
