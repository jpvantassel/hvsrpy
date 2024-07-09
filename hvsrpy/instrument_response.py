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

"""Functions associated with instrument response."""

# import re

import numpy as np
import scipy.signal as signal
# import obspy

from .timeseries import TimeSeries


class InstrumentTransferFunction():
    def __init__(self, poles, zeros, instrument_sensitivity, normalization_factor):
        """Initialize a ``InstrumentTransferFunction``.

        Parameters
        ----------
        poles : iterable of complex
            Series of complex numbers that define the poles of the
            transfer function. Denominator of H(s).
        zeros : iterable of complex
            Series of complex numbers that define the zeros of the
            transfer function. Numerator of H(s).
        instrument_sensitivity : float
            Defines the sensitivity of the sensing system. By definition
            the sensitivity is the output of the sensing system (sensor
            + data acquisition) over the input to the sensor. For a
            geophone the output would be counts (i.e., the digitized
            voltage) and the input would be particle velocity (m/s).
        normalization_factor : float
            Defines the normalization factor (A0) for the sensor's
            transfer function. The normalization factor ensure that
            the magnitude of the transfer function is 1 at the
            normalization frequency.

        Returns
        -------
        InstrumentTransferFunction
            Initialized ``SensingSystemTransferFunction``.

        """
        self.poles = [complex(p) for p in poles]
        self.zeros = [complex(z) for z in zeros]
        self.instrument_sensitivity = float(instrument_sensitivity)
        self.normalization_factor = float(normalization_factor)

    def _h(self, frequencies):
        b, a = signal.zpk2tf(self.zeros, self.poles, 1.)
        _, h = signal.freqs(b, a, frequencies*2*np.pi)
        h *= self.normalization_factor
        h *= self.instrument_sensitivity
        return h

    def response(self, frequencies):
        """Frequency response of the instrument.

        Parameters
        ----------
        frequencies : ndarray
            Frequencies at which to compute the frequency response of
            the instrument.

        Returns
        -------
        tuple
            Tuple of the form ``(abs(H), angle(H))`` corresponding to
            the amplitude and phase of the transfer function.

        """
        h = self._h(frequencies)
        return abs(h), np.rad2deg(np.angle(h))

    def from_resp(self, fname):
        pass

    # @classmethod
    # def from_station_xml(cls, fname, seed_id="IU.ANMO.00.BHZ"):
    #     inventory = obspy.read_inventory(fname)

    #     # grab start time
    #     with open(fname, "r") as f:
    #         text = f.read()
    #     match = re.search('startDate="(\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ)"', text)
    #     starttime = obspy.UTCDateTime(match.groups()[0])
    #     response = inventory.get_response(seed_id, starttime)
    #     paz = response.get_paz()
    #     return cls(poles=paz.poles,
    #                zeros=paz.zeros,
    #                volts_to_counts=response.instrument_sensitivity.value,
    #                counts_to_mps=1.0,
    #                normalization_constant=paz.normalization_factor)

    def __str__(self):
        return f"InstrumentTransferFunction at {id(self)}"

    def __repr__(self):
        return f"InstrumentTransferFunction(poles={self.poles}, zeros={self.zeros}, instrument_sensitivity={self.instrument_sensitivity}, normalization_factor={self.normalization_factor}"


def _domain_transform(transform_type, timeseries, fft_settings):
    """TODO(jpv) Add private function warning

    Function assumes timeseries has previously been detrended, windowed,
    filtered as appropriate.

    """
    n = fft_settings.get("n", timeseries.n_samples)
    fft = np.fft.rfft(timeseries.amplitude, **fft_settings)
    frq = np.fft.rfftfreq(n, d=timeseries.dt_in_seconds)

    if transform_type == "derivative":
        transfer_funtion = 2*np.pi*frq*1j
    elif transform_type == "integral":
        transfer_funtion = 1/(2*np.pi*frq*1j)
        transfer_funtion[0] = complex(0, 0)
    else:
        raise NotImplementedError
    fft *= transfer_funtion

    ifft = np.fft.irfft(fft, n)
    ifft = ifft[:timeseries.n_samples]
    return TimeSeries(ifft, dt_in_seconds=timeseries.dt_in_seconds)


def _differentiate(timeseries, fft_settings):
    """TODO(jpv) Add private function warning

    Function assumes timeseries has previously been detrended, windowed,
    filtered as appropriate.

    """
    return _domain_transform(transform_type="derivative", timeseries=timeseries, fft_settings=fft_settings)


def _integrate(timeseries, fft_settings):
    """TODO(jpv) Add private function warning

    Function assumes timeseries has previously been detrended, windowed,
    filtered as appropriate.

    """
    return _domain_transform(transform_type="integral", timeseries=timeseries, fft_settings=fft_settings)


def _remove_instrument_response(timeseries, instrument_transfer_function, fft_settings):
    """TODO(jpv) Add private function warning

    Function assumes timeseries has previously been detrended, windowed,
    filtered as appropriate.

    """
    n = fft_settings.get("n", timeseries.n_samples)
    fft = np.fft.rfft(timeseries.amplitude, **fft_settings)
    frq = np.fft.rfftfreq(n, d=timeseries.dt_in_seconds)
    h = instrument_transfer_function._h(frq)

    # invert the transfer function
    invh = np.empty_like(h)
    non_zero_hs = np.abs(h) > 0.
    invh[non_zero_hs] = 1/h[non_zero_hs]
    invh[~non_zero_hs] = complex(0., 0.)
    invh[0] = complex(0., 0.)

    fft *= invh
    ifft = np.fft.irfft(fft, n)
    ifft = ifft[:timeseries.n_samples]

    return TimeSeries(ifft, dt_in_seconds=timeseries.dt_in_seconds)
