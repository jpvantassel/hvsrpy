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

"""Definitions for frequency-domain smoothing functions."""

import numpy as np
from numba import njit


@njit(cache=True)
def konno_ohmachi(frequencies, spectrum, fcs, bandwidth=40.):
    """Fast Konno and Ohmachi smoothing.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies of the spectrum to be smoothed, must be of shape
        `(nfrequency,)`.
    spectrum : ndarray
        Spectrum(s) to be smoothed, must be of shape
        `(nspectrum, nfrequency)`.
    fcs : ndarray
        1D array of center frequencies where smoothed spectrum is
        calculated.
    bandwidth : float, optional
        Width of smoothing window, default is 40.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies (`fcs`).

    """
    # TODO(jpv): Add reference.
    n = 3
    upper_limit = np.power(10, +n/bandwidth)
    lower_limit = np.power(10, -n/bandwidth)

    nrows = spectrum.shape[0]
    ncols = fcs.size
    smoothed_spectrum = np.empty((nrows, ncols))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nrows)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_on_fc = f/fc

            if (f < 1E-6) or (f_on_fc > upper_limit) or (f_on_fc < lower_limit):
                continue
            elif np.abs(f - fc) < 1E-6:
                window = 1.
            else:
                window = bandwidth * np.log10(f_on_fc)
                window = np.sin(window) / window
                window *= window
                window *= window

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


@njit(cache=True)
def parzen(frequencies, spectrum, fcs, bandwidth=0.5):
    """Fast Pazen-style smoothing.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies of the spectrum to be smoothed, must be of shape
        `(nfrequency,)`.
    spectrum : ndarray
        Spectrum(s) to be smoothed, must be of shape
        `(nspectrum, nfrequency)`.
    fcs : ndarray
        1D array of center frequencies where smoothed spectrum is
        calculated.
    bandwidth : float, optional
        Width of smoothing window in Hz, default is 0.5.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies (`fcs`).

    """
    # TODO(jpv): Add reference.
    # after Konno & Ohmachi (1995) in Japanese.
    a = (np.pi*280) / (2*151)
    upper_limit = np.sqrt(6) * a/bandwidth
    lower_limit = -1 * upper_limit

    nrows = spectrum.shape[0]
    ncols = fcs.size
    smoothed_spectrum = np.empty((nrows, ncols))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nrows)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_minus_fc = f - fc

            if (f < 1E-6) or (f_minus_fc > upper_limit) or (f_minus_fc < lower_limit):
                continue
            elif np.abs(f - fc) < 1E-6:
                window = 1.
            else:
                window = a*f_minus_fc / bandwidth
                window = np.sin(window) / window
                window *= window
                window *= window

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


SAVITZKY_AND_GOLAY = {
    5: {"coeff": np.array([-3, 12, 17], dtype=np.float),
        "norm": 35},
    7: {"coeff": np.array([-2, 3, 6, 7], dtype=np.float),
        "norm": 21},
    9: {"coeff": np.array([-21, 14, 39, 54, 59], dtype=np.float),
        "norm": 231},
    11: {"coeff": np.array([-36, 9, 44, 69, 84, 89], dtype=np.float),
         "norm": 429},
    13: {"coeff": np.array([-11, 0, 9, 16, 21, 24, 25], dtype=np.float),
         "norm": 143},
    15: {"coeff": np.array([-78, -13, 42, 87, 122, 147, 162, 167], dtype=np.float),
         "norm": 1105},
    17: {"coeff": np.array([-21, -6, 7, 18, 27, 34, 39, 42, 43], dtype=np.float),
         "norm": 323},
    19: {"coeff": np.array([-136, -51, 24, 89, 144, 189, 224, 249, 264, 269], dtype=np.float),
         "norm": 2261},
    21: {"coeff": np.array([-171, -76, 9, 84, 149, 204, 249, 284, 309, 324, 329], dtype=np.float),
         "norm": 3059},
    23: {"coeff": np.array([-42, -21, -2, 15, 30, 43, 54, 63, 70, 75, 78, 79], dtype=np.float),
         "norm": 8059},
    25: {"coeff": np.array([-253, -138, -33, 62, 147, 222, 287, 322, 387, 422, 447, 462, 467], dtype=np.float),
         "norm": 5175},
}


def savitzky_and_golay(frequencies, spectrum, fcs, bandwidth=9):
    """Fast Savitzky and Golay (1964) smoothing.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies of the spectrum to be smoothed, must be of shape
        `(nfrequency,)`. Must be linearly spaced.
    spectrum : ndarray
        Spectrum(s) to be smoothed, must be of shape
        `(nspectrum, nfrequency)`.
    fcs : ndarray
        1D array of center frequencies where smoothed spectrum is
        calculated.
    bandwidth : int, optional
        Number of points in the smoothing operator, default is 9.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies (`fcs`).

    """
    # TODO(jpv): Add reference.
    try:
        coeff_dict = SAVITZKY_AND_GOLAY[bandwidth]
    except KeyError as e:
        msg = f"savitzky_and_golay smoothing with bandwidth={bandwidth}, "
        msg += "has not been implemented, available bandwidths "
        msg += f"are {list(SAVITZKY_AND_GOLAY.keys())}"
        raise NotImplementedError(msg)

    diff = np.diff(frequencies)
    if np.abs(np.min(diff) - np.max(diff)) > 1E-6:
        msg = "For savitzky_and_golay smoothing frequencies must be "
        msg += "linearly spaces."
        raise ValueError(msg)

    coefficients, normalization_factor = coeff_dict["coeff"], coeff_dict["norm"]

    df = diff[0]
    nfcs = np.round(fcs / df).astype(np.int)

    return _savitzky_and_golay(spectrum, nfcs, coefficients, normalization_factor)


@njit(cache=True)
def _savitzky_and_golay(spectrum, nfcs, coefficients, normalization_factor):

    nrows, nfreqs = spectrum.shape
    ncols = nfcs.size
    smoothed_spectrum = np.empty((nrows, ncols))

    ncoeff = coefficients.size

    for nfc_idx, spectrum_idx in enumerate(nfcs):

        if (spectrum_idx < ncoeff) or (spectrum_idx + ncoeff > nfreqs):
            smoothed_spectrum[:, nfc_idx] = 0
            continue

        summation = coefficients[-1] * spectrum[:, spectrum_idx]
        for rel_idx, coefficient in enumerate(coefficients[:-1][::-1]):
            summation += coefficient * (spectrum[:, spectrum_idx + rel_idx] +
                                        spectrum[:, spectrum_idx - rel_idx])

        smoothed_spectrum[:, nfc_idx] = summation / normalization_factor

    return smoothed_spectrum

@njit(cache=True)
def linear_rectangular(frequencies, spectrum, fcs, bandwidth=0.5):
    nspectra, _ = spectrum.shape
    nfcs = fcs.size
    smoothed_spectrum = np.empty((nspectra, nfcs))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nspectra)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_minus_fc = f - fc

            if (f < 1E-6) or (np.abs(f_minus_fc) > bandwidth/2):
                continue
            else:
                window = 1.

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum

@njit(cache=True)
def log_rectangular(frequencies, spectrum, fcs, bandwidth=0.05):
    lower_limit = np.power(10, -bandwidth/2)
    upper_limit = np.power(10, +bandwidth/2)

    nspectra, _ = spectrum.shape
    nfcs = fcs.size
    smoothed_spectrum = np.empty((nspectra, nfcs))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nspectra)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_on_fc = f / fc

            if (f < 1E-6) or (f_on_fc < lower_limit) or (f_on_fc > upper_limit):
                continue
            else:
                window = 1.

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum

@njit(cache=True)
def linear_triangular(frequencies, spectrum, fcs, bandwidth=0.5):
    nspectra, _ = spectrum.shape
    nfcs = fcs.size
    smoothed_spectrum = np.empty((nspectra, nfcs))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nspectra)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_minus_fc = f - fc

            if (f < 1E-6) or (np.abs(f_minus_fc) > bandwidth/2):
                continue
            else:
                window = 1. - np.abs(f_minus_fc)*(-2/bandwidth)

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum

@njit(cache=True)
def log_triangular(frequencies, spectrum, fcs, bandwidth=0.05):
    lower_limit = np.power(10, -bandwidth/2)
    upper_limit = np.power(10, +bandwidth/2)

    nspectra, _ = spectrum.shape
    nfcs = fcs.size
    smoothed_spectrum = np.empty((nspectra, nfcs))

    for fc_index, fc in enumerate(fcs):

        if fc < 1E-6:
            smoothed_spectrum[:, fc_index] = 0
            continue

        sumproduct = np.zeros(nspectra)
        sumwindow = 0

        for f_index, f in enumerate(frequencies):
            f_on_fc = f/fc

            if (f < 1E-6) or (f_on_fc < lower_limit) or (f_on_fc > upper_limit):
                continue
            elif f_on_fc < 1:
                window = (f_on_fc - lower_limit) / (1. - lower_limit)
            else:
                window = 1 - (f_on_fc - 1.) / (upper_limit - 1)

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


SMOOTHING_OPERATORS = {
    "konno_and_ohmachi" : konno_ohmachi,
    "parzen" : parzen,
    "savitzky_and_golay" : savitzky_and_golay,
    "linear_rectangular" : linear_rectangular,
    "log_rectangular" : log_rectangular,
    "linear_triangular" : linear_triangular,
    "log_rectangular" : log_rectangular,
}