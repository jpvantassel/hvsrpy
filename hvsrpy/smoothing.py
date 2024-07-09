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
def konno_and_ohmachi(frequencies, spectrum, fcs, bandwidth=40.): # pragma: no cover
    """Fast Konno and Ohmachi (1998) smoothing.

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
        Value inversely related to the width of the smoothing
        window, default is 40.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies (`fcs`).

    Reference
    ---------
    .. [1] Konno, K. and Ohmachi, T. (1998), "Ground-Motion
       Characteristics Estimated from Spectral Ratio between Horizontal
       and Vertical Components of Microtremor" Bull. Seism. Soc. Am. 88,
       228-241.

    """
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
def parzen(frequencies, spectrum, fcs, bandwidth=0.5): # pragma: no cover
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

    Reference
    ---------
    .. [1] Konno, K. and Ohmachi, T. (1995), "A smoothing function
       suitable for estimation of amplification factor of the surface
       ground from microtremor and its application" Doboku Gakkai
       Ronbunshu. 525, 247-259 (in Japanese).

    """
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


def savitzky_and_golay(frequencies, spectrum, fcs, bandwidth=9): # pragma: no cover
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

    Reference
    ---------
    .. [1] Savitzky, A. and Golay, M.J.E. (1964), "Smoothing and
       Differentiation of Data by Simplified Least Squares Procedures"
       Anal. Chem. 36, 1627-1639.

    """
    m = int(bandwidth)
    if m % 2 != 1:
        raise ValueError("bandwidth for savitzky_and_golay must be an odd integer.")

    nterms = ((m - 1) // 2) + 1
    coefficients = np.empty((nterms))
    for idx, i in enumerate(range(-(nterms-1), 1)):
        coefficients[idx] = ((3*m*m - 7 - 20*abs(i*i))/4)
    normalization_coefficient = (m*(m*m - 4)/3)

    diff = np.diff(frequencies)
    if np.abs(np.min(diff) - np.max(diff)) > 1E-6:
        msg = "For savitzky_and_golay frequency samples of input data "
        msg += "must be linearly spaced."
        raise ValueError(msg)

    df = diff[0]
    nfcs = np.round((fcs - np.min(frequencies)) / df).astype(int)

    return _savitzky_and_golay(spectrum, nfcs, coefficients, normalization_coefficient)


@njit(cache=True)
def _savitzky_and_golay(spectrum, nfcs, coefficients, normalization_coefficient): # pragma: no cover

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
            summation += coefficient * (spectrum[:, spectrum_idx + (rel_idx + 1)] +
                                        spectrum[:, spectrum_idx - (rel_idx + 1)])

        smoothed_spectrum[:, nfc_idx] = summation / normalization_coefficient

    return smoothed_spectrum


@njit(cache=True)
def linear_rectangular(frequencies, spectrum, fcs, bandwidth=0.5): # pragma: no cover
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
def log_rectangular(frequencies, spectrum, fcs, bandwidth=0.05): # pragma: no cover
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
            smoothed_spectrum[:, fc_index] = sumproduct/sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


@njit(cache=True)
def linear_triangular(frequencies, spectrum, fcs, bandwidth=0.5): # pragma: no cover
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
                window = 1. - np.abs(f_minus_fc)*(2/bandwidth)

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct/sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


@njit(cache=True)
def log_triangular(frequencies, spectrum, fcs, bandwidth=0.05): # pragma: no cover 
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
            else:
                window = 1 - np.abs(np.log10(f_on_fc))*(2/bandwidth)

            sumproduct += window*spectrum[:, f_index]
            sumwindow += window

        if sumwindow > 0:
            smoothed_spectrum[:, fc_index] = sumproduct/sumwindow
        else:
            smoothed_spectrum[:, fc_index] = 0

    return smoothed_spectrum


SMOOTHING_OPERATORS = {
    "konno_and_ohmachi": konno_and_ohmachi,
    "parzen": parzen,
    "savitzky_and_golay": savitzky_and_golay,
    "linear_rectangular": linear_rectangular,
    "log_rectangular": log_rectangular,
    "linear_triangular": linear_triangular,
    "log_triangular": log_triangular,
}
