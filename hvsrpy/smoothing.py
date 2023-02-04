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

"""Definitions for frequency-domain smoothing functions."""

import numpy as np
from numba import njit


def konno_ohmachi_1d(frequencies, spectrum, fcs, bandwidth=40.):
    spectrum = np.atleast_2d(spectrum)
    spectrum = konno_ohmachi(frequencies, spectrum, fcs, bandwidth=bandwidth)
    return spectrum.flatten()

@njit(cache=True)
def konno_ohmachi(frequencies, spectrum, fcs, bandwidth=40.):
    """Fast Konno and Ohmachi smoothing.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies of the spectrum to be smoothed.
    spectrum : ndarray
        Spectrum to be smoothed, must be the same size as
        frequencies.
    fcs : ndarray
        Array of center frequencies where smoothed spectrum is
        calculated.
    bandwidth : float, optional
        Width of smoothing window, default is 40.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies
        (`fcs`).

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
            elif np.abs(f - fc) < 1e-6:
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