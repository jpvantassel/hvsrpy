# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""Various Hvsr utilities."""

from hvsrpy import Hvsr
import numpy as np


def sesame_clarity(frequency, mean_curve, std_curve, f0_std):
    """Check SESAME (2004) clarity criteria.

    Parameters
    ----------
    frequency : ndarray
        Frequency vector for H/V curve.
    mean_curve : ndarray
        Mean H/V curve
        (assumes lognormal distribution).
    std_curve : ndarray
        Standard deviation of H/V curve
        (assumes lognormal distribution).
    f0_std : float
        Standard deviation of f0 peak
        (assumes normal distribution).

    Returns
    -------
    ndarray
        Of length 6 (one per condition), indicating a pass with a `1`
        and a fail with a `0`.

    """

    def peak_index(curve):
        pot_peak_indices, _ = Hvsr.find_peaks(curve)
        pot_peak_amp = curve[pot_peak_indices]
        pot_peak_index = np.argwhere(pot_peak_amp == np.max(pot_peak_amp))[0][0]
        peak_index = pot_peak_indices[pot_peak_index]
        return peak_index

    criteria = np.zeros(6)

    # Peak Information
    mc_peak_index = peak_index(mean_curve)
    mc_peak_frq = frequency[mc_peak_index]
    mc_peak_amp = mean_curve[mc_peak_index]

    # Criteria i)
    a_low = mean_curve[np.logical_and(frequency < mc_peak_frq,
                                      frequency > mc_peak_frq/4)]
    if np.sum(a_low < mc_peak_amp/2):
        criteria[0] = 1

    # Criteria ii)
    a_high = mean_curve[np.logical_and(frequency > mc_peak_frq,
                                       frequency < 4*mc_peak_frq)]
    
    if np.sum(a_high < mc_peak_amp/2):
        criteria[1] = 1

    # Criteria iii)
    if mc_peak_amp > 2:
        criteria[2] = 1

    # Criteria iv)
    upper_curve = np.exp(np.log(mean_curve) + std_curve)
    lower_curve = np.exp(np.log(mean_curve) - std_curve)

    upper_peak_index = peak_index(upper_curve)
    lower_peak_index = peak_index(lower_curve)

    f_plus = frequency[upper_peak_index]
    f_minus = frequency[lower_peak_index]

    cond_1 = f_plus > mc_peak_frq*0.95 and f_plus < mc_peak_frq*1.05
    cond_2 = f_minus > mc_peak_frq*0.95 and f_minus < mc_peak_frq*1.05

    if cond_1 and cond_2:
        criteria[3] = 1

    # Table for conditions v) and vi)
    if mc_peak_frq < 0.2:
        epsilon = 0.25
        theta = 3
    elif mc_peak_frq < 0.5:
        epsilon = 0.2
        theta = 2.5
    elif mc_peak_frq < 1:
        epsilon = 0.15
        theta = 2
    elif mc_peak_frq < 2:
        epsilon = 0.1
        theta = 1.78
    else:
        epsilon = 0.05
        theta = 1.58

    # Criteria v)
    if f0_std < epsilon*mc_peak_frq:
        criteria[4] = 1

    # Criteria vi)
    sigma_a = upper_curve/mean_curve
    sigma_a_peak = sigma_a[mc_peak_index]

    if sigma_a_peak < theta:
        criteria[5] = 1

    return criteria
