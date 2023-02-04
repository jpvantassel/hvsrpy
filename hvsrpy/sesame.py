# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2021 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

from termcolor import colored

from hvsrpy import HvsrTraditional
import numpy as np


def pass_fail(value):
    return colored("Pass", "green") if value > 0 else colored("Fail", "red")


def clean(number, decimals=3):
    return str(np.round(number, decimals=decimals))


def is_isnot(value):
    return colored("is" if value > 0 else "is not", attrs=["underline"])


def peak_index(curve):
    """Find index to the peak of the provided curve."""
    curve = curve.reshape((1, curve.size))
    pot_peak_indices, _ = HvsrTraditional.find_peaks(curve)
    pot_peak_indices = pot_peak_indices[0]
    pot_peak_amp = curve[0, pot_peak_indices]
    pot_peak_index = np.argwhere(pot_peak_amp == np.max(pot_peak_amp))[0][0]
    return pot_peak_indices[pot_peak_index]


def reliability(windowlength, passing_window_count,
                       frequency, mean_curve, std_curve,
                       search_limits=(None, None), verbose=1):
    """Check SESAME (2004) reliability criteria.

    Parameters
    ----------
    windowlength : float
        Length of time windows in seconds.
    passing_window_count : int
        Number of passing windows included in `mean_curve`.
    frequency : ndarray
        Frequency vector for HVSR curve.
    mean_curve : ndarray
        Mean HVSR curve
        (assumes lognormal distribution).
    std_curve : ndarray
        Standard deviation of HVSR curve
        (assumes lognormal distribution).
    f0_std : float
        Standard deviation of f0 peak from time windows
        (assumes normal distribution).
    search_limits : tuple
        Limits about which to search for f0, default is `(None, None)`
        indicating the full range will be considered.
    verbose : {0, 1, 2}, optional
        Level of verbose logging for SESAME criteria, amount of logging

    Returns
    -------
    ndarray
        Of size 3, one per reliability critera, with 0 indicating a
        failure and 1 indicating a pass.

    """
    limits = []
    limits_were_both_none = True
    for limit, default in zip(search_limits, [min(frequency), max(frequency)]):
        if limit is None:
            limits.append(default)
        else:
            limits.append(float(limit))
            limits_were_both_none = False
    search_limits = tuple(limits)
    if not limits_were_both_none:
        frequency, mean_curve, std_curve = trim_curve(search_limits, frequency,
                                                      mean_curve, std_curve,
                                                      verbose=verbose)

    # Peak Information
    mc_peak_index = peak_index(mean_curve)
    mc_peak_frq = frequency[mc_peak_index]

    if verbose > 0:
        print(colored("Assessing SESAME (2004) reliability criteria ... ",
                      attrs=["bold"]))

    criteria = np.zeros(3)

    # Criteria i)
    if mc_peak_frq > 10/windowlength:
        criteria[0] = 1

    if verbose > 0:
        msg = f"  Criteria i): {pass_fail(criteria[0])}"
        print(msg)

    if verbose > 1:
        msg = f"    f0mc={clean(mc_peak_frq)} {is_isnot(criteria[0])} > 10/windowlength={clean(10/windowlength)}"
        print(msg)

    # Criteria ii)
    nc = windowlength*passing_window_count*mc_peak_frq
    if nc > 200:
        criteria[1] = 1

    if verbose > 0:
        msg = f"  Criteria ii): {pass_fail(criteria[1])}"
        print(msg)

    if verbose > 1:
        msg = f"    nc(f0mc)={clean(nc)} {is_isnot(criteria[1])} > 200"
        print(msg)

    # Criteria iii)
    upper_curve = np.exp(np.log(mean_curve) + std_curve)
    sigma_a = upper_curve/mean_curve
    sigma_a_max = np.max(sigma_a[np.logical_and(frequency > 0.5*mc_peak_frq,
                                                frequency < 2*mc_peak_frq)])

    if mc_peak_frq > 0.5:
        if sigma_a_max < 2:
            criteria[2] = 1
    else:
        if sigma_a_max < 3:
            criteria[2] = 1

    if verbose > 0:
        msg = f"  Criteria iii): {pass_fail(criteria[2])}"
        print(msg)

    if verbose > 1:
        msg = f"    sigma_a(f)={clean(sigma_a_max)} {is_isnot(criteria[2])} < {2 if mc_peak_frq>0.5 else 3}"
        print(msg)

    if verbose > 0:
        overall = colored("PASSES", "green") if np.sum(
            criteria) == 3 else colored("FAILS", "red")
        msg = f"  The chosen peak {overall} the peak reliability criteria, with {int(np.sum(criteria))} of 3."
        print(msg)

    return criteria


def trim_curve(search_limits, frequency, mean_curve, std_curve, verbose=0):
    low_limit, upp_limit = min(search_limits), max(search_limits)
    rel_frq_low = np.abs(frequency - low_limit)
    lower_index = np.where(rel_frq_low == np.min(rel_frq_low))[0][0]
    rel_frq_upp = np.abs(frequency - upp_limit)
    upper_index = np.where(rel_frq_upp == np.min(rel_frq_upp))[0][0]+1

    if verbose > 0:
        msg = f"Considering only frequencies between {clean(low_limit)} and {clean(upp_limit)} Hz."
        print(msg)

    if verbose > 1:
        msg = f"    Lower frequency limit is {clean(frequency[lower_index])} Hz."
        print(msg)
        msg = f"    Upper frequency limit is {clean(frequency[upper_index-1])} Hz."
        print(msg)

    frequency = frequency[lower_index:upper_index]
    mean_curve = mean_curve[lower_index:upper_index]
    std_curve = std_curve[lower_index:upper_index]

    return (frequency, mean_curve, std_curve)


def clarity(frequency, mean_curve, std_curve, f0_std,
                   search_limits=(None, None), verbose=1):
    """Check SESAME (2004) clarity criteria.

    Parameters
    ----------
    frequency : ndarray
        Frequency vector for HVSR curve.
    mean_curve : ndarray
        Mean HVSR curve
        (assumes lognormal distribution).
    std_curve : ndarray
        Standard deviation of HVSR curve
        (assumes lognormal distribution).
    f0_std : float
        Standard deviation of f0 peak from time windows
        (assumes normal distribution).
    search_limits : tuple
        Limits about which to search for f0, default is `(None, None)`
        indicating the full range will be considered.
    verbose : {0, 1, 2}, optional
        Level of verbose logging for SESAME criteria, amount of logging

    Returns
    -------
    ndarray
        Of size 6, one per clarity critera, with 0 indicating a failure
        and 1 indicating a pass.

    """
    if verbose > 0:
        print(colored("Assessing SESAME (2004) clarity criteria ... ",
                      attrs=["bold"]))

    criteria = np.zeros(6)

    limits = []
    limits_were_both_none = True
    for limit, default in zip(search_limits, [min(frequency), max(frequency)]):
        if limit is None:
            limits.append(default)
        else:
            limits.append(float(limit))
            limits_were_both_none = False
    search_limits = tuple(limits)
    if not limits_were_both_none:
        frequency, mean_curve, std_curve = trim_curve(search_limits, frequency,
                                                      mean_curve, std_curve,
                                                      verbose=verbose)

    # Peak Information
    mc_peak_index = peak_index(mean_curve)
    mc_peak_frq = frequency[mc_peak_index]
    mc_peak_amp = mean_curve[mc_peak_index]

    # Criteria i)
    a_low = mean_curve[np.logical_and(frequency < mc_peak_frq,
                                      frequency > mc_peak_frq/4)]
    if np.sum(a_low < mc_peak_amp/2):
        criteria[0] = 1

    if verbose > 0:
        msg = f"  Criteria i): {pass_fail(criteria[0])}"
        print(msg)

    if verbose > 1:
        msg = f"    min(A[f0mc/4,f0mc])={clean(np.min(a_low))} {is_isnot(criteria[0])} < A0[f0mc]/2={clean(mc_peak_amp)}/2={clean(mc_peak_amp/2)}"
        print(msg)

    # Criteria ii)
    a_high = mean_curve[np.logical_and(frequency > mc_peak_frq,
                                       frequency < 4*mc_peak_frq)]

    if np.sum(a_high < mc_peak_amp/2):
        criteria[1] = 1

    if verbose > 0:
        msg = f"  Criteria ii): {pass_fail(criteria[1])}"
        print(msg)

    if verbose > 1:
        msg = f"    min(A[f0mc,f0mc*4])={clean(np.min(a_high))} {is_isnot(criteria[1])} < A0[f0mc]/2={clean(mc_peak_amp)}/2={clean(mc_peak_amp/2)}"
        print(msg)

    # Criteria iii)
    if mc_peak_amp > 2:
        criteria[2] = 1

    if verbose > 0:
        msg = f"  Criteria iii): {pass_fail(criteria[2])}"
        print(msg)

    if verbose > 1:
        msg = f"    A0[f0mc]={clean(mc_peak_amp)} {is_isnot(criteria[2])} > 2.0"
        print(msg)

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

    if verbose > 0:
        msg = f"  Criteria iv): {pass_fail(criteria[3])}"
        print(msg)

    if verbose > 1:
        msg = f"    f0_upper={clean(f_plus)} {is_isnot(cond_1)} within 5% of f0mc={clean(mc_peak_frq)}.\n"
        msg += f"    f0_lower={clean(f_minus)} {is_isnot(cond_2)} within 5% of f0mc={clean(mc_peak_frq)}."
        print(msg)

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

    if verbose > 0:
        msg = f"  Criteria v): {pass_fail(criteria[4])}"
        print(msg)

    if verbose > 1:
        msg = f"    f0_std={f0_std} {is_isnot(criteria[4])} less than "
        msg += f"epsilon*mc_peak_frq={clean(epsilon)}*{clean(mc_peak_frq)}={clean(epsilon*mc_peak_frq)}."
        print(msg)

    # Criteria vi)
    sigma_a = upper_curve/mean_curve
    sigma_a_peak = sigma_a[mc_peak_index]

    if sigma_a_peak < theta:
        criteria[5] = 1

    if verbose > 0:
        msg = f"  Criteria vi): {pass_fail(criteria[5])}"
        print(msg)

    if verbose > 1:
        msg = f"    sigma_a_peak={clean(sigma_a_peak)} {is_isnot(criteria[5])} less than theta={clean(theta)}."
        print(msg)

    if verbose > 0:
        overall = colored("PASSES", "green") if np.sum(
            criteria) > 4 else colored("FAILS", "red")
        print(
            f"  The chosen peak {overall} the peak clarity criteria, with {int(np.sum(criteria))} of 6.")

    return criteria
