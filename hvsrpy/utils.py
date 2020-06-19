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

def sesame_clarity(mean_curve, std_curve):
    """Check SESAME (2004) clarity criteria.
    
    Parameters
    ----------
    mean_curve : ndarray
        Mean H/V curve. Note mean must be from assuming a lognormal
        distribution.
    std_curve : ndarray
        Standard deviation of H/V curve. Note standard deviation must be
        from assuming a lognormal distribution.
    


    """
    criteria = np.zeros(6)

    # Find peaks
    peaks_indices, _ = Hvsr.find_peaks(mean_curve)

    # Find peak with highest amplitude
    potential_peaks = mean_curve[peaks_indices]
    peak_index = np.where(potential_peaks == np.max(potential_peaks))[0]
    
    # Peak
    mc_peak_amp = potential_peaks[peak_index]

    # Criteria iii) A0>2
    if mc_peak_amp > 2:
        criteria[3] = 1
    
    raise NotImplementedError
