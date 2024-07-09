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

"""Performance test  calculation."""

import numpy as np
import cProfile
import pstats

import hvsrpy
from testing_tools import get_full_path

full_path = get_full_path(__file__, result_as_string=False)


def main():
    fname = full_path / "data/input/mseed_combined/ut.stn11.a2_c50.mseed"
    srecords = hvsrpy.read([[fname]])

    preprocessing_settings = hvsrpy.HvsrPreProcessingSettings()
    preprocessing_settings.filter_corner_frequencies_in_hz = (0.1, 45)
    preprocessing_settings.window_length_in_seconds = 60.

    srecords = hvsrpy.preprocess(srecords, preprocessing_settings)

    processing_settings = hvsrpy.HvsrTraditionalProcessingSettings()
    processing_settings.smoothing = dict(
        operator="konno_and_ohmachi",
        bandwidth=40,
        center_frequencies_in_hz=np.geomspace(0.1, 50, 256),
    )

    hvsr = hvsrpy.process(srecords, processing_settings)

    _ = hvsrpy.frequency_domain_window_rejection(hvsr, max_iterations=10)


fname = str(full_path / "data/.tmp_profiler_run")
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.005)

# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2019 - 11 - 12 : 1.242 s
# 2020 - 03 - 19 : 1.300 s -> After overhaul of sigpropy smoothing.
# 2020 - 03 - 19 : 0.313 s -> Add caching to sigpropy.
# 2024 - 06 - 05 : 0.552 s -> Remove sigpropy as dependency.