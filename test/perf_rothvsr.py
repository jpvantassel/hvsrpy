# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
# Copyright (C) 2019-2021 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""Performance test for rotated hvsr calculation."""

import hvsrpy
import cProfile
import pstats
from testtools import get_full_path

full_path = get_full_path(__file__)


def main():
    windowlength = 60
    bp_filter = {"flag": True, "flow": 0.1, "fhigh": 45, "order": 5}
    width = 0.1
    bandwidth = 40
    resampling = {"minf": 0.1, "maxf": 50, "nf": 256, "res_type": "log"}
    method = 'rotate'
    azimuth = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    n = 2
    max_iter = 50

    fname = full_path+"data/a2/UT.STN11.A2_C50.miniseed"
    sensor = hvsrpy.Sensor3c.from_mseed(fname)
    hv = sensor.hv(windowlength, bp_filter, width,
                   bandwidth, resampling, method, azimuth=azimuth)
    hv.reject_windows(n=n, max_iterations=max_iter)

fname = full_path+"data/.tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)

# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 03 - 19 : 1.092 s -> Add caching to sigpropy.
