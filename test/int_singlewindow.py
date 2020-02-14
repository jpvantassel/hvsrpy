# This file is part of hvsrpy, a Python module for
# horizontal-to-vertical spectral ratio processing.
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

"""This file contains a single-window integration test for hvsrpy."""

import numpy as np
import hvsrpy as hv
import pandas as pd
import matplotlib.pyplot as plt
from testtools import get_full_path

full_path = get_full_path(__file__)

timerecords = [
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C50.miniseed",
    "data/a2/UT.STN12.A2_C50.miniseed",
    "data/a2/UT.STN11.A2_C150.miniseed",
    "data/a2/UT.STN12.A2_C150.miniseed",
]

known_solutions = [
    "data/integration/UT_STN11_c50_single_a.hv",
    "data/integration/UT_STN11_c50_single_b.hv",
    "data/integration/UT_STN11_c50_single_c.hv",
    "data/integration/UT_STN11_c50_single_d.hv",
    "data/integration/UT_STN11_c50_single_e.hv",
    "data/integration/UT_STN11_c50_single_f.hv",
    "data/integration/UT_STN11_c50_single_g.hv",
    "data/integration/UT_STN11_c50_single_h.hv",
    "data/integration/UT_STN11_c50_single_i.hv",
    "data/integration/UT_STN12_c50_single_j.hv",
    "data/integration/UT_STN11_c150_single_k.hv",
    "data/integration/UT_STN12_c150_single_l.hv",
]

settings = [
    {"length": 60, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 120, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 60, "width": 0.1, "b": 10,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 60, "width": 0.1, "b": 80,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 60, "width": 0.2, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 60, "width": 0.02, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 60, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 512, "res_type": "log"}},
    {"length": 60, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 4096, "res_type": "log"}},
    {"length": 30, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 30, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 30, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
    {"length": 30, "width": 0.1, "b": 40,
     "resampling": {"minf": 0.3, "maxf": 40, "nf": 2048, "res_type": "log"}},
]

bp_filter = {"flag": False, "flow": 0.001, "fhigh": 49.9, "order": 5}
ratio_type = 'squared-average'
distribution_type = 'log-normal'

for setting, fname, fname_geopsy in zip(settings, timerecords, known_solutions):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))

    sensor = hv.Sensor3c.from_mseed(full_path+fname)
    my_hv = sensor.hv(setting["length"], bp_filter, setting["width"],
                      setting["b"], setting["resampling"], ratio_type)
    ax.plot(my_hv.frq, my_hv.amp[0], color='#aaaaaa', label="hvsrpy")

    geopsy_hv = pd.read_csv(full_path+fname_geopsy, delimiter="\t", comment="#", names=[
                            "frq", "avg", "min", "max"])
    ax.plot(geopsy_hv.frq, geopsy_hv["avg"], color='r', linestyle="--", label="Geopsy")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Ampltidue")
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(full_path+f"../figs/singlewindow_{fname_geopsy[-4]}.png", dpi=200, bbox_inches='tight')
    plt.close()