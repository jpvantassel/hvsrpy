# This file is part of hvsrpy, a Python package for
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

"""Single-window integration test for hvsrpy."""

import json

import pandas as pd
import matplotlib.pyplot as plt

import hvsrpy as hv
from testtools import get_full_path

full_path = get_full_path(__file__)


with open(full_path+"data/integration/int_singlewindow_cases.json", "r") as f:
    cases = json.load(f)

bp_filter = {"flag": False, "flow": 0.001, "fhigh": 49.9, "order": 5}

for key, value in cases.items():
    print(f"Running: {key}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))

    sensor = hv.Sensor3c.from_mseed(full_path+value["fname_miniseed"])

    settings = value["settings"]
    my_hv = sensor.hv(settings["length"],
                      bp_filter, settings["width"],
                      settings["b"],
                      settings["resampling"],
                      settings["method"],
                      azimuth=settings.get("azimuth"))

    ax.plot(my_hv.frq, my_hv.amp[0], color='#aaaaaa', label="hvsrpy")

    geopsy_hv = pd.read_csv(full_path+value["fname_geopsy"], delimiter="\t",
                            comment="#", names=["frq", "avg", "min", "max"])
    ax.plot(geopsy_hv.frq, geopsy_hv["avg"],
            color='r', linestyle="--", label="Geopsy")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Ampltidue")
    ax.set_xscale('log')
    ax.legend()
    fig_name = full_path+f"singlewindow_{key}.png"
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()
