"""This file contains an integration test for hvsrpy."""

import numpy as np
import hvsrpy as hv
import pandas as pd
import matplotlib.pyplot as plt

timerecords = [
    "test/data/a2/UT.STN11.A2_C50.miniseed",
    "test/data/a2/UT.STN11.A2_C150.miniseed",
    "test/data/a2/UT.STN12.A2_C150.miniseed",
    "test/data/a2/UT.STN12.A2_C50.miniseed"
]

known_solutions = [
    # "test/data/integration_data/UT_STN11_c50_single_2g.hv",
    "test/data/integration_data/UT_STN11_c50.hv",
    # "test/data/integration_data/UT_STN11_c150.hv",
]

# Settings
windowlength = 60
flow = 0.001
fhigh = 49.9
forder = 5
width = 0.10
bandwidth = 10.
fmin = 0.3
fmax = 40.
fn = 2048
res_type = 'log'
ratio_type = 'squared-average'
distribution_type = 'log-normal'
# distribution_type = 'normal'
n = 3

# for fname, fname_geopsy in zip(timerecords, known_solutions):
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))

#     sensor = hv.Sensor3c.from_mseed(fname)
#     my_hv = sensor.hv(windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type)
#     ax.plot(my_hv.frq, my_hv.amp[0], color='#aaaaaa')

#     geopsy_hv = pd.read_csv(fname_geopsy, delimiter="\t", comment="#", names=["frq", "avg", "min", "max"])
#     ax.plot(geopsy_hv.frq, geopsy_hv["avg"], color='r', linestyle="--")

#     # print(np.mean(geopsy_hv["avg"]/my_hv.mean_curve(distribution_type)))
#     # fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
#     # ax2.plot(geopsy_hv.frq, geopsy_hv["avg"]/my_hv.mean_curve(distribution_type))
#     # ax2.set_xscale('log')

#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("H/V Ampltidue (#)")
#     ax.set_xscale('log')
# plt.show()

for fname, fname_geopsy in zip(timerecords, known_solutions):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))

    sensor = hv.Sensor3c.from_mseed(fname)
    my_hv = sensor.hv(windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type)
    for amp in my_hv.amp:
        plt.plot(my_hv.frq, amp, color='#aaaaaa')
    ax.plot(my_hv.frq, my_hv.mean_curve(), color='k')
    if distribution_type == "log-normal":
        ax.plot(my_hv.frq, np.exp(np.log(my_hv.mean_curve())+my_hv.std_curve()), color='k', linestyle='--')
        ax.plot(my_hv.frq, np.exp(np.log(my_hv.mean_curve())-my_hv.std_curve()), color='k', linestyle='--')
    elif distribution_type == "normal":
        ax.plot(my_hv.frq, my_hv.mean_curve(distribution_type)+my_hv.std_curve(distribution_type), color='k', linestyle='--')
        ax.plot(my_hv.frq, my_hv.mean_curve(distribution_type)-my_hv.std_curve(distribution_type), color='k', linestyle='--')
    else:
        raise ValueError

    geopsy_hv = pd.read_csv(fname_geopsy, delimiter="\t", comment="#", names=["frq", "avg", "min", "max"])
    ax.plot(geopsy_hv.frq, geopsy_hv["avg"], color='r', linestyle="--")
    ax.plot(geopsy_hv.frq, geopsy_hv["min"], color='r', linestyle=':')
    ax.plot(geopsy_hv.frq, geopsy_hv["max"], color='r', linestyle=':')

    print(np.mean(geopsy_hv["avg"]/my_hv.mean_curve(distribution_type)))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
    ax2.plot(geopsy_hv.frq, geopsy_hv["avg"]/my_hv.mean_curve(distribution_type))
    ax2.set_xscale('log')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Ampltidue (#)")
    ax.set_xscale('log')
plt.show()

