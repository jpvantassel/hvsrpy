# This file is part of hvsrpy a Python package for horizontal-to-vertical
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

"""File for organizing some useful plotting functions."""

import time

import matplotlib.pyplot as plt
import pandas as pd

import hvsrpy


def quick_plot(file_name, windowlength=60, width=0.1, bandwidth=40,
               minf=0.2, maxf=20, nf=128, res_type="log",
               filter_bool=False, flow=0.1, fhigh=30, forder=5,
               method="geometric-mean", azimuth=None,
               distribution_f0="log-normal",
               distribution_mc="log-normal",
               rejection_bool=True, n=2, n_iteration=50):  # pragma: no cover
    """Create standard five panel (3-time histories and 2-Hvsr)

    Parameters
    ----------
        file_name : str or dict
            If `str`, name of 3-component time record, may be relative
            or full path, else `dict` of three files (one-component per
            file).
        windowlength : float, optional
            Length of time windows, default is 60 seconds.
        width : {0.0 - 1.0}, optional
            Length of cosine taper, default is 0.1 (5% on each side) of
            time window.
        bandwidth : int, optional
            Bandwidth coefficient for Konno & Ohmachi (1998) smoothing,
            default is 40.
        minf, maxf : float, optional
            Minimum and maximum frequency in Hz to consider when
            resampling, defaults are 0.2 and 20, respectively.
        nf : int, optional
            Number of samples in resampled curve, default is 128.
        res_type : {"log","linear"}, optional
            Type of resampling, default is "log".
        filter_bool : bool
            Controls where the bandpass filter is applied, default is
            `False`.
        flow, fhigh : float, optional
            Upper and lower frequency limits of bandpass filter in Hz,
            defaults are 0.1 and 30 respectively.
        forder : int, optional
            Filter order, default is 5 (i.e., 5th order filter).
        method : {'squared-average', 'geometric-mean'}, optional
            Method for combining the horizontal components, default is
            `geometric-mean`.
        distribution_f0 : {"normal", "log-normal"}, optional
            Distribution assumed to describe the fundamental site
            frequency, default is "log-normal".
        , distribution_mc : {"normal", "log-normal"}, optional
            Distribution assumed to describe the median curve, default
            is "log-normal".
        rejection_bool : bool, optional
            Determines whether the rejection is performed, default is
            True.
        n : float, optional
            Number of standard deviations to consider when performing
            the rejection, default is 2.
        n_iteration : int, optional
            Number of permitted iterations to convergence, default is
            50.

    Returns
    -------
    tuple
        Of the form (fig, axs), where `fig` is the figure object and
        `axs` is a tuple of 5 axes objects.


    """
    if isinstance(file_name, str):
        sensor = hvsrpy.Sensor3c.from_mseed(fname=file_name)
    elif isinstance(file_name, dict):
        sensor = hvsrpy.Sensor3c.from_mseed(fnames_1c=file_name)
    else:
        raise TypeError

    # Direct form simple_hvsrpy_interface.ipynb
    fig = plt.figure(figsize=(6, 6), dpi=150)
    gs = fig.add_gridspec(nrows=6, ncols=6)

    ax0 = fig.add_subplot(gs[0:2, 0:3])
    ax1 = fig.add_subplot(gs[2:4, 0:3])
    ax2 = fig.add_subplot(gs[4:6, 0:3])

    if rejection_bool:
        ax3 = fig.add_subplot(gs[0:3, 3:6])
        ax4 = fig.add_subplot(gs[3:6, 3:6])
    else:
        ax3 = fig.add_subplot(gs[0:3, 3:6])
        ax4 = False

    start = time.time()
    # sensor = hvsrpy.Sensor3c.from_mseed(file_name)
    bp_filter = {"flag": filter_bool, "flow": flow,
                 "fhigh": fhigh, "order": forder}
    resampling = {"minf": minf, "maxf": maxf, "nf": nf, "res_type": res_type}
    hv = sensor.hv(windowlength, bp_filter, width, bandwidth,
                   resampling, method, azimuth=azimuth)
    end = time.time()
    print(f"Elapsed Time: {str(end-start)[0:4]} seconds")

    individual_width = 0.3
    median_width = 1.3
    for ax, title in zip([ax3, ax4], ["Before Rejection", "After Rejection"]):
        # Rejected Windows
        if title == "After Rejection":
            if len(hv.rejected_window_indices) > 0:
                label = "Rejected"
                for amp in hv.amp[hv.rejected_window_indices]:
                    ax.plot(hv.frq, amp, color='#00ffff',
                            linewidth=individual_width, zorder=2, label=label)
                    label = None

        # Accepted Windows
        label = "Accepted"
        for amp in hv.amp[hv.valid_window_indices]:
            ax.plot(hv.frq, amp, color='#888888', linewidth=individual_width,
                    label=label if title == "Before Rejection" else "")
            label = None

        # Window Peaks
        ax.plot(hv.peak_frq, hv.peak_amp, linestyle="", zorder=2,
                marker='o', markersize=2.5, markerfacecolor="#ffffff", markeredgewidth=0.5, markeredgecolor='k',
                label="" if title == "Before Rejection" and rejection_bool else r"$f_{0,i}$")

        # Peak Mean Curve
        ax.plot(hv.mc_peak_frq(distribution_mc), hv.mc_peak_amp(distribution_mc), linestyle="", zorder=4,
                marker='D', markersize=4, markerfacecolor='#66ff33', markeredgewidth=1, markeredgecolor='k',
                label="" if title == "Before Rejection" and rejection_bool else r"$f_{0,mc}$")

        # Mean Curve
        label = r"$LM_{curve}$" if distribution_mc == "log-normal" else "Mean"
        ax.plot(hv.frq, hv.mean_curve(distribution_mc), color='k', linewidth=median_width,
                label="" if title == "Before Rejection" and rejection_bool else label)

        # Mean +/- Curve
        label = r"$LM_{curve}$" + \
            " ± 1 STD" if distribution_mc == "log-normal" else "Mean ± 1 STD"
        ax.plot(hv.frq, hv.nstd_curve(-1, distribution_mc),
                color='k', linestyle='--', linewidth=median_width, zorder=3,
                label="" if title == "Before Rejection" and rejection_bool else label)
        ax.plot(hv.frq, hv.nstd_curve(+1, distribution_mc),
                color='k', linestyle='--', linewidth=median_width, zorder=3)

        label = r"$LM_{f0}$" + \
            " ± 1 STD" if distribution_f0 == "log-normal" else "Mean f0 ± 1 STD"
        ymin, ymax = ax.get_ylim()
        ax.plot([hv.mean_f0_frq(distribution_f0)]*2,
                [ymin, ymax], linestyle="-.", color="#000000")
        ax.fill([hv.nstd_f0_frq(-1, distribution_f0)]*2 + [hv.nstd_f0_frq(+1, distribution_f0)]*2, [ymin, ymax, ymax, ymin],
                color="#ff8080",
                label="" if title == "Before Rejection" and rejection_bool else label)

        ax.set_ylim((ymin, ymax))
        ax.set_xscale('log')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("HVSR Ampltidue")
        if rejection_bool:
            if title == "Before Rejection":
                print("\nStatistics before rejection:")
                hv.print_stats(distribution_f0)
                c_iter = hv.reject_windows(n, max_iterations=n_iteration,
                                           distribution_f0=distribution_f0, distribution_mc=distribution_mc)
            elif title == "After Rejection":
                fig.legend(ncol=4, loc='lower center',
                           bbox_to_anchor=(0.51, 0), columnspacing=2)

                print("\nAnalysis summary:")
                display(pd.DataFrame(columns=[""], index=["Window length", "No. of windows", "Number of iterations to convergence", "No. of rejected windows"],
                                     data=[f"{windowlength}s", str(sensor.ns.n_windows), f"{c_iter} of {n_iteration} allowed", str(len(hv.rejected_window_indices))]))
                print("\nStatistics after rejection:")
                hv.print_stats(distribution_f0)
        else:
            display(pd.DataFrame(columns=[""], index=["Window length", "No. of windows"],
                                 data=[f"{windowlength}s", str(sensor.ns.n_windows)]))
            hv.print_stats(distribution_f0)
            fig.legend(loc="upper center", bbox_to_anchor=(0.77, 0.4))
            break
        ax.set_title(title)

    norm_factor = sensor.normalization_factor
    for ax, timerecord, name in zip([ax0, ax1, ax2], [sensor.ns, sensor.ew, sensor.vt], ["NS", "EW", "VT"]):
        ctime = timerecord.time
        amp = timerecord.amp/norm_factor
        ax.plot(ctime.T, amp.T, linewidth=0.2, color='#888888')
        ax.set_title(f"Time Records ({name})")
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(0, windowlength*timerecord.n_windows)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Amplitude')
        for window_index in hv.rejected_window_indices:
            ax.plot(ctime[window_index], amp[window_index],
                    linewidth=0.2, color="cyan")

    if rejection_bool:
        axs = [ax0, ax3, ax1, ax4, ax2]
    else:
        axs = [ax0, ax3, ax1, ax2]

    for ax, letter in zip(axs, list("abcde")):
        ax.text(0.97, 0.97, f"({letter})", ha="right",
                va="top", transform=ax.transAxes, fontsize=12)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=1, w_pad=2, rect=(0, 0.08, 1, 1))

    return (fig, (ax0, ax1, ax2, ax3, ax4))


def quick_voronoi_plot(points, vertice_set, boundary, ax=None,
                       fig_kwargs=None):  # pragma: no cover
    """Plot Voronoi regions with boundary."""

    ax_was_none = False
    if ax is None:
        ax_was_none = True

        default_fig_kwargs = dict(figsize=(4, 4), dpi=150)
        if fig_kwargs is not None:
            fig_kwargs = {*fig_kwargs, *default_fig_kwargs}

        fig, ax = plt.subplots(**default_fig_kwargs)

    for vertices in vertice_set:
        ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.4)

    ax.plot(points[:, 0], points[:, 1], 'ko')
    ax.plot(boundary[:, 0], boundary[:, 1], color="k")

    ax.set_xlabel("Relative Northing (m)")
    ax.set_ylabel("Relative Easting (m)")

    if ax_was_none:
        return fig, ax
