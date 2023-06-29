# This file is part of hvsrpy a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import stats

from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal

__all__ = ["single_plot", "simple_plot", "azimuthal_plot", "voronoi_plot"]

DEFAULT_KWARGS = {
    "width_of_individual_hvsr_curve": 0.3,
    "color_of_individual_valid_hvsr_curve": "#888888",
    "label_of_individual_valid_hvsr_curve": "Accepted HVSR Curve",
    "color_of_individual_invalid_hvsr_curve": "#00ffff",
    "label_of_individual_invalid_hvsr_curve": "Rejected HVSR Curve",
    "color_of_mean_hvsr_curve": "black",
    "width_of_mean_hvsr_curve": 1.3,
    "label_of_mean_hvsr_curve": "Mean Curve",
    "color_of_nth_std_mean_hvsr_curve": "black",
    "width_of_nth_std_mean_hvsr_curve": 1.3,
    "label_of_nth_std_mean_hvsr_curve": r"$\pm$ 1 Std Curve",
    "linestyle_of_nth_std_mean_hvsr_curve": "--",
    "color_of_nth_std_frequency_range": "#ff8080",
    "label_of_nth_std_frequency_range_normal": r"$\mu_{fn} \pm \sigma_{fn}$",
    "label_of_nth_std_frequency_range_lognormal": r"$(\mu_{ln,fn} \pm \sigma_{ln,fn})^*$",
}


def _plot_individual_hvsr_curves(ax, hvsr, valid=True, plot_kwargs=None): # pragma: no cover
    """Plot individual HVSR curves.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else:
        msg = "Can only plot valid HVSR curves from HvsrTraditional "
        msg += "and HvsrAzimuthal objects."
        raise NotImplementedError(msg)

    if valid:
        default_kwargs = dict(linewidth=DEFAULT_KWARGS["width_of_individual_hvsr_curve"],
                              color=DEFAULT_KWARGS["color_of_individual_valid_hvsr_curve"],
                              label=DEFAULT_KWARGS["label_of_individual_valid_hvsr_curve"])
    else:
        default_kwargs = dict(linewidth=DEFAULT_KWARGS["width_of_individual_hvsr_curve"],
                              color=DEFAULT_KWARGS["color_of_individual_invalid_hvsr_curve"],
                              label=DEFAULT_KWARGS["label_of_individual_invalid_hvsr_curve"])
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    for hvsr in hvsrs:
        to_plot = hvsr.valid_window_boolean_mask if valid else ~hvsr.valid_window_boolean_mask
        for amplitude in hvsr.amplitude[to_plot]:
            ax.plot(hvsr.frequency, amplitude, **plot_kwargs)
            plot_kwargs["label"] = None


def _plot_mean_hvsr_curve(ax, hvsr, distribution="lognormal", plot_kwargs=None): # pragma: no cover
    """Plot mean HVSR curve.

    .. warning::
        Private methods are subject to change without warning.

    """
    default_kwargs = dict(linewidth=DEFAULT_KWARGS["width_of_mean_hvsr_curve"],
                          color=DEFAULT_KWARGS["color_of_mean_hvsr_curve"],
                          label=DEFAULT_KWARGS["label_of_mean_hvsr_curve"])
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}
    ax.plot(hvsr.frequency, hvsr.mean_curve(distribution=distribution), **plot_kwargs)


def _plot_nth_std_hvsr_curve(ax, hvsr, distribution="lognormal", n=1., plot_kwargs=None): # pragma: no cover
    """Plot nth standard deviation HVSR curve.

    .. warning::
        Private methods are subject to change without warning.

    """
    default_kwargs = dict(linewidth=DEFAULT_KWARGS["width_of_nth_std_mean_hvsr_curve"],
                          color=DEFAULT_KWARGS["color_of_nth_std_mean_hvsr_curve"],
                          label=DEFAULT_KWARGS["label_of_nth_std_mean_hvsr_curve"],
                          linestyle=DEFAULT_KWARGS["linestyle_of_nth_std_mean_hvsr_curve"])
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}
    ax.plot(hvsr.frequency, hvsr.nth_std_curve(n=n, distribution=distribution), **plot_kwargs)


def _plot_nth_std_frequency_range(ax, hvsr, distribution="lognormal", n=1., fill_kwargs=None): # pragma: no cover
    """Plot nth standard deviation frequency.

    .. warning::
        Private methods are subject to change without warning.

    """
    default_kwargs = dict(color=DEFAULT_KWARGS["color_of_nth_std_frequency_range"],
                          label=DEFAULT_KWARGS[f"label_of_nth_std_frequency_range_{distribution}"],)
    fill_kwargs = default_kwargs if fill_kwargs is None else {**default_kwargs, **fill_kwargs}
    (y_min, y_max) = ax.get_ylim()
    f_min = hvsr.nth_std_fn_frequency(n=-n, distribution=distribution)
    f_max = hvsr.nth_std_fn_frequency(n=+n, distribution=distribution)
    ax.fill([f_min, f_min, f_max, f_max],
            [y_min, y_max, y_max, y_min], **fill_kwargs)
    ax.set_ylim((y_min, y_max))


def _plot_resonance_pdf(ax, hvsr, distribution="lognormal", contour_kwargs=None): # pragma: no cover
    """Plot resonance probability density function (PDF).

    .. warning::
        Private methods are subject to change without warning.

    """
    # define bivariate (log)normal.
    mean_frequency = hvsr.mean_fn_frequency(distribution=distribution)
    mean_frequency = mean_frequency if distribution == "normal" else np.log(mean_frequency)
    mean_amplitude = hvsr.mean_fn_amplitude(distribution=distribution)
    mean_amplitude = mean_amplitude if distribution == "normal" else np.log(mean_amplitude)
    cov = hvsr.cov_fn(distribution=distribution)
    std_frequency = np.sqrt(cov[0, 0])
    std_amplitude = np.sqrt(cov[1, 1])

    # plot pdf.
    f_lower, f_upper = mean_frequency - 3*std_frequency, mean_frequency + 3*std_frequency
    a_lower, a_upper = mean_amplitude - 3*std_amplitude, mean_amplitude + 3*std_amplitude
    x = np.linspace(f_lower, f_upper, 50)
    y = np.linspace(a_lower, a_upper, 50)
    pdf_x = np.ones((len(y), len(x))) * x
    pdf_y = (np.ones_like(pdf_x) * y).T
    pdf_all = (np.vstack((pdf_x.flatten(), pdf_y.flatten())).T)
    dist = stats.multivariate_normal((mean_frequency, mean_amplitude), cov=cov)
    pdf_values = dist.pdf(pdf_all).reshape(pdf_x.shape)
    cmap = cm.get_cmap("Reds").copy()
    cmap.set_under("white")
    pdf_x = pdf_x if distribution == "normal" else np.exp(pdf_x)
    pdf_y = pdf_y if distribution == "normal" else np.exp(pdf_y)
    default_contour_kwargs = dict(levels=3, cmap=cmap, vmin=0.001,
                                  linewidths=0.8, zorder=7)
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
    ax.contour(pdf_x, pdf_y, pdf_values, **contour_kwargs)


def plot_single_panel_hvsr_curves(hvsr,
                                  distribution_mc="lognormal",
                                  distribution_fn="lognormal",
                                  plot_mc=True,
                                  plot_fn=True,
                                  ax=None,
                                  ): # pragma: no cover
    """Plot valid & invalid HVSR windows with curve & resonance statistics."""
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(figsize=(3.75, 2.5), dpi=150)

    # individual hvsr curves - valid
    _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=True)

    # individual hvsr curves - invalid
    _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=False)

    if plot_mc:
        # mean hvsr curve
        _plot_mean_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc)

        # +/- 1 std hvsr curve
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc, n=+1)
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc, n=-1, plot_kwargs=dict(label=None))

    # +/- std fn
    if plot_fn:
        _plot_nth_std_frequency_range(ax=ax, hvsr=hvsr, distribution=distribution_fn, n=+1)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    ax.legend(loc="upper right")

    if ax_was_none:
        return (fig, ax)
    else:
        return ax


# def plot_single_panel_hvsr_median(hvsr,
#                                   distribution_mc="lognormal",
#                                   ax=None,
#                                   median_hvsr_width=1.3,
#                                   ylims=None
#                                   ):
#     ax_was_none = False
#     if ax is None:
#         ax_was_none = True
#         fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

#     # Mean Curve
#     if np.sum(~hvsr.rejected_window_boolean_mask) > 0:
#         label = r"$LM_{curve}$" if distribution_mc == "lognormal" else "Mean"
#         ax.plot(hvsr.frequency,
#                 hvsr.mean_curve(distribution_mc),
#                 color="black",
#                 linewidth=median_hvsr_width,
#                 label=label)

#     # Mean +/- Curve
#     if np.sum(~hvsr.rejected_window_boolean_mask) > 1:
#         label = r"$LM_{curve}$" + \
#             " ± 1 STD" if distribution_mc == "lognormal" else "Mean ± 1 STD"
#         ax.plot(hvsr.frequency, hvsr.nstd_curve(-1, distribution_mc),
#                 color="black",
#                 linewidth=median_hvsr_width,
#                 linestyle='--',
#                 label=label,
#                 zorder=3)

#         ax.plot(hvsr.frequency, hvsr.nstd_curve(+1, distribution_mc),
#                 color="black",
#                 linewidth=median_hvsr_width,
#                 linestyle='--',
#                 zorder=3)

#     ax.set_xscale("log")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("HVSR Amplitude")
#     ax.legend(loc="upper right")

#     if ylims is not None:
#         ax.set_ylim(ylims)

#     if ax_was_none:
#         return (fig, ax)
#     else:
#         return ax


# def plot_single_panel_hvsr(hvsr,
#                            distribution_mc="lognormal",
#                            ax=None,
#                            individual_hvsr_width=0.3,
#                            accepted_color='#888888',
#                            rejcted_color='#00ffff',
#                            median_hvsr_width=1.3,
#                            ylims=None
#                            ):
#     """Plot Accepted """
#     ax_was_none = False
#     if ax is None:
#         ax_was_none = True
#         fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

#     plot_single_panel_hvsr_curves(hvsr=hvsr,
#                                   ax=ax,
#                                   individual_hvsr_width=individual_hvsr_width,
#                                   accepted_color=accepted_color,
#                                   rejcted_color=rejcted_color,
#                                   ylims=ylims
#                                   )

#     plot_single_panel_hvsr_median(hvsr,
#                                   distribution_mc=distribution_mc,
#                                   ax=ax,
#                                   median_hvsr_width=median_hvsr_width,
#                                   ylims=ylims
#                                   )

#     ax.set_xscale("log")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("HVSR Amplitude")
#     ax.legend(loc="upper right")

#     if ylims is not None:
#         ax.set_ylim(ylims)

#     if ax_was_none:
#         return (fig, ax)
#     else:
#         return ax


# def plot_seismic_recording_3c(srecord3c, axs=None, ylims=None):
#     """Plot SeismicRecording3C object."""
#     axs_was_none = False
#     if axs is None:
#         axs_was_none = True
#         fig, axs = plt.subplots(nrows=3, figsize=(6, 3), dpi=150, sharex=True)

#     for ax, component in zip(axs, ["ns", "ew", "vt"]):
#         tseries = getattr(srecord3c, component)
#         ax.plot(tseries.time(), tseries.amplitude, label=component, color="k", linewidth=0.5)
#         ax.legend(loc="upper right")
#         ax.set_ylabel("Amplitude")

#     axs[-1].set_xlabel("Time (s)")

#     if ylims is None:
#         ymin, ymax = 0, 0
#         for ax in axs:
#             _ymin, _ymax = ax.get_ylim()
#             ymin = min(_ymin, ymin)
#             ymax = max(_ymax, ymax)
#         ylims = (ymin, ymax)

#     for ax in axs:
#         ax.set_ylim(ylims)

#     if axs_was_none:
#         return (fig, axs)
#     else:
#         return axs


# def simple_plot(sensor, hv, windowlength, distribution_f0, distribution_mc,
#                 rejection_bool, n, n_iteration, ymin, ymax):  # pragma: no cover

#     fig = plt.figure(figsize=(6, 6), dpi=150)
#     gs = fig.add_gridspec(nrows=6, ncols=6)

#     ax0 = fig.add_subplot(gs[0:2, 0:3])
#     ax1 = fig.add_subplot(gs[2:4, 0:3])
#     ax2 = fig.add_subplot(gs[4:6, 0:3])

#     if rejection_bool:
#         ax3 = fig.add_subplot(gs[0:3, 3:6])
#         ax4 = fig.add_subplot(gs[3:6, 3:6])
#     else:
#         ax3 = fig.add_subplot(gs[0:3, 3:6])
#         ax4 = False

#     individual_width = 0.3
#     median_width = 1.3
#     for ax, title in zip([ax3, ax4], ["Before Rejection", "After Rejection"]):
#         # Rejected Windows
#         if title == "After Rejection":
#             if len(hv.rejected_window_indices) > 0:
#                 label = "Rejected"
#                 for amp in hv.amp[hv.rejected_window_indices]:
#                     ax.plot(hv.frq, amp, color='#00ffff',
#                             linewidth=individual_width, zorder=2, label=label)
#                     label = None

#         # Accepted Windows
#         label = "Accepted"
#         for amp in hv.amp[hv.valid_window_indices]:
#             ax.plot(hv.frq, amp, color='#888888', linewidth=individual_width,
#                     label=label if title == "Before Rejection" else "")
#             label = None

#         # Window Peaks
#         ax.plot(hv.peak_frq, hv.peak_amp, linestyle="", zorder=2,
#                 marker='o', markersize=2.5, markerfacecolor="#ffffff", markeredgewidth=0.5, markeredgecolor='k',
#                 label="" if title == "Before Rejection" and rejection_bool else r"$f_{0,i}$")

#         # Peak Mean Curve
#         ax.plot(hv.mc_peak_frq(distribution_mc), hv.mc_peak_amp(distribution_mc), linestyle="", zorder=4,
#                 marker='D', markersize=4, markerfacecolor='#66ff33', markeredgewidth=1, markeredgecolor='k',
#                 label="" if title == "Before Rejection" and rejection_bool else r"$f_{0,mc}$")

#         # Mean Curve
#         label = r"$LM_{curve}$" if distribution_mc == "lognormal" else "Mean"
#         ax.plot(hv.frq, hv.mean_curve(distribution_mc), color='k', linewidth=median_width,
#                 label="" if title == "Before Rejection" and rejection_bool else label)

#         # Mean +/- Curve
#         label = r"$LM_{curve}$" + \
#             " ± 1 STD" if distribution_mc == "lognormal" else "Mean ± 1 STD"
#         ax.plot(hv.frq, hv.nstd_curve(-1, distribution_mc),
#                 color='k', linestyle='--', linewidth=median_width, zorder=3,
#                 label="" if title == "Before Rejection" and rejection_bool else label)
#         ax.plot(hv.frq, hv.nstd_curve(+1, distribution_mc),
#                 color='k', linestyle='--', linewidth=median_width, zorder=3)

#         # f0 +/- STD
#         if ymin is not None and ymax is not None:
#             ax.set_ylim((ymin, ymax))
#         label = r"$LM_{f0}$" + \
#             " ± 1 STD" if distribution_f0 == "lognormal" else "Mean f0 ± 1 STD"
#         _ymin, _ymax = ax.get_ylim()
#         ax.plot([hv.mean_f0_frq(distribution_f0)]*2,
#                 [_ymin, _ymax], linestyle="-.", color="#000000")
#         ax.fill([hv.nstd_f0_frq(-1, distribution_f0)]*2 + [hv.nstd_f0_frq(+1, distribution_f0)]*2, [_ymin, _ymax, _ymax, _ymin],
#                 color="#ff8080",
#                 label="" if title == "Before Rejection" and rejection_bool else label)
#         ax.set_ylim((_ymin, _ymax))

#         ax.set_xscale('log')
#         ax.set_xlabel("Frequency (Hz)")
#         ax.set_ylabel("HVSR Amplitude")
#         if rejection_bool:
#             if title == "Before Rejection":
#                 # print("\nStatistics before rejection:")
#                 # hv.print_stats(distribution_f0)
#                 hv.reject_windows(n, max_iterations=n_iteration,
#                                   distribution_f0=distribution_f0, distribution_mc=distribution_mc)
#             elif title == "After Rejection":
#                 fig.legend(ncol=4, loc='lower center',
#                            bbox_to_anchor=(0.51, 0), columnspacing=2)

#                 # print("\nAnalysis summary:")
#                 # display(pd.DataFrame(columns=[""], index=["Window length", "No. of windows", "Number of iterations to convergence", "No. of rejected windows"],
#                 #         data=[f"{windowlength}s", str(sensor.ns.nseries), f"{c_iter} of {n_iteration} allowed", str(sum(hv.rejected_window_indices))]))
#                 # print("\nStatistics after rejection:")
#                 # hv.print_stats(distribution_f0)
#         else:
#             # display(pd.DataFrame(columns=[""], index=["Window length", "No. of windows"],
#             #                 data=[f"{windowlength}s", str(sensor.ns.nseries)]))
#             # hv.print_stats(distribution_f0)
#             fig.legend(loc="upper center", bbox_to_anchor=(0.77, 0.4))
#             break
#         ax.set_title(title)

#     norm_factor = sensor.normalization_factor
#     for ax, timerecord, name in zip([ax0, ax1, ax2], [sensor.ns, sensor.ew, sensor.vt], ["NS", "EW", "VT"]):
#         ctime = timerecord.time
#         amp = timerecord.amp/norm_factor
#         ax.plot(ctime.T, amp.T, linewidth=0.2, color='#888888')
#         ax.set_title(f"Time Records ({name})")
#         ax.set_yticks([-1, -0.5, 0, 0.5, 1])
#         ax.set_xlim(0, windowlength*timerecord.nseries)
#         ax.set_ylim(-1, 1)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Normalized Amplitude')
#         ax.plot(ctime[hv.rejected_window_indices].T,
#                 amp[hv.rejected_window_indices].T, linewidth=0.2, color="cyan")

#     if rejection_bool:
#         axs = [ax0, ax3, ax1, ax4, ax2]
#     else:
#         axs = [ax0, ax3, ax1, ax2]

#     for ax, letter in zip(axs, list("abcde")):
#         ax.text(0.97, 0.97, f"({letter})", ha="right",
#                 va="top", transform=ax.transAxes, fontsize=12)
#         for spine in ["top", "right"]:
#             ax.spines[spine].set_visible(False)

#     fig.tight_layout(h_pad=1, w_pad=2, rect=(0, 0.08, 1, 1))

#     return (fig, axs)


# def azimuthal_plot(hv, distribution_f0, distribution_mc,
#                    rejection_bool, n, n_iteration, ymin, ymax):  # pragma: no cover

#     if rejection_bool:
#         hv.reject_windows(n=n, max_iterations=n_iteration,
#                           distribution_f0=distribution_f0, distribution_mc=distribution_mc)

#     azimuths = [*hv.azimuths, 180.]
#     mesh_frq, mesh_azi = np.meshgrid(hv.frq, azimuths)
#     mesh_amp = hv.mean_curves(distribution=distribution_mc)
#     mesh_amp = np.vstack((mesh_amp, mesh_amp[0]))

#     # Layout
#     fig = plt.figure(figsize=(6, 5), dpi=150)
#     gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.3,
#                           hspace=0.1, width_ratios=(1.2, 0.8))
#     ax0 = fig.add_subplot(gs[0:2, 0:1], projection='3d')
#     ax1 = fig.add_subplot(gs[0:1, 1:2])
#     ax2 = fig.add_subplot(gs[1:2, 1:2])
#     fig.subplots_adjust(bottom=0.21)

#     # Settings
#     individual_width = 0.3
#     median_width = 1.3

#     # 3D Median Curve
#     ax = ax0
#     ax.plot_surface(np.log10(mesh_frq), mesh_azi, mesh_amp, rstride=1,
#                     cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
#     for coord in list("xyz"):
#         getattr(ax, f"w_{coord}axis").set_pane_color((1, 1, 1))
#     ax.set_xticks(np.log10(np.array([0.01, 0.1, 1, 10, 100])))
#     ax.set_xticklabels(["$10^{"+str(x)+"}$" for x in range(-2, 3)])
#     ax.set_xlim(np.log10((0.1, 30)))
#     ax.view_init(elev=30, azim=245)
#     ax.dist = 12
#     ax.set_yticks(np.arange(0, 180+45, 45))
#     ax.set_ylim(0, 180)
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("Azimuth (deg)")
#     ax.set_zlabel("HVSR Amplitude")
#     pfrqs, pamps = hv.mean_curves_peak(distribution=distribution_mc)
#     pfrqs = np.array([*pfrqs, pfrqs[0]])
#     pamps = np.array([*pamps, pamps[0]])
#     ax.scatter(np.log10(pfrqs), azimuths, pamps*1.01,
#                marker="s", c="w", edgecolors="k", s=9)

#     # 2D Median Curve
#     ax = ax1
#     contour = ax.contourf(mesh_frq, mesh_azi, mesh_amp,
#                           cmap=cm.plasma, levels=10)
#     ax.set_xscale("log")
#     ax.set_xticklabels([])
#     ax.set_ylabel("Azimuth (deg)")
#     ax.set_yticks(np.arange(0, 180+30, 30))
#     ax.set_ylim(0, 180)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("top", size="5%", pad=0.05)
#     fig.colorbar(contour, cax=cax, orientation="horizontal")
#     cax.xaxis.set_ticks_position("top")

#     ax.plot(pfrqs, azimuths, marker="s", color="w", linestyle="", markersize=3, markeredgecolor="k",
#             label=r"$f_{0,mc,\alpha}$")

#     # 2D Median Curve
#     ax = ax2

#     # Accepted Windows
#     label = "Accepted"
#     for amps in hv.amp:
#         for amp in amps:
#             ax.plot(hv.frq, amp, color="#888888",
#                     linewidth=individual_width, zorder=2, label=label)
#             label = None

#     # Mean Curve
#     label = r"$LM_{curve,AZ}$" if distribution_mc == "lognormal" else r"$Mean_{curve,AZ}$"
#     ax.plot(hv.frq, hv.mean_curve(distribution_mc), color='k',
#             label=label, linewidth=median_width, zorder=4)

#     # Mean +/- Curve
#     label = r"$LM_{curve,AZ}$" + \
#         " ± 1 STD" if distribution_mc == "lognormal" else r"$Mean_{curve,AZ}$"+" ± 1 STD"
#     ax.plot(hv.frq, hv.nstd_curve(-1, distribution=distribution_mc), color="k", linestyle="--",
#             linewidth=median_width, zorder=4, label=label)
#     ax.plot(hv.frq, hv.nstd_curve(+1, distribution=distribution_mc), color="k", linestyle="--",
#             linewidth=median_width, zorder=4)

#     # Window Peaks
#     label = r"$f_{0,i,\alpha}$"
#     for frq, amp in zip(hv.peak_frq, hv.peak_amp):
#         ax.plot(frq, amp, linestyle="", zorder=3, marker='o', markersize=2.5, markerfacecolor="#ffffff",
#                 markeredgewidth=0.5, markeredgecolor='k', label=label)
#         label = None

#     # Peak Mean Curve
#     ax.plot(hv.mc_peak_frq(distribution_mc), hv.mc_peak_amp(distribution_mc), linestyle="", zorder=5,
#             marker='D', markersize=4, markerfacecolor='#66ff33', markeredgewidth=1, markeredgecolor='k',
#             label=r"$f_{0,mc,AZ}$")

#     # f0,az
#     if ymin is not None and ymax is not None:
#         ax.set_ylim((ymin, ymax))
#     label = r"$LM_{f0,AZ}$" + \
#         " ± 1 STD" if distribution_f0 == "lognormal" else "Mean " + \
#             r"$f_{0,AZ}$"+" ± 1 STD"
#     _ymin, _ymax = ax.get_ylim()
#     ax.plot([hv.mean_f0_frq(distribution_f0)]*2, [ymin, ymax],
#             linestyle="-.", color="#000000", zorder=6)
#     ax.fill([hv.nstd_f0_frq(-1, distribution_f0)]*2 + [hv.nstd_f0_frq(+1, distribution_f0)]*2, [_ymin, _ymax, _ymax, _ymin],
#             color="#ff8080", label=label, zorder=1)
#     ax.set_ylim((_ymin, _ymax))

#     # Limits and labels
#     ax.set_xscale("log")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("HVSR Amplitude")
#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)

#     # Lettering
#     xs, ys = [0.45, 0.85, 0.85], [0.81, 0.81, 0.47]
#     for x, y, letter in zip(xs, ys, list("abc")):
#         fig.text(x, y, f"({letter})", fontsize=12)

#     # Legend
#     handles, labels = [], []
#     for ax in [ax2, ax1, ax0]:
#         _handles, _labels = ax.get_legend_handles_labels()
#         handles += _handles
#         labels += _labels
#     new_handles, new_labels = [], []
#     for index in [0, 5, 1, 2, 3, 4, 6]:
#         new_handles.append(handles[index])
#         new_labels.append(labels[index])
#     fig.legend(new_handles, new_labels, loc="lower center", bbox_to_anchor=(0.47, 0), ncol=4,
#                columnspacing=0.5, handletextpad=0.4)

#     # # Print stats
#     # print("\nStatistics after rejection considering azimuth:")
#     # hv.print_stats(distribution_f0)

#     return (fig, (ax0, ax1, ax2))


# def voronoi_plot(points, vertice_set, boundary, ax=None,
#                  fig_kwargs=None):  # pragma: no cover
#     """Plot Voronoi regions with boundary."""

#     ax_was_none = False
#     if ax is None:
#         ax_was_none = True

#         default_fig_kwargs = dict(figsize=(4, 4), dpi=150)
#         if fig_kwargs is None:
#             fig_kwargs = {}
#         fig_kwargs = {**default_fig_kwargs, **fig_kwargs}

#         fig, ax = plt.subplots(**fig_kwargs)

#     for vertices in vertice_set:
#         ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.4)

#     ax.plot(points[:, 0], points[:, 1], 'ko')
#     ax.plot(boundary[:, 0], boundary[:, 1], color="k")

#     ax.set_xlabel("Relative Northing (m)")
#     ax.set_ylabel("Relative Easting (m)")

#     if ax_was_none:
#         return fig, ax
