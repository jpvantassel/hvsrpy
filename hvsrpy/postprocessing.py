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
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import stats
import pandas as pd
from IPython.display import display

from .seismic_recording_3c import SeismicRecording3C
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .hvsr_diffuse_field import HvsrDiffuseField

__all__ = [
    "DEFAULT_KWARGS",
    "plot_single_panel_hvsr_curves",
    "plot_seismic_recordings_3c",
    "plot_pre_and_post_rejection",
    "summarize_hvsr_statistics",
    "plot_azimuthal_contour_2d",
    "plot_azimuthal_contour_3d",
    "plot_azimuthal_summary",
    "plot_voronoi",
    "summarize_spatial_statistics",
    "HVSRPY_MPL_STYLE",
]

DEFAULT_KWARGS = {
    "individual_valid_hvsr_curve": {
        "linewidth": 0.3,
        "color": "#888888",
        "label": "Accepted HVSR Curve",
    },
    "individual_invalid_hvsr_curve": {
        "linewidth": 0.3,
        # "color" : "#00ffff", # pre-v2.0.0
        "color": "lightpink",
        "label": "Rejected HVSR Curve",
    },
    "mean_hvsr_curve": {
        "linewidth": 1.3,
        "color": "black",
        "label": "Mean Curve",
    },
    "nth_std_mean_hvsr_curve": {
        "linewidth": 1.3,
        "color": "black",
        "label": r"$\pm$ 1 Std Curve",
        "linestyle":  "--",
    },
    "nth_std_frequency_range_normal": {
        "color": "#ff8080",
        "label": "r$\mu_{fn} \pm \sigma_{fn}$",
    },
    "nth_std_frequency_range_lognormal": {
        "color": "#ff8080",
        "label": r"$(\mu_{ln,fn} \pm \sigma_{ln,fn})^*$",
    },
    "peak_mean_hvsr_curve": {
        "linestyle": "",
        "marker": "D",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc}$",
    },
    "peak_mean_hvsr_curve_azimuthal": {
        "linestyle": "",
        "marker": "D",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,az}$",
    },
    "peak_mean_hvsr_curve_azimuthal_2d": {
        "linestyle": "",
        # "marker": "D",  # pre-v2.0.0
        "marker": "s",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,\alpha}$",
    },
    "peak_mean_hvsr_curve_azimuthal_3d": {
        # "marker": "D",  # pre-v2.0.0
        "marker": "s",
        "s": 16,
        "c": "lightgreen",
        "edgecolors": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,\alpha}$",
    },
    "peak_individual_valid_hvsr_curve": {
        "linestyle": "",
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "white",
        "markeredgewidth": 0.5,
        "markeredgecolor": "black",
        "zorder": 2,
        "label": r"$f_{n,i,accepted}$",
    },
    "peak_individual_invalid_hvsr_curve": {
        "linestyle": "",
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "lightpink",
        "markeredgewidth": 0.5,
        "markeredgecolor": "white",
        "zorder": 2,
        "label": r"$f_{n,i,rejected}$",
    }
}

HVSRPY_MPL_STYLE = {
    "axes.titlesize": 8,
    "lines.linewidth": 0.75,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "font.size": 8,
    "legend.handlelength": 1.5,
    "legend.columnspacing": 0.5,
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.framealpha": 1,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _plot_individual_hvsr_curves(ax, hvsr, valid=True, plot_kwargs=None):  # pragma: no cover
    """Plot individual HVSR curves.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        msg = "Can only plot valid HVSR curves from HvsrTraditional "
        msg += "and HvsrAzimuthal objects."
        raise NotImplementedError(msg)

    if valid:
        default_kwargs = DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy()
    else:
        default_kwargs = DEFAULT_KWARGS["individual_invalid_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {
        **default_kwargs, **plot_kwargs}

    for hvsr in hvsrs:
        to_plot = hvsr.valid_window_boolean_mask if valid else ~hvsr.valid_window_boolean_mask
        for amplitude in hvsr.amplitude[to_plot]:
            ax.plot(hvsr.frequency, amplitude, **plot_kwargs)
            plot_kwargs["label"] = None


def _plot_peak_individual_hvsr_curve(ax, hvsr, valid=True, plot_kwargs=None):  # pragma: no cover
    """Plot peaks of individual HVSR curves.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        msg = "Can only plot valid HVSR curves from HvsrTraditional "
        msg += "and HvsrAzimuthal objects."
        raise NotImplementedError(msg)

    if valid:
        default_kwargs = DEFAULT_KWARGS["peak_individual_valid_hvsr_curve"].copy(
        )
    else:
        default_kwargs = DEFAULT_KWARGS["peak_individual_invalid_hvsr_curve"].copy(
        )
    plot_kwargs = default_kwargs if plot_kwargs is None else {
        **default_kwargs, **plot_kwargs}

    for hvsr in hvsrs:
        to_plot = hvsr.valid_peak_boolean_mask if valid else ~hvsr.valid_peak_boolean_mask
        frequency, amplitude = hvsr._main_peak_frq[to_plot], hvsr._main_peak_amp[to_plot]
        # could be a case where first azimuth was completely rejected.
        if len(frequency) > 0:
            ax.plot(frequency, amplitude, **plot_kwargs)
            plot_kwargs["label"] = None


def _plot_peak_mean_hvsr_curve(ax, hvsr, distribution="lognormal", plot_kwargs=None):  # pragma: no cover
    """Plot peak of mean HVSR curve.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrAzimuthal):
        default_kwargs = DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal"].copy(
        )
    else:
        default_kwargs = DEFAULT_KWARGS["peak_mean_hvsr_curve"].copy()

    if plot_kwargs is None:
        plot_kwargs = default_kwargs
    else:
        plot_kwargs = {**default_kwargs, **plot_kwargs}

    ax.plot(*hvsr.mean_curve_peak(distribution=distribution), **plot_kwargs)


def _plot_mean_hvsr_curve(ax, hvsr, distribution="lognormal", plot_kwargs=None):  # pragma: no cover
    """Plot mean HVSR curve.

    .. warning::
        Private methods are subject to change without warning.

    """
    default_kwargs = DEFAULT_KWARGS["mean_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {
        **default_kwargs, **plot_kwargs}
    ax.plot(hvsr.frequency, hvsr.mean_curve(
        distribution=distribution), **plot_kwargs)


def _plot_nth_std_hvsr_curve(ax, hvsr, distribution="lognormal", n=1., plot_kwargs=None):  # pragma: no cover
    """Plot nth standard deviation HVSR curve.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrDiffuseField):
        return None

    default_kwargs = DEFAULT_KWARGS["nth_std_mean_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {
        **default_kwargs, **plot_kwargs}
    ax.plot(hvsr.frequency, hvsr.nth_std_curve(
        n=n, distribution=distribution), **plot_kwargs)


def _plot_nth_std_frequency_range(ax, hvsr, distribution="lognormal", n=1., fill_kwargs=None):  # pragma: no cover
    """Plot nth standard deviation frequency.

    .. warning::
        Private methods are subject to change without warning.

    """
    if isinstance(hvsr, HvsrDiffuseField):
        return None

    default_kwargs = DEFAULT_KWARGS[f"nth_std_frequency_range_{distribution}"].copy(
    )
    fill_kwargs = default_kwargs if fill_kwargs is None else {
        **default_kwargs, **fill_kwargs}
    _, y_max = ax.get_ylim()
    f_min = hvsr.nth_std_fn_frequency(n=-n, distribution=distribution)
    f_max = hvsr.nth_std_fn_frequency(n=+n, distribution=distribution)
    ax.fill([f_min, f_min, f_max, f_max],
            [0, 100, 100, 0], **fill_kwargs)
    ax.set_ylim((0, np.ceil(y_max)))


def _plot_resonance_pdf(ax, hvsr, distribution="lognormal", contour_kwargs=None):  # pragma: no cover
    """Plot resonance probability density function (PDF).

    .. warning::
        Private methods are subject to change without warning.

    """
    # define bivariate (log)normal.
    mean_frequency = hvsr.mean_fn_frequency(distribution=distribution)
    mean_frequency = mean_frequency if distribution == "normal" else np.log(
        mean_frequency)
    mean_amplitude = hvsr.mean_fn_amplitude(distribution=distribution)
    mean_amplitude = mean_amplitude if distribution == "normal" else np.log(
        mean_amplitude)
    cov = hvsr.cov_fn(distribution=distribution)
    std_frequency = np.sqrt(cov[0, 0])
    std_amplitude = np.sqrt(cov[1, 1])

    # plot pdf.
    f_lower, f_upper = mean_frequency - 3 * \
        std_frequency, mean_frequency + 3*std_frequency
    a_lower, a_upper = mean_amplitude - 3 * \
        std_amplitude, mean_amplitude + 3*std_amplitude
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
    default_contour_kwargs = dict(levels=5, cmap=cmap, vmin=0.001,
                                  linewidths=0.8, zorder=7)
    if contour_kwargs is None:
        contour_kwargs = {}
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}
    ax.contour(pdf_x, pdf_y, pdf_values, **contour_kwargs)


def plot_single_panel_hvsr_curves(hvsr,
                                  distribution_mc="lognormal",
                                  distribution_fn="lognormal",
                                  plot_valid_curves=True,
                                  plot_invalid_curves=False,
                                  plot_mean_curve=True,
                                  plot_frequency_std=True,
                                  plot_peak_mean_curve=True,
                                  plot_peak_individual_valid_curves=True,
                                  plot_peak_individual_invalid_curves=False,
                                  ax=None,
                                  subplots_kwargs=None,
                                  ):  # pragma: no cover
    """Plot valid & invalid HVSR windows with curve & resonance statistics."""
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_subplots_kwargs = dict(figsize=(3.75, 2.5), dpi=150)
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, ax = plt.subplots(**subplots_kwargs)

    if plot_valid_curves:
        # individual hvsr curves - valid
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=True)

    if plot_invalid_curves:
        # individual hvsr curves - invalid
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=False)

    if plot_mean_curve:
        # mean hvsr curve
        _plot_mean_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc)

        # +/- 1 std hvsr curve
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr,
                                 distribution=distribution_mc, n=+1)
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr,
                                 distribution=distribution_mc, n=-1,
                                 plot_kwargs=dict(label=None))

    if plot_frequency_std:
        # +/- std fn
        _plot_nth_std_frequency_range(ax=ax, hvsr=hvsr,
                                      distribution=distribution_fn, n=+1)

    if plot_peak_mean_curve:
        # peak mean hvsr curves
        _plot_peak_mean_hvsr_curve(ax=ax, hvsr=hvsr,
                                   distribution=distribution_mc)

    if plot_peak_individual_valid_curves:
        # plot peak of individual valid hvsr curves
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=True)

    if plot_peak_individual_invalid_curves:
        # plot peak of individual invalid hvsr curves
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=False)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    ax.legend(loc="upper right")

    if ax_was_none:
        return (fig, ax)
    else:
        return ax


def plot_seismic_recordings_3c(srecords,
                               valid_window_boolean_mask=None,
                               axs=None,
                               subplots_kwargs=None,
                               normalize=True
                               ):  # pragma: no cover
    """Plot iterable of SeismicRecording3C objects in three panels."""
    axs_was_none = False
    if axs is None:
        axs_was_none = True
        default_subplots_kwargs = dict(nrows=3, figsize=(4, 4.5), dpi=150,
                                       sharex=True, sharey=True,
                                       gridspec_kw=dict(hspace=0.5))
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, axs = plt.subplots(**subplots_kwargs)

    if len(axs) != 3:
        raise ValueError(f"axs is of length {len(axs)}, must be 3.")

    if isinstance(srecords, SeismicRecording3C):
        srecords = [srecords]

    if valid_window_boolean_mask is None:
        valid_window_boolean_mask = [True]*len(srecords)

    if len(valid_window_boolean_mask) != len(srecords):
        msg = "length of valid_window_boolean_mask "
        msg += f"({len(valid_window_boolean_mask)}) must match the length "
        msg += f"of srecords ({len(srecords)})."
        raise ValueError(msg)

    if normalize:
        # normalization factor
        normalization_factor = 0.
        for component in ["ns", "ew", "vt"]:
            for srecord in srecords:
                c_max = np.max(np.abs(getattr(srecord, component).amplitude))
                if c_max > normalization_factor:
                    normalization_factor = c_max
    else:
        normalization_factor = 1.

    for ax, component in zip(axs, ["ns", "ew", "vt"]):
        start_time = 0.
        for srecord, valid in zip(srecords, valid_window_boolean_mask):
            tseries = getattr(srecord, component)
            time = tseries.time()+start_time

            if valid:
                default_kwargs = DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy(
                )
            else:
                default_kwargs = DEFAULT_KWARGS["individual_invalid_hvsr_curve"].copy(
                )
            default_kwargs["label"] = None

            ax.plot(time,
                    tseries.amplitude / normalization_factor,
                    **default_kwargs)
            start_time = time[-1]

        ax.set_title(f"{component.upper()} Recording")
        ax.set_ylabel(
            "Normalized\nAmplitude" if normalize else "Amplitude\n(Counts)")
        ax.set_xlim(0, time[-1])

    axs[-1].set_xlabel("Time (s)")

    if normalize:
        for ax in axs:
            ax.set_ylim(-1, 1)

    if axs_was_none:
        return (fig, axs)
    else:
        return axs


def plot_pre_and_post_rejection(srecords,
                                hvsr,
                                distribution_mc="lognormal",
                                distribution_fn="lognormal"
                                ):  # pragma: no cover
    """Plot pre- and post-rejection."""

    if not isinstance(hvsr, HvsrTraditional):
        raise NotImplementedError("Can only plot HvsrTraditional results.")

    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    gs = fig.add_gridspec(nrows=6, ncols=6)

    ax0 = fig.add_subplot(gs[0:2, 0:3])
    ax1 = fig.add_subplot(gs[2:4, 0:3])
    ax2 = fig.add_subplot(gs[4:6, 0:3])
    ax3 = fig.add_subplot(gs[0:3, 3:6])
    ax4 = fig.add_subplot(gs[3:6, 3:6])

    # plot waveforms
    plot_seismic_recordings_3c(
        srecords,
        valid_window_boolean_mask=hvsr.valid_window_boolean_mask,
        axs=(ax0, ax1, ax2)
    )

    # plot pre-rejection
    ax = ax3
    store_valid_window_boolean_mask = np.array(hvsr.valid_window_boolean_mask)
    store_valid_peak_boolean_mask = np.array(hvsr.valid_peak_boolean_mask)
    hvsr.valid_window_boolean_mask = np.full_like(
        store_valid_window_boolean_mask, True)
    hvsr.valid_peak_boolean_mask = np.full_like(
        store_valid_peak_boolean_mask, True)
    plot_single_panel_hvsr_curves(hvsr,
                                  distribution_mc=distribution_mc,
                                  distribution_fn=distribution_fn,
                                  plot_peak_individual_valid_curves=True,
                                  plot_peak_mean_curve=True,
                                  ax=ax,
                                  )
    ax.set_title("Before Rejection")
    ax.get_legend().remove()

    # plot post-rejection
    ax = ax4
    hvsr.valid_window_boolean_mask = store_valid_window_boolean_mask
    hvsr.valid_peak_boolean_mask = store_valid_peak_boolean_mask
    plot_single_panel_hvsr_curves(hvsr,
                                  distribution_mc=distribution_mc,
                                  distribution_fn=distribution_fn,
                                  plot_peak_mean_curve=True,
                                  plot_invalid_curves=True,
                                  plot_peak_individual_valid_curves=True,
                                  plot_peak_individual_invalid_curves=True,
                                  ax=ax,
                                  )
    ax4.get_legend().remove()
    ax.set_title("After Rejection")

    axs = (ax0, ax3, ax1, ax4, ax2)
    for ax, letter in zip(axs, list("abcde")):
        text = ax.text(0.02, 0.97, f"({letter})",
                       ha="left", va="top", transform=ax.transAxes)
        text.set_bbox(dict(facecolor='white', edgecolor='none',
                           boxstyle='round', pad=0.15))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=1, w_pad=1, rect=(0, 0.05, 1, 1))

    # add legend after tight layout, otherwise user warning.
    ax4.legend(loc="upper center", bbox_to_anchor=(-0.2, -0.23), ncols=4)

    return (fig, axs)


def summarize_hvsr_statistics(hvsr,
                              distribution_mc="lognormal",
                              distribution_fn="lognormal"
                              ):   # pragma: no cover

    if isinstance(hvsr, (HvsrTraditional, HvsrAzimuthal)):
        if distribution_fn == "lognormal":
            columns = ["Exponentitated Lognormal Median (units)",
                       "Lognormal Standard Deviation (log units)",
                       "-1 Lognormal Standard Deviation (units)",
                       "+1 Lognormal Standard Deviation (units)"
                       ]
            data = np.array([
                [
                    hvsr.mean_fn_frequency(distribution=distribution_fn),
                    hvsr.std_fn_frequency(distribution=distribution_fn),
                    hvsr.nth_std_fn_frequency(-1,
                                              distribution=distribution_fn),
                    hvsr.nth_std_fn_frequency(+1,
                                              distribution=distribution_fn),
                ],
                [
                    1/hvsr.mean_fn_frequency(distribution=distribution_fn),
                    hvsr.std_fn_frequency(distribution=distribution_fn),
                    1/hvsr.nth_std_fn_frequency(-1,
                                                distribution=distribution_fn),
                    1/hvsr.nth_std_fn_frequency(+1,
                                                distribution=distribution_fn),
                ],
                [
                    hvsr.mean_fn_amplitude(distribution=distribution_fn),
                    hvsr.std_fn_amplitude(distribution=distribution_fn),
                    hvsr.nth_std_fn_amplitude(-1,
                                              distribution=distribution_fn),
                    hvsr.nth_std_fn_amplitude(+1,
                                              distribution=distribution_fn),
                ],
            ])

        elif distribution_fn == "normal":
            columns = ["Mean (units)", "Standard Deviation (units)",
                       "-1 Standard Deviation (units)", "+1 Standard Deviation (units)"]
            data = np.array([
                [
                    hvsr.mean_fn_frequency(distribution=distribution_fn),
                    hvsr.std_fn_frequency(distribution=distribution_fn),
                    hvsr.nth_std_fn_frequency(-1,
                                              distribution=distribution_fn),
                    hvsr.nth_std_fn_frequency(+1,
                                              distribution=distribution_fn),
                ],
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    hvsr.mean_fn_amplitude(distribution=distribution_fn),
                    hvsr.std_fn_amplitude(distribution=distribution_fn),
                    hvsr.nth_std_fn_amplitude(-1,
                                              distribution=distribution_fn),
                    hvsr.nth_std_fn_amplitude(+1,
                                              distribution=distribution_fn),
                ],
            ])

        df = pd.DataFrame(data=data, columns=columns,
                          index=["Resonant Site Frequency, fn (Hz)",
                                 "Resonant Site Period, Tn (s)",
                                 "Resonance Amplitude, An"])
        mc_f, mc_a = hvsr.mean_curve_peak(distribution=distribution_mc)
        caption = f"The peak of the mean curve is at {mc_f:.3f} Hz with amplitude {mc_a:.3f}."

        s = df.style.format(precision=3)
        s = s.set_caption(caption)\
            .set_table_styles([{
                'selector': 'caption',
                'props': 'caption-side: bottom; font-size:1.25em;'
            }], overwrite=False)
        with pd.option_context('display.max_colwidth', None):
            display(s)
    elif isinstance(hvsr, HvsrDiffuseField):
        mc_f, mc_a = hvsr.mean_curve_peak(distribution=distribution_mc)
        caption = f"The peak of the mean curve is at {mc_f:.3f} Hz with amplitude {mc_a:.3f}."
        print(caption)
    else:
        raise NotImplementedError


def _azimuthal_mesh_from_hvsr(hvsr, distribution_mc="lognormal"):  # pragma: no cover
    azimuths = [*hvsr.azimuths, 180.]
    mesh_frq, mesh_azi = np.meshgrid(hvsr.frequency, azimuths)
    mesh_amp = hvsr.mean_curve_by_azimuth(distribution=distribution_mc)
    mesh_amp = np.vstack((mesh_amp, mesh_amp[0]))
    return mesh_frq, mesh_azi, mesh_amp


def plot_azimuthal_contour_2d(hvsr,
                              distribution_mc="lognormal",
                              plot_mean_curve_peak_by_azimuth=True,
                              fig=None,
                              ax=None,
                              subplots_kwargs=None,
                              contourf_kwargs=None):  # pragma: no cover
    if not isinstance(hvsr, HvsrAzimuthal):
        raise NotImplementedError("Can only plot HvsrAzimuthal results.")

    # define data
    mesh_frq, mesh_azi, mesh_amp = _azimuthal_mesh_from_hvsr(hvsr,
                                                             distribution_mc=distribution_mc)

    # define axes
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_subplots_kwargs = dict(figsize=(3.75, 3), dpi=150)
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, ax = plt.subplots(**subplots_kwargs)

    # contourf plot
    default_contourf_kwargs = dict(cmap=cm.plasma, levels=10)
    if contourf_kwargs is None:
        contourf_kwargs = {}
    contourf_kwargs = {**default_contourf_kwargs, **contourf_kwargs}
    contour = ax.contourf(mesh_frq, mesh_azi, mesh_amp, **contourf_kwargs)

    # axes formatting
    ax.set_xscale("log")
    ax.set_xlim(hvsr.frequency[0], hvsr.frequency[-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_yticks(np.arange(0, 180+30, 30))
    ax.set_ylim(0, 180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    if np.max(mesh_amp) < 6.5:
        ticks = np.arange(0, 7, 1)
    elif np.max(mesh_amp) < 14:
        ticks = np.arange(0, 16, 2)
    else:
        ticks = np.arange(0, (np.max(mesh_amp)//5+1)*5, 5)
    plt.colorbar(contour, cax=cax, orientation="horizontal", ticks=ticks)
    cax.xaxis.set_ticks_position("top")

    if plot_mean_curve_peak_by_azimuth:
        # mean curve peaks
        fpeak, _ = hvsr.mean_curve_peak_by_azimuth(
            distribution=distribution_mc)
        ax.plot(fpeak,
                hvsr.azimuths,
                **DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal_2d"])
        ax.legend()

    if ax_was_none:
        return (fig, (ax, cax))


def plot_azimuthal_contour_3d(hvsr,
                              distribution_mc="lognormal",
                              ax=None,
                              plot_mean_curve_peak_by_azimuth=True,
                              camera_elevation=35,
                              camera_azimuth=250,
                              camera_distance=13
                              ):  # pragma: no cover
    # layout
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig = plt.figure(figsize=(3.75, 5), dpi=150)
        ax = fig.add_subplot(projection="3d")

    # define data
    mesh_frq, mesh_azi, mesh_amp = _azimuthal_mesh_from_hvsr(hvsr,
                                                             distribution_mc=distribution_mc)

    # 3d median Curve
    ax.plot_surface(np.log10(mesh_frq), mesh_azi, mesh_amp, rstride=1,
                    cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
    for coord in list("xyz"):
        getattr(ax, f"{coord}axis").pane.fill = False
        getattr(ax, f"{coord}axis").pane.set_edgecolor('white')
    ax.set_xticks(np.log10(np.array([0.01, 0.1, 1, 10, 100])))
    ax.set_xticklabels(["$10^{"+str(x)+"}$" for x in range(-2, 3)])
    ax.set_xlim(np.log10((hvsr.frequency[0], hvsr.frequency[-1])))
    ax.view_init(elev=camera_elevation, azim=camera_azimuth)
    ax.dist = camera_distance
    ax.set_yticks(np.arange(0, 180+45, 45))
    ax.set_ylim(0, 180)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_zlabel("HVSR Amplitude")

    if plot_mean_curve_peak_by_azimuth:
        # mean curve peaks
        fpeak, apeak = hvsr.mean_curve_peak_by_azimuth(
            distribution=distribution_mc)
        fpeak = np.array([*fpeak, fpeak[0]])
        azimuths = np.array([*hvsr.azimuths, 180.])
        apeak = np.array([*apeak, apeak[0]])
        ax.scatter(np.log10(fpeak), azimuths, apeak*1.05,
                   **DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal_3d"])
        ax.legend()

    if ax_was_none:
        return (fig, ax)


def plot_azimuthal_summary(hvsr,
                           distribution_mc="lognormal",
                           distribution_fn="lognormal",
                           plot_mean_curve_peak_by_azimuth=True,
                           plot_valid_curves=True,
                           plot_invalid_curves=False,
                           plot_mean_curve=True,
                           plot_frequency_std=True,
                           plot_peak_mean_curve=True,
                           plot_peak_individual_valid_curves=True,
                           plot_peak_individual_invalid_curves=False,
                           ):  # pragma: no cover

    fig = plt.figure(figsize=(6, 5), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=2, wspace=0.3,
                          hspace=0.2, width_ratios=(1.2, 0.8))
    ax0 = fig.add_subplot(gs[0:3, 0:1], projection='3d')
    ax1 = fig.add_subplot(gs[0:2, 1:2])
    ax2 = fig.add_subplot(gs[2:4, 1:2])
    fig.subplots_adjust(bottom=0.21)

    # plot 3d contour
    ax = ax0
    plot_azimuthal_contour_3d(
        hvsr,
        distribution_mc=distribution_mc,
        ax=ax,
        plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth
    )

    # plot 2d contour
    ax = ax1
    plot_azimuthal_contour_2d(
        hvsr,
        distribution_mc=distribution_mc,
        plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth,
        ax=ax
    )
    ax.set_xlabel("")
    ax.set_xticks([])

    # plot traditional
    ax = ax2
    plot_single_panel_hvsr_curves(
        hvsr,
        distribution_mc=distribution_mc,
        distribution_fn=distribution_fn,
        plot_valid_curves=plot_valid_curves,
        plot_invalid_curves=plot_invalid_curves,
        plot_mean_curve=plot_mean_curve,
        plot_frequency_std=plot_frequency_std,
        plot_peak_mean_curve=plot_mean_curve,
        plot_peak_individual_valid_curves=plot_peak_individual_valid_curves,
        plot_peak_individual_invalid_curves=plot_peak_individual_invalid_curves,
        ax=ax
    )
    ax.get_legend().remove()
    ax.legend(loc="lower left", bbox_to_anchor=(-1.9, -0.1), ncols=2)

    if plot_peak_mean_curve:
        _plot_peak_mean_hvsr_curve(ax=ax,
                                   hvsr=hvsr,
                                   distribution=distribution_mc)

    # lettering
    xs, ys = [0.15, 0.64, 0.64], [0.83, 0.83, 0.50]
    for x, y, letter in zip(xs, ys, list("abc")):
        text = fig.text(x, y, f"({letter})")
        text.set_bbox(dict(facecolor='white', edgecolor='none',
                           boxstyle='round', pad=0.15))

    return (fig, (ax0, ax1, ax2))


def plot_voronoi(valid_sensor_coordinates,
                 valid_mean_fn,
                 tesselation_vertices,
                 boundary,
                 ax=None,
                 fig_kwargs=None):  # pragma: no cover
    """Plot Voronoi regions with boundary."""

    ax_was_none = False
    if ax is None:
        ax_was_none = True

        default_fig_kwargs = dict(figsize=(3.5, 3.5), dpi=150)
        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**default_fig_kwargs, **fig_kwargs}

        fig, ax = plt.subplots(**fig_kwargs)

    # fn colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = cm.colors.Normalize(vmin=np.min(valid_mean_fn),
                               vmax=np.max(valid_mean_fn))
    cmap = cm.autumn
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                              label="Resonant Frequency (Hz)")

    # plot tesselations
    for _dat, vertices in zip(valid_mean_fn, tesselation_vertices):
        ax.fill(vertices[:, 0], vertices[:, 1],
                facecolor=mpl.colors.rgb2hex(cmap(norm(_dat))[:3]),
                edgecolor="black", linewidth=0.5)

    # plot valid sensor coordinates
    ax.plot(valid_sensor_coordinates[:, 0], valid_sensor_coordinates[:, 1],
            markerfacecolor="cornflowerblue",
            marker="o",
            linestyle="",
            markeredgecolor="black",
            label="Sensor Location")

    # plot boundary
    closed_boundary = np.vstack((boundary, boundary[0, :]))
    ax.plot(closed_boundary[:, 0], closed_boundary[:, 1],
            color="black", label="Boundary", linewidth=3)

    ax.set_xlabel("Relative Easting (m)")
    ax.set_ylabel("Relative Northing (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    if ax_was_none:
        return fig, ax


def summarize_spatial_statistics(spatial_mean,
                                 spatial_stddev,
                                 spatial_distribution,
                                 ):  # pragma: no cover

    if spatial_distribution == "lognormal":
        data = np.array([
            [spatial_mean,
             spatial_stddev,
             np.exp(np.log(spatial_mean) - spatial_stddev),
             np.exp(np.log(spatial_mean) + spatial_stddev),
             ],
            [1/spatial_mean,
             spatial_stddev,
             1/np.exp(np.log(spatial_mean) - spatial_stddev),
             1/np.exp(np.log(spatial_mean) + spatial_stddev),
             ],
        ])
        columns = [
            "Exponentiated Lognormal Median (units)",
            "Lognormal Standard Deviation (log units)",
            "-1 Lognormal Standard Deviation (units)",
            "+1 Lognormal Standard Deviation (units)",
        ]
    elif spatial_distribution == "normal":
        data = np.array([
            [spatial_mean,
             spatial_stddev,
             spatial_mean - spatial_stddev,
             spatial_mean + spatial_stddev,
             ],
            [np.nan,
             np.nan,
             np.nan,
             np.nan,
             ]
        ])
        columns = [
            "Mean (units)",
            "Standard Deviation (units)",
            "-1 Standard Deviation (units)",
            "+1 Standard Deviation (units)",
        ]
    else:
        msg = f"spatial_distribution={spatial_distribution} not recognized."
        raise ValueError(msg)

    df = pd.DataFrame(data=data,
                      columns=columns,
                      index=[
                          "Resonant Site Frequency, fn (Hz)",
                          "Resonant Site Period, Tn (s)",
                      ])

    s = df.style.format(precision=3)
    with pd.option_context('display.max_colwidth', None):
        display(s)
