# This file is part of hvsrpy, a Python package for horizontal-to-vertical
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

"""Collection of functions for window rejection."""

import logging

import numpy as np

from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .interact import ginput_session, plot_continue_button, is_absolute_point_in_relative_box
from .plot_tools import plot_single_panel_hvsr_curves

logger = logging.getLogger(__name__)


def sta_lta_window_rejection(records, sta_seconds, lta_seconds,
                             min_sta_lta_ratio, max_sta_lta_ratio,
                             components=("ns", "ew", "vt")):
    """Performs window rejection using STA - LTA ratio.

    Parameters
    ----------
    records: iterable of SeismicRecording3C
        Time-domain data in the form of an iterable object containing
        ``SeismicRecording3C`` objects. This data is assumed to have
        already been preprocessed using ``hvsrpy.preprocess``.
    sta_seconds : float
        Length of time used to determine the short term average (STA)
        in seconds.
    lta_seconds : float
        Length of time used to determine the long term average (LTA)
        in seconds.
    min_sta_lta_ratio : float
        Minimum allowable ratio of STA/LTA for a window; windows with
        an STA/LTA below this value will be rejected.
    max_sta_lta_ratio : float
        Maximum allowable ratio of STA/LTA for a window; windows with
        an STA/LTA above this value will be rejected.
    components : iterable, optional
        Components on which the STA/LTA filter will be evaluated,
        default is all components, that is, ``("ns", "ew", "vt")``.

    Returns
    -------
    list of SeismicRecordings3C
        List of recordings with those that violate the STA/LTA ratio
        limits removed.   

    """
    passing_records = []
    for record in records:
        for component in components:
            timeseries = getattr(record, component)
            n_samples = timeseries.n_samples

            # compute sta values.
            npts_in_sta = int(sta_seconds // timeseries.dt_in_seconds)
            if npts_in_sta > n_samples:
                msg = "sta_seconds must be shorter than record length;"
                msg += f"sta_seconds is {sta_seconds} and "
                msg += f"record length is {timeseries.time()[-1]}."
                raise IndexError(msg)
            n_sta_in_window = int(timeseries.n_samples // npts_in_sta)
            short_timeseries = timeseries.amplitude[:npts_in_sta*n_sta_in_window]
            sta_values = np.mean(np.abs(short_timeseries.reshape((n_sta_in_window, npts_in_sta))), axis=1)

            # compute lta.
            npts_in_lta = int(lta_seconds // timeseries.dt_in_seconds)
            if npts_in_lta > n_samples:
                msg = "lta_seconds must be shorter than record length;"
                msg += f"lta_seconds is {lta_seconds} and "
                msg += f"record length is {timeseries.time()[-1]}."
                raise IndexError(msg)
            lta = np.mean(np.abs(short_timeseries[:npts_in_lta]))

            # check mininum and maximum sta - lta ratio.
            if (np.max(sta_values/lta) > max_sta_lta_ratio) or (np.min(sta_values/lta) < min_sta_lta_ratio):
                break
        else:
            passing_records.append(record)

    return passing_records

# TODO(jpv): Write tests for maximum_value_window_rejection.
def maximum_value_window_rejection(records, maximum_value_threshold,
                                   normalized=True, components=("ns", "ew", "vt")): # pragma: no cover
    """Performs window rejection based on maximum value of time series.

    Parameters
    ----------
    records: iterable of SeismicRecording3C
        Time-domain data in the form of an iterable object containing
        ``SeismicRecording3C`` objects. This data is assumed to have
        already been preprocessed using ``hvsrpy.preprocess``.
    maximum_value_threshold : float
        Absolute value of timeseries, that if exceeded, the record will
        be rejected. Can be relative or absolute, see parameter
        ``normalized``.
    normalized : bool, optional
        Defines whether the ``maximum_value_threshold`` is absolute or
        relative to the maximum value observed across all records and
        all components, default is ``True`` indicating the
        ``maximum_value_threshold`` is relative to the maximum value
        observed.
    components : iterable, optional
        Components on which the maximum value filter will be evaluated,
        default is all components, that is, ``("ns", "ew", "vt")``.

    Returns
    -------
    list of SeismicRecordings3C
        List of ``SeismicRecording3C`` objects with those that exceed
        the maximum value threshold removed.

    """
    # determine the maximum value for each record, across all components.
    maximum_values = np.empty(len(records))
    for idx, record in enumerate(records):
        maximum_value = 0
        for component in components:
            timeseries = getattr(record, component)
            component_max = np.max(np.abs(timeseries.amplitude))
            if component_max > maximum_value:
                maximum_value = component_max
        maximum_values[idx] = maximum_value

    # normalize maximums, if applicable.
    if normalized:
        maximum_values /= np.max(np.abs(maximum_values))

    # remove records that exceed threshold.
    passing_records = []
    for record, maximum_value in zip(records, maximum_values):
        if maximum_value < maximum_value_threshold:
            passing_records.append(record)

    return passing_records


def frequency_domain_window_rejection(hvsr,
                                      n=2,
                                      max_iterations=50,
                                      distribution_fn="lognormal",
                                      distribution_mc="lognormal",
                                      search_range_in_hz=(None, None),
                                      find_peaks_kwargs=None):
    """Frequency-domain window rejection algorithm by Cox et al., (2020).

    Parameters
    ----------
    hvsr : HvsrTraditional or HvsrAzimuthal
        HVSR object on which the window rejection algorithm will
        be applied.
    n : float, optional
        Tuning parameter of the Cox et al., (2020) algorithm, indicates
        the number of standard deviations from the mean to be removed,
        default value is ``2``.
    max_iterations : int, optional
        Maximum number of rejection iterations, default value is
        ``50``.
    distribution_fn : {"lognormal", "normal"}, optional
        Assumed distribution of ``fn`` from HVSR curves, the
        default is ``"lognormal"``.
    distribution_mc : {"lognormal", "normal"}, optional
        Assumed distribution of mean curve, the default is
        ``"lognormal"``.
    search_range_in_hz : tuple, optional
        Frequency range to be searched for peaks.
        Half open ranges can be specified with ``None``, default is
        ``(None, None)`` indicating the full frequency range will be
        searched.
    find_peaks_kwargs : dict
        Keyword arguments for the ``scipy`` function
        `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
        see ``scipy`` documentation for details, default is ``None``
        indicating defaults will be used.

    Returns
    -------
    int
        Number of iterations required for convergence.

    """
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else: # pragma: no cover
        msg = "The frequency domain window rejection algorithm can only "
        msg += "be applied to HvsrTraditional and HvsrAzimuthal objects, not "
        msg += f"{type(hvsr)} type objects."
        raise NotImplementedError(msg)

    hvsr.meta["performed window rejection"] = "FDWRA by Cox et al. (2020)"
    hvsr.meta["window rejection algorithm arguments"] = dict(n=n,
                                                             max_iterations=max_iterations,
                                                             distribution_fn=distribution_fn,
                                                             distribution_mc=distribution_mc,
                                                             search_range_in_hz=search_range_in_hz,
                                                             find_peaks_kwargs=find_peaks_kwargs)

    max_performed_iterations = 0
    for hvsr in hvsrs:
        hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz,
                                   find_peaks_kwargs=find_peaks_kwargs)
        iterations = _frequency_domain_window_rejection(hvsr=hvsr,
                                                        n=n,
                                                        max_iterations=max_iterations,
                                                        distribution_fn=distribution_fn,
                                                        distribution_mc=distribution_mc)
        if iterations > max_performed_iterations:
            max_performed_iterations = iterations

    return max_performed_iterations


def _frequency_domain_window_rejection(hvsr,
                                       n=2,
                                       max_iterations=50,
                                       distribution_fn="lognormal",
                                       distribution_mc="lognormal"):
    for c_iteration in range(1, max_iterations+1):
        logger.debug(f"c_iteration: {c_iteration}")
        logger.debug(f"valid_window_boolean_mask: {hvsr.valid_window_boolean_mask}")
        logger.debug(f"valid_peak_boolean_mask: {hvsr.valid_peak_boolean_mask}")

        mean_fn_before = hvsr.mean_fn_frequency(distribution_fn)
        std_fn_before = hvsr.std_fn_frequency(distribution_fn)
        mc_peak_frq_before, _ = hvsr.mean_curve_peak(distribution_mc)
        diff_before = abs(mean_fn_before - mc_peak_frq_before)

        logger.debug(f"\tmean_fn_before: {mean_fn_before}")
        logger.debug(f"\tstd_fn_before: {std_fn_before}")
        logger.debug(f"\tmc_peak_frq_before: {mc_peak_frq_before}")
        logger.debug(f"\tdiff_before: {diff_before}")

        lower_bound = hvsr.nth_std_fn_frequency(-n, distribution_fn)
        upper_bound = hvsr.nth_std_fn_frequency(+n, distribution_fn)

        for _idx, (c_valid, c_peak) in enumerate(zip(hvsr.valid_peak_boolean_mask, hvsr._main_peak_frq)):
            if not c_valid:
                continue

            if c_peak > lower_bound and c_peak < upper_bound:
                hvsr.valid_window_boolean_mask[_idx] = True
                hvsr.valid_peak_boolean_mask[_idx] = True
            else:
                hvsr.valid_window_boolean_mask[_idx] = False
                hvsr.valid_peak_boolean_mask[_idx] = False

        mean_fn_after = hvsr.mean_fn_frequency(distribution_fn)
        std_fn_after = hvsr.std_fn_frequency(distribution_fn)
        mc_peak_frq_after, _ = hvsr.mean_curve_peak(distribution_mc)
        d_after = abs(mean_fn_after - mc_peak_frq_after)

        logger.debug(f"\tmean_fn_after: {mean_fn_after}")
        logger.debug(f"\tstd_fn_after: {std_fn_after}")
        logger.debug(f"\tmc_peak_frq_after: {mc_peak_frq_after}")
        logger.debug(f"\td_after: {d_after}")

        if diff_before == 0 or std_fn_before == 0 or std_fn_after == 0:
            msg = f"Performed {c_iteration} iterations, returning b/c 0 values."
            logger.warning(msg)
            return c_iteration

        d_diff = abs(d_after - diff_before)/diff_before
        s_diff = abs(std_fn_after - std_fn_before)

        logger.debug(f"\td_diff: {d_diff}")
        logger.debug(f"\ts_diff: {s_diff}")

        if (d_diff < 0.01) and (s_diff < 0.01):
            msg = f"Performed {c_iteration} iterations, returning b/c rejection converged."
            logger.info(msg)
            return c_iteration

# TODO(jpv): Write tests for manual_window_rejection.
def manual_window_rejection(hvsr,
                            ylims=None,
                            distribution_fn="lognormal",
                            distribution_mc="lognormal",
                            search_range_in_hz=(None, None),
                            find_peaks_kwargs=None,
                            upper_right_corner_relative=(0.95, 0.95),
                            box_size_relative=(0.1, 0.05)): # pragma: no cover
    """Reject HVSR curves manually.

    Parameters
    ----------
    hvsr : HvsrTraditional or HvsrAzimuthal
        HVSR object on which the window rejection algorithm will
        be applied.
    ylims : tuple, optional
        Upper and lower limits of plotted HVSR amplitude, default is
        ``None`` indicating limits will automatically be selected.
    distribution_fn : {"lognormal", "normal"}, optional
        Assumed distribution of ``fn`` from HVSR curves, the
        default is ``"lognormal"``.
    distribution_mc : {"lognormal", "normal"}, optional
        Assumed distribution of mean curve, the default is
        ``"lognormal"``.
    search_range_in_hz : tuple, optional
        Frequency range to be searched for peaks.
        Half open ranges can be specified with ``None``, default is
        ``(None, None)`` indicating the full frequency range will be
        searched.
    find_peaks_kwargs : dict
        Keyword arguments for the ``scipy`` function
        `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
        see ``scipy`` documentation for details, default is ``None``
        indicating defaults will be used.

    Returns
    -------
    None
        Modifies Hvsr object's internal state.

    """
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else:
        msg = "The manual window rejection algorithm can only be "
        msg += "applied to HvsrTraditional and HvsrAzimuthal objects, "
        msg += f"not {type(hvsr)} type objects."
        raise NotImplementedError(msg)

    hvsr.meta["window rejection algorithm"] = "Manual after SESAME (2004)"
    hvsr.meta["window rejection algorithm arguments"] = dict(ylims=ylims,
                                                             distribution_fn=distribution_fn,
                                                             distribution_mc=distribution_mc,
                                                             search_range_in_hz=search_range_in_hz,
                                                             find_peaks_kwargs=find_peaks_kwargs,
                                                             upper_right_corner_relative=upper_right_corner_relative,
                                                             box_size_relative=box_size_relative)

    # update peaks of hvsr object.
    if find_peaks_kwargs is None:
        find_peaks_kwargs = {}

    hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz,
                               find_peaks_kwargs=find_peaks_kwargs)

    # plot hvsr.
    fig, ax = plot_single_panel_hvsr_curves(hvsr=hvsr,
                                            distribution_mc=distribution_mc,
                                            distribution_fn=distribution_fn)
    ax.autoscale(enable=False)
    plot_continue_button(ax,
                         upper_right_corner_relative=upper_right_corner_relative,
                         box_size_relative=box_size_relative)
    fig.show()
    if ylims is not None:
        ax.set_ylim(ylims)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    while True:
        xs, ys = ginput_session(fig, ax, initial_adjustment=False,
                                npts=2, ask_to_confirm_point=False, ask_to_continue=False)

        # only look at frequencies in drawn box to accelerate search.
        selected_columns = np.logical_and(hvsr.frequency > min(xs),
                                          hvsr.frequency < max(xs))
        was_empty = True
        for hvsr in hvsrs:
            for idx, amplitude in enumerate(hvsr.amplitude[:, selected_columns]):
                if np.any(np.logical_and(amplitude > min(ys), amplitude < max(ys))):
                    was_empty = False
                    hvsr.valid_window_boolean_mask[idx] = False
        hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz,
                                  find_peaks_kwargs=find_peaks_kwargs)

        # Clear, set axis limits, and lock axis.
        ax.clear()
        # Note: ax.clear() re-enables autoscale.
        ax.autoscale(enable=False)
        ax = plot_single_panel_hvsr_curves(hvsr=hvsr,
                                           distribution_mc=distribution_mc,
                                           distribution_fn=distribution_fn,
                                           ax=ax)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        plot_continue_button(ax,
                             upper_right_corner_relative=upper_right_corner_relative,
                             box_size_relative=box_size_relative)
        fig.canvas.draw_idle()

        if was_empty:
            in_continue_box = False
            for _x, _y in zip(xs, ys):
                if is_absolute_point_in_relative_box(ax=ax,
                                                     absolute_point=(_x, _y),
                                                     upper_right_corner_relative=upper_right_corner_relative,
                                                     box_size_relative=box_size_relative):
                    in_continue_box = True
                    break

            if in_continue_box:
                fig.close()
                break
            else:
                continue
