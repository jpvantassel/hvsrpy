# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2022 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

import numpy as np


def sta_lta_window_rejection(records,
                             sta_seconds,
                             lta_seconds,
                             min_sta_lta_ratio,
                             max_sta_lta_ratio,
                             components=("ns", "ew", "vt")
                             ):
    """Perform window rejection using STA - LTA ratio.

    Paramters
    ---------
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
        Components on which the STA/LTA filter will be performed,
        default is all components `("ns", "ew", "vt")`.

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
            nsamples = timeseries.nsamples

            # compute sta values.
            npts_in_sta = sta_seconds // timeseries.dt
            if npts_in_sta > nsamples:
                msg = "sta_seconds must be shorter than record length;"
                msg += f"sta_seconds is {sta_seconds} and "
                msg += f"record length is {timeseries.time()[-1]}."
                raise IndexError(msg)
            n_sta_in_window = timeseries.samples // npts_in_sta
            short_timeseries = timeseries.amplitude[:npts_in_sta*n_sta_in_window]
            sta_values = np.mean(short_timeseries.reshape((n_sta_in_window, npts_in_sta)), axis=1)

            # compute lta.
            npts_in_lta = lta_seconds // timeseries.dt
            if npts_in_lta > nsamples:
                msg = "lta_seconds must be shorter than record length;"
                msg += f"lta_seconds is {lta_seconds} and "
                msg += f"record length is {timeseries.time()[-1]}."
                raise IndexError(msg)
            lta = np.mean(short_timeseries[:npts_in_lta])

            # check mininum and maximum sta - lta ratio.
            if (np.max(sta_values/lta) > max_sta_lta_ratio) or (np.min(sta_values/lta) < min_sta_lta_ratio):
                break
        else:
            passing_records.append(record)

    return passing_records

# TODO(jpv): Reject windows based on maximum value in timeseries.


# TODO(jpv): Review the frequency domain window rejection algorithm


def frequency_domain_window_rejection(self, n=2, max_iterations=50,
                                      distribution_f0='lognormal',
                                      distribution_mc='lognormal',
                                      search_range_in_hz=(None, None)):
    """Frequency-domain window rejection by Cox et al., (2020).

    Parameters
    ----------
    n : float, optional
        Number of standard deviations from the mean, default
        value is 2.
    max_iterations : int, optional
        Maximum number of rejection iterations, default value is
        50.
    distribution_f0 : {'lognormal', 'normal'}, optional
        Assumed distribution of `f0` from time windows, the
        default is 'lognormal'.
    distribution_mc : {'lognormal', 'normal'}, optional
        Assumed distribution of mean curve, the default is
        'lognormal'.

    Returns
    -------
    int
        Number of iterations required for convergence.

    """
    # TODO(jpv): Review documentation here.

    self.meta["n"] = n
    self.meta["Performed Rejection"] = True

    for c_iteration in range(1, max_iterations+1):

        logger.debug(f"c_iteration: {c_iteration}")
        logger.debug(f"valid_window_boolean_mask: {self.valid_window_boolean_mask}")

        mean_f0_before = self.mean_f0_frq(distribution_f0)
        std_f0_before = self.std_f0_frq(distribution_f0)
        mc_peak_frq_before, _ = self.mean_curve_peak(distribution_mc)
        d_before = abs(mean_f0_before - mc_peak_frq_before)

        logger.debug(f"\tmean_f0_before: {mean_f0_before}")
        logger.debug(f"\tstd_f0_before: {std_f0_before}")
        logger.debug(f"\tmc_peak_frq_before: {mc_peak_frq_before}")
        logger.debug(f"\td_before: {d_before}")

        lower_bound = self.nstd_f0_frq(-n, distribution_f0)
        upper_bound = self.nstd_f0_frq(+n, distribution_f0)

        valid_indices = np.zeros(self.nseries, dtype=bool)
        for c_window, (c_valid, c_peak) in enumerate(zip(self.valid_window_boolean_mask, self._main_peak_frq)):
            if not c_valid:
                continue

            if c_peak > lower_bound and c_peak < upper_bound:
                valid_indices[c_window] = True

        self.valid_window_boolean_mask = valid_indices

        mean_f0_after = self.mean_f0_frq(distribution_f0)
        std_f0_after = self.std_f0_frq(distribution_f0)
        mc_peak_frq_after, _ = self.mean_curve_peak(distribution_mc)
        d_after = abs(mean_f0_after - mc_peak_frq_after)

        logger.debug(f"\tmean_f0_after: {mean_f0_after}")
        logger.debug(f"\tstd_f0_after: {std_f0_after}")
        logger.debug(f"\tmc_peak_frq_after: {mc_peak_frq_after}")
        logger.debug(f"\td_after: {d_after}")

        if d_before == 0 or std_f0_before == 0 or std_f0_after == 0:
            msg = f"Performed {c_iteration} iterations, returning b/c 0 values."
            logger.warning(msg)
            return c_iteration

        d_diff = abs(d_after - d_before)/d_before
        s_diff = abs(std_f0_after - std_f0_before)

        logger.debug(f"\td_diff: {d_diff}")
        logger.debug(f"\ts_diff: {s_diff}")

        if (d_diff < 0.01) and (s_diff < 0.01):
            msg = f"Performed {c_iteration} iterations, returning b/c rejection converged."
            logger.info(msg)
            return c_iteration


def reject_windows_manual(self,
                          distribution_mc='lognormal',
                          find_peaks_kwargs=None,
                          ylims=None):
    """Rejection spurious HVSR windows manually.

    Parameters
    ----------
    distribution_f0 : {'lognormal', 'normal'}, optional
        Assumed distribution of `f0` from time windows, the
        default is 'lognormal'.
    distribution_mc : {'lognormal', 'normal'}, optional
        Assumed distribution of mean curve, the default is
        'lognormal'.

    Returns
    -------
    None
        Modifies Hvsr object's internal state.

    """
    if find_peaks_kwargs is None:
        raise NotImplementedError
    (valid_idxs, _) = self.find_peaks(self.amp, **find_peaks_kwargs)

    frqs, amps = [], []
    for amp, col_ids in zip(self.amp, valid_idxs):
        frqs.extend(self.frq[col_ids])
        amps.extend(amp[col_ids])
    peaks_valid = (frqs, amps)
    peaks_invalid = ([], [])

    fig, ax = single_plot(
        self, peaks_valid, peaks_invalid, distribution_mc, ylims=ylims)
    ax.autoscale(enable=False)
    fig.show()
    pxlim = ax.get_xlim()
    pylim = ax.get_ylim()

    # continue button
    upper_right_corner = (0.05, 0.95)
    _xc, _yc = upper_right_corner
    box_size = 0.1
    scale_x = (np.log10(max(pxlim)) - np.log10(min(pxlim)))
    scale_y = max(pylim) - min(pylim)
    x_lower, x_upper = np.exp(_xc*scale_x + np.log10(min(pxlim))
                              ), np.exp((_xc+box_size)*scale_x + np.log10(min(pxlim)))
    y_lower, y_upper = (_yc - box_size)*scale_y + \
        min(pylim), _yc*scale_y + min(pylim)

    def draw_continue_box(ax):
        ax.fill([x_lower, x_upper, x_upper, x_lower],
                [y_upper, y_upper, y_lower, y_lower], color="lightgreen")
        ax.text(_xc, _yc-box_size/2, "continue?", ha="left",
                va="center", transform=ax.transAxes)
    draw_continue_box(ax)

    while True:
        xs, ys = ginput_session(fig, ax, initial_adjustment=False,
                                npts=2, ask_to_confirm_point=False, ask_to_continue=False)

        selected_columns = np.logical_and(
            self.frq > min(xs), self.frq < max(xs))
        was_empty = True
        for idx, (amp) in enumerate(zip(self.amp[:, selected_columns])):
            if np.any(np.logical_and(amp > min(ys), amp < max(ys))):
                was_empty = False
                self.valid_window_boolean_mask[idx] = False

        vfrqs, vamps = [], []
        ivfrqs, ivamps = [], []
        (valid_idxs, _) = self.find_peaks(self.amp, **find_peaks_kwargs)
        for amp, col_ids, window_valid in zip(self.amp, valid_idxs, self.valid_window_boolean_mask):
            if window_valid:
                vfrqs.extend(self.frq[col_ids])
                vamps.extend(amp[col_ids])
            else:
                ivfrqs.extend(self.frq[col_ids])
                ivamps.extend(amp[col_ids])

        peaks_valid = (vfrqs, vamps)
        peaks_invalid = (ivfrqs, ivamps)

        # Clear, set axis limits, and lock axis.
        ax.clear()
        ax.set_xlim(pxlim)
        ax.set_ylim(pylim)
        # Note: ax.clear() re-enables autoscale.
        ax.autoscale(enable=False)
        ax = single_plot(self, peaks_valid, peaks_invalid,
                         distribution_mc, ax=ax, ylims=ylims)
        draw_continue_box(ax)
        fig.canvas.draw_idle()

        if was_empty:
            in_continue_box = False
            for _x, _y in zip(xs, ys):
                if (_x < x_upper) and (_x > x_lower) and (_y > y_lower) and (_y < y_upper):
                    in_continue_box = True
                    break

            if in_continue_box:
                plt.close()
                break
            else:
                continue
