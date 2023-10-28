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

"""Plot interaction module."""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor


def ginput_session(fig, ax,
                   initial_adjustment=True,
                   initial_adjustment_message=None,
                   n_points=1,
                   ask_to_confirm_point=True,
                   ask_to_continue=True,
                   ask_to_continue_message=None): # pragma: no cover
    """Start ginput session using the provided axes object.

    Parameters
    ----------
    fig : Figure
        Active Figure.
    ax : Axes
        Axes on which points are to be selected.
    initial_adjustment : bool, optional
        Allow user to pan and zoom prior to the selection of the
        first point, default is ``True``.
    initial_adjustment_message : str, optional
        Message to print and display during ``initial_adjustment``
         stage, default is ``None`` so a predefined message is
         displayed.
    n_points : int, optional
        Predefine the number of points the user is allowed to
        select, the default is ``1``.
    ask_to_continue : bool, optional
        Pause the selection process after each point. This allows
        the user to pan and zoom the figure as well as select when
        to continue, default is ``True``.
    ask_to_continue_message : str, optional
        Message to print and display prior to select stage,
        default is ``None`` so a predefined message is displayed.

    Returns
    -------
    tuple
        Of the form ``(xs, ys)`` where ``xs`` is a ``list`` of x
        coordinates and ``ys`` is a ``list`` of y coordinates in the
        order in which they were picked.

    """
    # Enable cursor to make precise selection easier.
    Cursor(ax, useblit=True, color="k", linewidth=1)

    # Permit initial adjustment with blocking call to figure.
    if initial_adjustment:
        if initial_adjustment_message is None:
            initial_adjustment_message = "Adjust view,\nspacebar when ready."
        text = ax.text(0.95, 0.95, initial_adjustment_message,
                       ha="right", va="top", transform=ax.transAxes)
        while True:
            if plt.waitforbuttonpress(timeout=-1):
                text.set_visible(False)
                break

    # Begin selection of n_points.
    npt, xs, ys = 0, [], []
    while npt < n_points:
        if ask_to_confirm_point:
            selection_message = "Left click to add,\nright click to remove,\nenter to accept."
            text = ax.text(0.95, 0.95, selection_message,
                           ha="right", va="top", transform=ax.transAxes)
            vals = plt.ginput(n=-1, timeout=0)
            text.set_visible(False)
        else:
            vals = plt.ginput(n=1, timeout=0)

        if len(vals) > 1:
            msg = "More than one point selected, ignoring all but the last point."
            warnings.warn(msg)

        if len(vals) == 0:
            msg = "No points selected, try again."
            warnings.warn(msg)
            continue

        x, y = vals[-1]
        ax.plot(x, y, "r", marker="+", linestyle="")
        xs.append(x)
        ys.append(y)
        npt += 1
        fig.canvas.draw_idle()

        if ask_to_continue:
            if ask_to_continue_message is None:
                ask_to_continue_message = "Adjust view,\npress spacebar\nonce to contine,\ntwice to exit."
            text = ax.text(0.95, 0.95, ask_to_continue_message,
                           ha="right", va="top",
                           transform=ax.transAxes)
            while True:
                if plt.waitforbuttonpress(timeout=-1):
                    text.set_visible(False)
                    break

        if plt.waitforbuttonpress(timeout=0.3):
            break

    finish_message = "Interactive session complete,\nclose figure(s) when ready."
    ax.text(0.95, 0.95, finish_message, ha="right", va="top",
            transform=ax.transAxes)

    return (xs, ys)


def _relative_to_absolute(relative, range_absolute, scale="linear"):
    """Convert relative value (between 0 and 1) to absolute value.

    .. warning::
        Private methods are subject to change without warning.

    """
    abs_min, abs_max = range_absolute
    if scale == "linear":
        return abs_min + relative*(abs_max-abs_min)
    elif scale == "log":
        value = np.log10(abs_min) + relative*(np.log10(abs_max/abs_min))
        return np.power(10, value)
    else: # pragma: no cover
        raise NotImplementedError


def _absolute_to_relative(absolute, range_absolute, scale="linear"): # pragma: no cover
    """Convert absolute value to a relative value (between 0 and 1).

    .. warning::
        Private methods are subject to change without warning.

    """
    abs_min, abs_max = range_absolute
    if scale == "linear":
        return (absolute - abs_min) / (abs_max - abs_min)
    elif scale == "log":
        return np.log10(absolute/abs_min) / (np.log10(abs_max/abs_min))
    else: # pragma: no cover
        raise NotImplementedError


def _relative_box_coordinates(upper_right_corner_relative=(0.95, 0.95),
                              box_size_relative=(0.1, 0.05)): # pragma: no cover
    """Relative box coordinates from relative location and size.

    .. warning::
        Private methods are subject to change without warning.

    """
    x_upper_rel, y_upper_rel = upper_right_corner_relative
    x_box_rel, y_box_rel = box_size_relative
    x_lower_rel, y_lower_rel = x_upper_rel - x_box_rel, y_upper_rel - y_box_rel
    return (x_lower_rel, x_upper_rel, y_lower_rel, y_upper_rel)


def _absolute_box_coordinates(x_range_absolute,
                              y_range_absolute,
                              upper_right_corner_relative=(0.95, 0.95),
                              box_size_relative=(0.1, 0.05),
                              x_scale="linear",
                              y_scale="linear"): # pragma: no cover
    """Absolute box coordinates from relative location and size.

    .. warning::
        Private methods are subject to change without warning.

    """
    # define box in relative coordinates (0 to 1).
    rel_coordinates = _relative_box_coordinates(upper_right_corner_relative=upper_right_corner_relative,
                                                box_size_relative=box_size_relative)
    x_lower_rel, x_upper_rel, y_lower_rel, y_upper_rel = rel_coordinates

    # scale box to absolute coordinates.
    x_box_lower_abs = _relative_to_absolute(x_lower_rel, x_range_absolute, scale=x_scale)
    x_box_upper_abs = _relative_to_absolute(x_upper_rel, x_range_absolute, scale=x_scale)
    y_box_lower_abs = _relative_to_absolute(y_lower_rel, y_range_absolute, scale=y_scale)
    y_box_upper_abs = _relative_to_absolute(y_upper_rel, y_range_absolute, scale=y_scale)
    return (x_box_lower_abs, x_box_upper_abs, y_box_lower_abs, y_box_upper_abs)


def plot_continue_button(ax, upper_right_corner_relative=(0.95, 0.95),
                         box_size_relative=(0.1, 0.05), fill_kwargs=None): # pragma: no cover
    """Draw continue button on axis.

    .. warning::
        Private methods are subject to change without warning.

    """
    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()
    x_range_absolute = ax.get_xlim()
    y_range_absolute = ax.get_ylim()
    box_rel = _relative_box_coordinates(upper_right_corner_relative=upper_right_corner_relative,
                                        box_size_relative=box_size_relative)
    (x_lower_rel, x_upper_rel, y_lower_rel, y_upper_rel) = box_rel

    box_abs = _absolute_box_coordinates(x_range_absolute=x_range_absolute,
                                        y_range_absolute=y_range_absolute,
                                        upper_right_corner_relative=upper_right_corner_relative,
                                        box_size_relative=box_size_relative,
                                        x_scale=x_scale,
                                        y_scale=y_scale)
    (x_box_lower_abs, x_box_upper_abs, y_box_lower_abs, y_box_upper_abs) = box_abs

    default_kwargs = dict(color="lightgreen")
    if fill_kwargs is None:
        fill_kwargs = {}
    fill_kwargs = {**default_kwargs, **fill_kwargs}

    ax.fill([x_box_lower_abs, x_box_lower_abs, x_box_upper_abs, x_box_upper_abs],
            [y_box_lower_abs, y_box_upper_abs, y_box_upper_abs, y_box_lower_abs],
            **fill_kwargs)
    ax.text((x_lower_rel + x_upper_rel)/2, (y_lower_rel + y_upper_rel)/2, "continue?",
            ha="center", va="center", transform=ax.transAxes)


def is_absolute_point_in_relative_box(ax,
                                      absolute_point,
                                      upper_right_corner_relative=(0.95, 0.95),
                                      box_size_relative=(0.1, 0.05)): # pragma: no cover
    """Determines if a point (defined in absolute coordinates) is inside
    of a box (defined in relative coordinates).

    .. warning::
        Private methods are subject to change without warning.

    """
    x_min, x_max, y_min, y_max = _relative_box_coordinates(upper_right_corner_relative=upper_right_corner_relative,
                                                           box_size_relative=box_size_relative)
    abs_x, abs_y = absolute_point
    rel_x = _absolute_to_relative(abs_x, ax.get_xlim(), ax.get_xscale())
    rel_y = _absolute_to_relative(abs_y, ax.get_ylim(), ax.get_yscale())
    if (rel_x > x_min) and (rel_x < x_max) and (rel_y > y_min) and (rel_y < y_max):
        return True
    return False
