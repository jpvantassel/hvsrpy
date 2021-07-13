# This file is part of hvsrpy a Python package for horizontal-to-vertical
# spectral ratio processing.
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

"""File for organizing a command line interface (CLI)."""

import time
import configparser
import os

import click
import numpy as np

from .plottools import simple_plot, azimuthal_plot
from .sensor3c import Sensor3c

TRANSLATOR = {
    "Time Domain Settings": {
        "windowlength": float,
        "filter_bool": lambda x: False if x.lower() == "false" else True,
        "filter_flow": float,
        "filter_fhigh": float,
        "filter_forder": int,
        "width": float
    },
    "Frequency Domain Settings": {
        "bandwidth": float,
        "resample_fmin": float,
        "resample_fmax": float,
        "resample_fnum": int,
        "resample_type": str,
        "peak_f_lower": lambda x: None if x.lower() == 'none' else float(x),
        "peak_f_upper": lambda x: None if x.lower() == 'none' else float(x),
    },
    "HVSR Settings": {
        "method": str,
        "azimuth": float,
        "azimuthal_interval": float,
        "rejection_bool": lambda x: False if x.lower() == "false" else True,
        "n": float,
        "max_iterations": int,
        "distribution_f0": str,
        "distribution_mc": str
    },
    "Plot Settings": {
        "ymin": lambda x: None if x.lower() == 'none' else float(x),
        "ymax": lambda x: None if x.lower() == 'none' else float(x),
    }
}


def parse_config(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found.")

    config = configparser.ConfigParser()
    config.read(fname)

    config_kwargs = {}
    for category in TRANSLATOR:
        for key, fxn in TRANSLATOR[category].items():
            config_kwargs[key] = fxn(config[category][key].strip())

    return config_kwargs


@click.command()
@click.argument('file_names', nargs=-1, type=click.Path())
@click.option('--config', default=None, type=click.Path(), help="Path to configuration file.")
@click.option('--windowlength', default=60., help="Length of time windows, default is 60 seconds.")
@click.option('--filter_bool', default=False, help="Controls whether a bandpass filter is applied, default is False.")
@click.option('--filter_flow', default=0.1, help="Low frequency limit of bandpass filter in Hz, default is 0.1.")
@click.option('--filter_fhigh', default=30., help="High frequency limit of bandpass filter in Hz, default is 30.")
@click.option('--filter_order', default=5, help="Filter order, default is 5 (i.e., 5th order filter).")
@click.option('--width', default=0.1, help=r"Length of cosine taper, default is 0.1 (5% on each side) of time window.")
@click.option('--bandwidth', default=40., help="Bandwidth coefficient for Konno & Ohmachi (1998) smoothing, default is 40.")
@click.option('--resample_fmin', default=0.2, help="Minimum frequency in Hz to consider when resampling, defaults is 0.2.")
@click.option('--resample_fmax', default=20., help="Maximum frequency in Hz to consider when resampling, defaults is 20.")
@click.option('--resample_fnum', default=128, help="Number of samples in resampled curve, default is 128.")
@click.option('--resample_type', default="log", type=click.Choice(['log', 'linear']), help="Type of resampling, default is 'log'.")
@click.option('--peak_f_lower', default=None, type=float, help="Lower frequency limit of peak selection, defaults to entire range.")
@click.option('--peak_f_upper', default=None, type=float, help="Upper frequency limit of peak selection, defaults to entire range.")
@click.option('--method', default='geometric-mean', type=click.Choice(['squared-average', 'geometric-mean', 'single-azimuth', 'multiple-azimuths']), help="Method for combining the horizontal components, default is 'geometric-mean'.")
@click.option('--azimuth', default=0., help="Azimuth to orient horizontal components when method is 'single-azimuth', default is 0.")
@click.option('--azimuthal_interval', default=15., help="Interval in degrees between azimuths when method is 'multiple-azimuths', default is 15.")
@click.option('--rejection_bool', default=True, help="Determines whether the rejection is performed, default is True.")
@click.option('--n', default=2., help="Number of standard deviations to consider when performing the rejection, default is 2.")
@click.option('--max_iterations', default=50, help="Number of permitted iterations to convergence, default is 50.")
@click.option('--distribution_f0', default='lognormal', type=click.Choice(['lognormal', 'normal']), help="Distribution assumed to describe the fundamental site frequency, default is 'lognormal'.")
@click.option('--distribution_mc', default='lognormal', type=click.Choice(['lognormal', 'normal']), help="Distribution assumed to describe the median curve, default is 'lognormal'.")
@click.option('--no_time', is_flag=True, help="Flag to suppress HVSR compute time.")
@click.option('--no_figure', is_flag=True, help="Flag to prevent figure creation.")
@click.option('--ymin', default=None, type=float, help="Manually set the lower y limit of the HVSR figure.")
@click.option('--ymax', default=None, type=float, help="Manually set the upper y limit of the HVSR figure.")
@click.option('--summary_type', default="hvsrpy", type=click.Choice(["none", "hvsrpy", "geopsy"]), help="Summary file format to save, default is 'hvsrpy'.")
@click.pass_context
def cli(ctx, **kwargs):
    """Command line interface to hvsrpy."""
    # If config file is provided use entries as new defaults.
    if kwargs["config"] is not None:
        config_kwargs = parse_config(kwargs["config"])

        # Use value provided in config as default.
        # Argument precedence: default < config file < commandline
        for key, value in config_kwargs.items():
            if ctx.get_parameter_source("key") != "COMMANDLINE":
                kwargs[key] = value

    # TODO (jpv): Add multi-processing
    for fname in kwargs["file_names"]:
        start = time.time()
        sensor = Sensor3c.from_mseed(fname)

        bp_filter = {"flag": kwargs["filter_bool"],
                     "flow": kwargs["filter_flow"],
                     "fhigh": kwargs["filter_fhigh"],
                     "order": kwargs["filter_order"]}

        resampling = {"minf": kwargs["resample_fmin"],
                      "maxf": kwargs["resample_fmax"],
                      "nf": kwargs["resample_fnum"],
                      "res_type": kwargs["resample_type"]}

        if kwargs["method"] == "multiple-azimuths":
            azimuth = np.arange(0, 180, kwargs["azimuthal_interval"])
        else:
            azimuth = kwargs["azimuth"]

        hv = sensor.hv(kwargs["windowlength"], bp_filter, kwargs["width"],
                       kwargs["bandwidth"], resampling, kwargs["method"],
                       f_low=kwargs["peak_f_lower"], f_high=kwargs["peak_f_upper"],
                       azimuth=azimuth)

        end = time.time()
        if not kwargs["no_time"]:
            click.echo(f"{fname} completed in {(end-start):.2f} seconds")

        fname_short = fname.split("/")[-1]
        fname_short_no_ext = fname_short.split(".")[:-1]
        fname_short_no_ext_out = ".".join(fname_short_no_ext)

        if not kwargs["no_figure"]:
            if kwargs["method"] in ["squared-average", "geometric-mean"]:
                fig, _ = simple_plot(sensor, hv, kwargs["windowlength"], kwargs["distribution_f0"],
                                     kwargs["distribution_mc"], kwargs["rejection_bool"], kwargs["n"], kwargs["max_iterations"],
                                     kwargs["ymin"], kwargs["ymax"])
            else:
                fig, _ = azimuthal_plot(hv, kwargs["distribution_f0"], kwargs["distribution_mc"],
                                        kwargs["rejection_bool"], kwargs["n"], kwargs["max_iterations"],
                                        kwargs["ymin"], kwargs["ymax"])

            suffix = "_az.png" if kwargs["method"] == "multiple-azimuths" else ".png"
            fig.savefig(f"{fname_short_no_ext_out}{suffix}",
                        dpi=300, bbox_inches="tight")

        else:
            if kwargs["rejection_bool"]:
                hv.reject_windows(kwargs["n"], kwargs["max_iterations"],
                                  kwargs["distribution_f0"], kwargs["distribution_mc"])

        if kwargs["summary_type"] != "none":
            suffix = "_az.hv" if kwargs["method"] == "multiple-azimuths" else ".hv"
            hv.to_file(f"{fname_short_no_ext_out}_{kwargs['summary_type']}{suffix}",
                       kwargs["distribution_f0"], kwargs["distribution_mc"],
                       data_format=kwargs["summary_type"])
