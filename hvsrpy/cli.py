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

"""File for organizing a command line interface (CLI)."""

import time
import itertools
import os
from multiprocessing import Pool
import pathlib

import click
import matplotlib.pyplot as plt

import hvsrpy
from hvsrpy.object_io import read_settings_object_from_file

def _process_hvsr(fname, preprocessing_settings, processing_settings, settings): # pragma: no cover
    start = time.perf_counter()
    srecords = hvsrpy.read([[fname]])
    srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
    hvsr = hvsrpy.process(srecords, processing_settings)

    if not settings["no_figure"]:
        plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)
        fig, ax = plt.subplots(figsize=(3.75, 2.5), dpi=150)
        ax.set_ylim((0, settings["ymax"]))
        hvsrpy.plot_single_panel_hvsr_curves(hvsr, ax=ax)
        fig.savefig(f"{pathlib.Path(fname).stem}.png")
        plt.close()

    if not settings["no_file"]:
        hvsrpy.write_hvsr_object_to_file(hvsr,
                                  f"{pathlib.Path(fname).stem}.csv",
                                  distribution_mc=settings["distribution_mc"],
                                  distribution_fn=settings["distribution_fn"],
                                  )
    end = time.perf_counter()
    print(f"{fname} completed in {end-start:.3f} seconds.")

@click.command()
@click.argument('file_names', nargs=-1, type=click.Path())
@click.option('--preprocessing_settings_file', default=None, type=click.Path(), help="Path to preprocessing settings file.")
@click.option('--processing_settings_file', default=None, type=click.Path(), help="Path to processing settings file.")
@click.option('--distribution_fn', default='lognormal', type=click.Choice(['lognormal', 'normal']), help="Distribution assumed to describe the site frequency, default is 'lognormal'.")
@click.option('--distribution_mc', default='lognormal', type=click.Choice(['lognormal', 'normal']), help="Distribution assumed to describe the median curve, default is 'lognormal'.")
@click.option('--no_figure', is_flag=True, help="Flag to prevent figure creation.")
@click.option('--no_file', is_flag=True, help="Flag to prevent HVSR from being saved.")
@click.option('--ymax', default=10., type=float, help="Manually set the upper y limit of the HVSR figure, default is 10.")
@click.option('--nproc', default=None, type=int, help="Number of subprocesses to launch, default is number of CPUs minus 1.")
@click.pass_context
def cli(ctx, **kwargs):  # pragma: no cover
    """Command line interface to hvsrpy."""
    preprocessing_settings = read_settings_object_from_file(kwargs.pop("preprocessing_settings_file"))
    processing_settings = read_settings_object_from_file(kwargs.pop("processing_settings_file"))

    if kwargs["no_figure"] and kwargs["no_file"]:
        return
    
    nproc = os.cpu_count()-1 if kwargs["nproc"] is None else kwargs["nproc"]
    fnames = kwargs.pop("file_names")
    ntasks = len(fnames)
    with Pool(min(ntasks, nproc)) as p:
        p.starmap(_process_hvsr,
                  zip(fnames,
                      itertools.repeat(preprocessing_settings),
                      itertools.repeat(processing_settings),
                      itertools.repeat(kwargs),
                    ),
                  chunksize=max(1, ntasks//nproc))
