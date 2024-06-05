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

"""Summary of functions to control hvsrpy input-output (IO)."""

import json
from copy import deepcopy

import numpy as np

from .hvsr_diffuse_field import HvsrDiffuseField
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal
from .settings import *
from .regex import azimuth_exec

__all__ = [
    "write_hvsr_object_to_file",
    "read_hvsr_object_from_file",
    "write_settings_object_to_file",
    "read_settings_object_from_file",
]

def write_hvsr_object_to_file(hvsr,
                       fname,
                       distribution_mc="lognormal",
                       distribution_fn="lognormal"):
    """Write HVSR object to text-based file.

    Parameters
    ----------
    hvsr : {HvsrTraditional, HvsrAzimuthal, HvsrDiffuseField}
        HVSR object that should be archived to a file on disk.
    fname : str
        Name of output file where the contents of the HVSR object are
        to be stored. May be a relative or the full path.
    distribution_mc : {"normal", "lognormal"}, optional
        Assumed distribution of mean curve, default is "lognormal".
        Ignored for ``HvsrDiffuseField`` objects.
    distribution_fn : {"normal", "lognormal"}, optional
        Assumed distribution of ``fn``, the default is ``"lognormal"``.
        Ignored for ``HvsrDiffuseField`` objects.

    Returns
    -------
    None
        Instead writes HVSR object to disk in text-based format.

    """
    meta = deepcopy(hvsr.meta)

    if isinstance(hvsr, HvsrTraditional):
        # valid peaks and windows
        meta["valid_peak_boolean_mask"] = hvsr.valid_peak_boolean_mask.tolist()
        meta["valid_window_boolean_mask"] = hvsr.valid_window_boolean_mask.tolist()

        # data headers
        data_headers_line = ["frequency (Hz)"]
        data_headers_line.extend(
            [f"hvsr curve {x}" for x in range(1, hvsr.n_curves+1)])
        data_headers_line.extend(
            [f"mean curve ({distribution_mc})", f"mean curve std ({distribution_mc})"])
        header_line = ",".join(data_headers_line)

        # curve data
        array = np.empty((len(hvsr.frequency), len(data_headers_line)))
        array[:, 0] = hvsr.frequency
        array[:, 1:-2] = hvsr.amplitude.T
        array[:, -2] = hvsr.mean_curve(distribution=distribution_mc)
        array[:, -1] = hvsr.std_curve(distribution=distribution_mc)

    elif isinstance(hvsr, HvsrAzimuthal):
        # Note: HvsrAzimuthal does not require each HvsrTraditional to have the same number of curves.

        # valid peaks and windows
        valid_window_boolean_masks = []
        valid_peak_boolean_masks = []
        for _hvsr in hvsr.hvsrs:
            valid_window_boolean_masks.append(
                _hvsr.valid_window_boolean_mask.tolist())
            valid_peak_boolean_masks.append(
                _hvsr.valid_peak_boolean_mask.tolist())
        meta["valid_window_boolean_masks"] = valid_window_boolean_masks
        meta["valid_peak_boolean_masks"] = valid_peak_boolean_masks

        # data headers
        data_headers_line = ["frequency (Hz)"]
        for azimuth, _hvsr in zip(hvsr.azimuths, hvsr.hvsrs):
            for curve_idx in range(1, _hvsr.n_curves+1):
                data_headers_line.append(
                    f"azimuth {azimuth} deg | hvsr curve {curve_idx}")
        data_headers_line.extend(
            [f"mean curve ({distribution_mc})", f"mean curve std ({distribution_mc})"])
        header_line = ",".join(data_headers_line)

        # curve data
        array = np.empty((len(hvsr.frequency), len(data_headers_line)))
        array[:, 0] = hvsr.frequency
        start_index = 1
        for hvsr in hvsr.hvsrs:
            stop_index = start_index + hvsr.n_curves
            array[:, start_index:stop_index] = hvsr.amplitude.T
            start_index = stop_index
        array[:, -2] = hvsr.mean_curve(distribution=distribution_mc)
        array[:, -1] = hvsr.std_curve(distribution=distribution_mc)

    elif isinstance(hvsr, HvsrDiffuseField):
        # data headers
        data_headers_line = ["frequency (Hz)", "hvsr curve 1"]
        header_line = ",".join(data_headers_line)

        array = np.empty((len(hvsr.frequency), len(data_headers_line)))
        array[:, 0] = hvsr.frequency
        array[:, 1] = hvsr.amplitude

    else:
        raise NotImplementedError

    header = "".join([
        json.dumps(meta, indent=2), "\n",
        header_line,
    ])
    np.savetxt(fname, array, delimiter=",", header=header, encoding="utf-8")


def read_hvsr_object_from_file(fname):
    """Reads HVSR object from text-based file.

    Parameters
    ----------
    fname : str
        Name of output file where the contents of the HVSR object are
        stored. May be a relative or the full path.

    Returns
    -------
    hvsr : {HvsrTraditional, HvsrAzimuthal, HvsrDiffuseField}
        HVSR object that was archived in a file on disk.

    """
    with open(fname, "r") as f:
        lines = f.readlines()

    header_lines = []
    for line in lines:
        if line.startswith("#"):
            header_lines.append(line[2:])  # remove "# "
        else:
            break
    meta = json.loads("\n".join(header_lines[:-1]))
    header_line = header_lines[-1].split(",")
    array = np.loadtxt(fname, comments="#", delimiter=",")

    if meta["processing_method"] == "traditional":
        hvsr = HvsrTraditional(array[:, 0], array[:, 1:-2].T)
        hvsr.meta = meta
        hvsr.update_peaks_bounded(
            search_range_in_hz=tuple(meta["search_range_in_hz"]),
            find_peaks_kwargs=meta["find_peaks_kwargs"]
        )
        hvsr.valid_window_boolean_mask = np.array(
            meta.pop("valid_window_boolean_mask")
        )
        hvsr.valid_peak_boolean_mask = np.array(
            meta.pop("valid_peak_boolean_mask")
        )

    elif meta["processing_method"] == "azimuthal":

        prev_azimuth = azimuth_exec.search(header_line[1]).groups()[0]
        hvsrs = []
        start_idx = 1
        azimuths = []
        for idx, header in enumerate(header_line[1:-2], start=1):
            curr_azimuth = azimuth_exec.search(header).groups()[0]
            if curr_azimuth != prev_azimuth:
                azimuths.append(float(prev_azimuth))
                hvsrs.append(HvsrTraditional(array[:, 0],
                                             array[:, start_idx:idx].T,
                                             meta={})
                             )
                start_idx = idx
                prev_azimuth = curr_azimuth
        azimuths.append(float(curr_azimuth))
        hvsrs.append(HvsrTraditional(array[:, 0],
                                     array[:, start_idx:idx+1].T,
                                     meta={})
                     )

        hvsr = HvsrAzimuthal(hvsrs=hvsrs, azimuths=azimuths, meta=meta)
        hvsr.meta = meta
        hvsr.update_peaks_bounded(
            search_range_in_hz=tuple(meta["search_range_in_hz"]),
            find_peaks_kwargs=meta["find_peaks_kwargs"]
        )

        for _hvsr, _vwbm, _vpbm in zip(hvsr.hvsrs,
                                       meta.pop("valid_window_boolean_masks"),
                                       meta.pop("valid_peak_boolean_masks")):
            _hvsr.valid_window_boolean_mask = np.array(_vwbm)
            _hvsr.valid_peak_boolean_mask = np.array(_vpbm)

    elif meta["processing_method"] == "diffuse_field":
        hvsr = HvsrDiffuseField(array[:, 0], array[:, 1], meta=meta)
        hvsr.update_peaks_bounded(search_range_in_hz=meta["search_range_in_hz"],
                                  find_peaks_kwargs=meta["find_peaks_kwargs"])

    return hvsr


def write_settings_object_to_file(settings_object, fname):
    """Write HVSR settings object to text-based file.

    Parameters
    ----------
    settings_object : Settings
        HVSR settings object that should be archived to a file on disk.
    fname : str
        Name of output file where the contents of the HVSR settings
        object are to be stored. May be a relative or the full path.

    Returns
    -------
    None
        Instead writes HVSR settings object to disk in text-based format.

    """
    settings_object.save(fname)


def read_settings_object_from_file(fname):
    """Create HVSR settings object from text-based file.

    Parameters
    ----------
    fname : str
        Name of output file where the HVSR settings will be
        stored. May be a relative or the full path.

    Returns
    -------
    hvsr : Settings
        HVSR settings object that was archived in a file on disk.

    """

    with open(fname, "r") as f:
        attr_dict = json.load(f)

    # preprocessing settings
    if "preprocessing_method" in attr_dict.keys():
        if attr_dict["preprocessing_method"] == "psd":
            settings_object = PsdPreProcessingSettings()
        elif attr_dict["preprocessing_method"] == "hvsr":
            settings_object = HvsrPreProcessingSettings()
        else:
            raise NotImplementedError

    elif "processing_method" in attr_dict.keys():
        if attr_dict["processing_method"] == "psd":
            settings_object = PsdProcessingSettings()
        elif attr_dict["processing_method"] == "azimuthal":
            settings_object = HvsrAzimuthalProcessingSettings()
        elif attr_dict["processing_method"] == "diffuse_field":
            settings_object = HvsrDiffuseFieldProcessingSettings()
        elif attr_dict["processing_method"] == "traditional":
            if attr_dict["method_to_combine_horizontals"] == "rotdpp":
                settings_object = HvsrTraditionalRotDppProcessingSettings()
            elif attr_dict["method_to_combine_horizontals"] == "single_azimuth":
                settings_object = HvsrTraditionalSingleAzimuthProcessingSettings()
            else:
                settings_object = HvsrTraditionalProcessingSettings()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    settings_object.load(fname)
    return settings_object
