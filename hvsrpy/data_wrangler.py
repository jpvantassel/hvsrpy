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

"""Data-related I/O."""

import pathlib
import warnings
import logging
import itertools

import obspy
import numpy as np

from .regex import saf_npts_exec, saf_fs_exec, saf_row_exec, saf_v_ch_exec, saf_n_ch_exec, saf_e_ch_exec, saf_north_rot_exec
from .regex import mshark_npts_exec, mshark_fs_exec, mshark_gain_exec, mshark_conversion_exec, mshark_row_exec
from .regex import peer_direction_exec, peer_npts_exec, peer_dt_exec, peer_sample_exec

from .timeseries import TimeSeries
from .seismic_recording_3c import SeismicRecording3C

logger = logging.getLogger('hvsrpy')


def _arrange_traces(traces):
    """Sort `list` of 3 `Trace` objects according to direction."""
    found_ew, found_ns, found_vt = False, False, False
    for trace in traces:
        if trace.meta.channel.endswith("E") and not found_ew:
            ew = TimeSeries.from_trace(trace)
            found_ew = True
        elif trace.meta.channel.endswith("N") and not found_ns:
            ns = TimeSeries.from_trace(trace)
            found_ns = True
        elif trace.meta.channel.endswith("Z") and not found_vt:
            vt = TimeSeries.from_trace(trace)
            found_vt = True
        else:
            msg = "Missing, duplicate, or incorrectly named components."
            raise ValueError(msg)
    return ns, ew, vt


def _check_npts(npts_header, npts_found):
    if npts_header != npts_found:
        msg = f"Points listed in file header ({npts_header}) does not match "
        msg += f"the number of points found ({npts_found}) please report this "
        msg += "issue to the hvsrpy developers via GitHub issues "
        msg += "(https://github.com/jpvantassel/hvsrpy/issues)."
        raise ValueError(msg)


def _quiet_obspy_read(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = obspy.read(*args, **kwargs)
    return results


def read_mseed(fnames, read_kwargs=None, degrees_from_north=None):
    """Read ambient noise data from file(s) in miniSEED format.

    Parameters
    ----------
    fnames : {str, list}
        If `str` then `fnames` is the name of the miniseed file,
        full path may be used if desired. The file should contain
        three traces with the appropriate channel names. Refer to
        the `SEED` Manual
        `here <https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.
        for specifics.
        If `list` then `fnames` is a list of length three with the
        names of miniseed files for each component.
    obspy_read_kwargs : dict, optional
        For passing arguments to the `obspy.read` function to
        customize its behavior, default is `None`indicating
        not keyword arguments will be passed.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positve. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.
    (TODO) JPV: Check formatting of text above.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if read_kwargs is None:
        read_kwargs = {"format": "MSEED"}

    # one miniSEED file with all three components.
    if isinstance(fnames, (str, pathlib.Path)):
        traces = _quiet_obspy_read(fnames, **read_kwargs)
    # three miniSEED files, one per component.
    elif isinstance(fnames, (list, tuple)):
        trace_list = []
        for fname in fnames:
            stream = _quiet_obspy_read(fname, **read_kwargs)
            if len(stream) != 1:
                msg = f"File {fname} contained {len(stream)}"
                msg += "traces, rather than 1 as was expected."
                raise IndexError(msg)
            trace = stream[0]
            trace_list.append(trace)
        traces = obspy.Stream(trace_list)
    else:
        msg = "`fnames` must be either `str` or `list`"
        msg += f"cannot be {type(fnames)}."
        raise ValueError(msg)

    if len(traces) != 3:
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"File Name(s)": fnames,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def read_saf(fnames, read_kwargs=None, degrees_from_north=None):
    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 saf file allowed; {len(fnames)} provided. "
        msg += "Only taking first."
        warnings.warn(msg, UserWarning)
    else:
        fname = fnames

    with open(fname, "r") as f:
        text = f.read()

    npts_header = int(saf_npts_exec.search(text).groups()[0])
    dt = 1/float(saf_fs_exec.search(text).groups()[0])
    v_ch = int(saf_v_ch_exec.search(text).groups()[0])
    n_ch = int(saf_n_ch_exec.search(text).groups()[0])
    e_ch = int(saf_e_ch_exec.search(text).groups()[0])
    if degrees_from_north is None:
        try:
            north_rot = float(saf_north_rot_exec.search(text).groups()[0])
        except:
            msg = f"The provided saf file {fname} does not include the "
            msg += "NORTH_ROT keyword, assuming equal to zero."
            warnings.warn(msg, UserWarning)
            degrees_from_north = 0.
        else:
            if n_ch == 1:
                degrees_from_north = north_rot
            elif e_ch == 1:
                degrees_from_north = north_rot + 90.
            else:
                msg = f"The provided saf file {fname} is not properly "
                msg += "formatted. CH1 must be veritcal; CH2 & CH3 the horizontals"
                raise ValueError(msg)

    data = np.empty((npts_header, 3), dtype=np.float32)

    idx = 0
    for group in saf_row_exec.finditer(text):
        channels = group.groups()
        data[idx, 0] = float(channels[v_ch])
        data[idx, 1] = float(channels[n_ch])
        data[idx, 2] = float(channels[e_ch])
        idx += 1

    _check_npts(npts_header, idx)

    vt, ns, ew = data.T

    vt = TimeSeries(vt, dt=dt)
    ns = TimeSeries(ns, dt=dt)
    ew = TimeSeries(ew, dt=dt)

    meta = {"File Name(s)": fname,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def read_minishark(fnames, read_kwargs=None, degrees_from_north=None):
    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 minishark file allowed; {len(fnames)} provided. "
        msg += "Only taking first."
        warnings.warn(msg, IndexError)
    else:
        fname = fnames

    with open(fname, "r") as f:
        text = f.read()

    npts_header = int(mshark_npts_exec.search(text).groups()[0])
    dt = 1/float(mshark_fs_exec.search(text).groups()[0])
    conversion = int(mshark_conversion_exec.search(text).groups()[0])
    gain = int(mshark_gain_exec.search(text).groups()[0])

    data = np.empty((npts_header, 3), dtype=np.float32)

    idx = 0
    for group in mshark_row_exec.finditer(text):
        vt, ns, ew = group.groups()
        data[idx, 0] = float(vt)
        data[idx, 1] = float(ns)
        data[idx, 2] = float(ew)
        idx += 1

    _check_npts(npts_header, idx)

    data /= gain
    data /= conversion

    vt, ns, ew = data.T

    vt = TimeSeries(vt, dt=dt)
    ns = TimeSeries(ns, dt=dt)
    ew = TimeSeries(ew, dt=dt)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"File Name(s)": fname,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def read_sac(fnames, read_kwargs=None, degrees_from_north=None):
    if read_kwargs is None:
        read_kwargs = {"format": "SAC"}

    if not isinstance(fnames, (list, tuple)):
        msg = "Must provide 3 sac files (one per trace); only one provided."
        raise ValueError(msg)

    trace_list = []
    for fname in fnames:
        for byteorder in ["little", "big"]:
            read_kwargs["byteorder"] = byteorder
            try:
                stream = _quiet_obspy_read(fname, **read_kwargs)
            except Exception as e:
                msg = f"Tried reading as sac {byteorder} endian, "
                msg += f"got exception |  {e}"
                logger.info(msg)
                pass
            else:
                break
        else:
            raise e

        trace = stream[0]
        trace_list.append(trace)
    traces = obspy.Stream(trace_list)

    if len(traces) != 3:
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"File Name(s)": fnames,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def read_gcf(fnames, read_kwargs=None, degrees_from_north=None):
    if read_kwargs is None:
        read_kwargs = {"format": "GCF"}

    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 gcf file allowed; {len(fnames)} provided. "
        msg += "Only taking first."
        warnings.warn(msg, IndexError)
    else:
        fname = fnames

    # one gcf file with all three components.
    if isinstance(fname, (str, pathlib.Path)):
        traces = _quiet_obspy_read(fname, **read_kwargs)

    if len(traces) != 3:
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"File Name(s)": fname,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def read_peer(fnames, read_kwargs=None, degrees_from_north=None):
    if read_kwargs is None:
        read_kwargs = {}

    if not isinstance(fnames, (list, tuple)):
        msg = "Must provide 3 peer files (one per trace) as list or tuple, "
        msg += f"not {type(fnames)}."
        raise ValueError(msg)

    component_list = []
    component_keys = []
    for fname in fnames:
        with open(fname, "r") as f:
            text = f.read()

        component_keys.append(peer_direction_exec.search(text).groups()[0])

        npts_header = int(peer_npts_exec.search(text).groups()[0])
        dt = float(peer_dt_exec.search(text).groups()[0])

        amplitude = np.empty((npts_header))
        idx = 0
        for group in peer_sample_exec.finditer(text):
            sample, = group.groups()
            amplitude[idx] = sample
            idx += 1

        _check_npts(npts_header, idx)

        component_list.append(TimeSeries(amplitude, dt=dt))

    # organize components - vertical
    vt_id = component_keys.index("UP")
    vt = component_list[vt_id]
    del component_list[vt_id], component_keys[vt_id]

    # organize components - horizontals
    component_keys_abs = np.array(component_keys, dtype=int)
    component_keys_rel = component_keys_abs.copy()
    component_keys_rel[component_keys_abs > 180] -= 360
    ns_id = np.argmin(abs(component_keys_rel))
    ns = component_list[ns_id]
    ew_id = np.argmax(abs(component_keys_rel))
    ew = component_list[ew_id]
    del component_list, component_keys

    # set rotation iff degrees_from_north is not already set.
    if degrees_from_north is None:
        degrees_from_north = component_keys_abs[ns_id]

    meta = {"File Name(s)": fname,
            "Deployed Degrees from North (deg)": degrees_from_north,
            "Current Degrees from North (deg)": degrees_from_north}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


READ_FUNCTION_DICT = {
    "mseed": read_mseed,
    "saf": read_saf,
    "minishark": read_minishark,
    "sac": read_sac,
    "gcf": read_gcf,
    "peer": read_peer
}


def read_single(fnames, read_kwargs=None, degrees_from_north=None):
    """Read file(s) associated with a single recording.

    Parameters
    ----------
    fnames: {list, str}
        If `str`, name of file to be read, may include a relative or the
        full path. The file should contain all three components
        (2 horizontal and 1 vertical).
        If `list`, names of files to be read, each may be a relative or
        the full path. Each file should contain only one component.
    read_kwargs : dict, optional
        Keyword arguments to be passed directly to `obspy.read`, in
        general this should not be needed, default is `None` indicating
        no custom arguments will be passed to `obspy.read`.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positve. Default is `None`
        indicating either the metadata in the file denoting the sensor's
        orientation is correct and should be used or (if the sensor's
        orientation is not listed in teh file) the sensor's north
        component is aligned with magnetic north
        (i.e., `degrees_from_north=0`).

    Returns
    -------
    SeismicRecording3C
        Initialized three-component seismic recording object.
    # TODO (jpv): Check this renders correctly.

    """
    logger.info(f"Attempting to read {fnames}")
    for ftype, read_function in READ_FUNCTION_DICT.items():
        try:
            srecording_3c = read_function(fnames,
                                          read_kwargs=read_kwargs,
                                          degrees_from_north=degrees_from_north)
        except Exception as e:
            logger.info(f"Tried reading as {ftype}, got exception |  {e}")
            pass
        else:
            logger.info(f"File type identified as {ftype}.")
            break
    else:
        msg = "File format not recognized. Only the following are supported: "
        msg += f"{READ_FUNCTION_DICT.keys()}."
        raise ValueError(msg)
    return srecording_3c


def read(fnames, read_kwargs=None, degrees_from_north=None):
    """Read file(s) presented.

    Parameters
    ----------
    fnames : iterable of iterable of str or interable of str
        Collection of file name(s) to be read. All entries should be
        readable by the function `hvsrpy.read_single()`.
    read_kwargs : dict or iterable of dicts, optional
        Keyword arguments to be passed directly to
        `hvsrpy.read_single()`. If `dict`, keyword argument will be
        repeated for all file names provided. If `iterable of dicts`
        each keyword arguments will be provided in order. Default is
        `None` indicating standard read behavior will be used.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positve. Default is `None`
        indicating either the metadata in the file denoting the sensor's
        orientation is correct and should be used or (if the sensor's
        orientation is not listed in teh file) the sensor's north
        component is aligned with magnetic north
        (i.e., `degrees_from_north=0`).

    Returns
    -------
    list
        Of initialized SeismicRecording3C objects, one for each each
        file name provided.
    # TODO (jpv): Check this renders correctly.

    """
    # scale read_kwargs and degrees_from_north as needed to match fnames.
    if isinstance(read_kwargs, (dict, type(None))):
        read_kwargs_iter = itertools.repeat(read_kwargs)
        degrees_from_north_iter = itertools.repeat(degrees_from_north)
    else:
        read_kwargs_iter = read_kwargs
        degrees_from_north_iter = degrees_from_north

    seismic_recordings = []
    for fname, read_kwargs, degrees_from_north in zip(fnames, read_kwargs_iter, degrees_from_north_iter):

        # if entry is a list with only a single entry, remove the list.
        if len(fname) == 1:
            fname = fname[0]

        seismic_recordings.append(read_single(fname,
                                              read_kwargs=read_kwargs,
                                              degrees_from_north=degrees_from_north))

    return seismic_recordings
