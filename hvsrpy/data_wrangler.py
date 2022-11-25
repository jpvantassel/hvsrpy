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

"""Data-related I/O."""

import pathlib
import warnings

import obspy
import numpy as np

from .regex import saf_npts_exec, saf_fs_exec, saf_row_exec
from .regex import mshark_npts_exec, mshark_fs_exec, mshark_gain_exec, mshark_conversion_exec, mshark_row_exec
from .regex import peer_direction_exec, peer_npts_exec, peer_dt_exec, peer_sample_exec

from .timeseries import TimeSeries
from .seismic_recording_3c import SeismicRecording3C

def read_mseed(fnames, read_kwargs=None):
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
        (TODO) JPV: Check formatting of text above.
    obspy_read_kwargs : dict, optional
        For passing arguments to the `obspy.read` function to
        customize its behavior, default is `None`indicating
        not keyword arguments will be passed.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if read_kwargs is None:
        read_kwargs = {"format":"MSEED"}

    # one miniSEED file with all three components.
    if isinstance(fnames, (str, pathlib.Path)):
        traces = obspy.read(fnames, **read_kwargs)
    # three miniSEED files, one per component.
    elif isinstance(fnames, (list, tuple)):
        trace_list = []
        for fname in fnames:
            stream = obspy.read(fname, **read_kwargs)
            if len(stream) != 1:
                msg = f"File {fname} contained {len(stream)}"
                msg += "traces, rather than 1 as was expected."
                raise IndexError(msg)
            trace = stream[0]
            trace_list.append(trace)
        traces = obspy.Stream(trace_list)
    else:
        msg = f"`fnames` must be either `str` or `list` cannot be {type(fnames)}."
        raise ValueError(msg)

    if len(traces) != 3:
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

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

    meta = {"File Name(s)": fnames}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

def read_saf(fnames, read_kwargs=None):
    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 saf file allowed; {len(fnames)} provided. Only taking first."
        warnings.warn(msg, IndexError)
    else:
        fname = fnames

    with open(fname, "r") as f:
        text = f.read()

    npts = int(saf_npts_exec.search(text).groups()[0])
    dt = 1/float(saf_fs_exec.search(text).groups()[0])

    data = np.empty((npts, 3), dtype=np.float32)

    idx = 0
    for group in saf_row_exec.finditer(text):
        vt, ns, ew = group.groups()
        data[idx, 0] = float(vt)
        data[idx, 1] = float(ns)
        data[idx, 2] = float(ew)
        idx += 1

    if idx != npts:
        msg = f"Points listed in file header ({npts}) does not match the number of points found ({idx})"
        msg += " please report this issue to the hvsrpy developers via GitHub issues"
        msg += " (https://github.com/jpvantassel/hvsrpy/issues)."
        raise ValueError(msg)

    vt, ns, ew = data.T
        
    vt = TimeSeries(vt, dt=dt)
    ns = TimeSeries(ns, dt=dt)
    ew = TimeSeries(ew, dt=dt)
            
    meta = {"File Name(s)": fname}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

def read_minishark(fnames, read_kwargs=None):
    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 minishark file allowed; {len(fnames)} provided. Only taking first."
        warnings.warn(msg, IndexError)
    else:
        fname = fnames

    with open(fname, "r") as f:
        text = f.read()
        
    npts = int(mshark_npts_exec.search(text).groups()[0])
    dt = 1/float(mshark_fs_exec.search(text).groups()[0])
    conversion = int(mshark_conversion_exec.search(text).groups()[0])
    gain = int(mshark_gain_exec.search(text).groups()[0])

    data = np.empty((npts, 3), dtype=np.float32)

    idx = 0
    for group in mshark_row_exec.finditer(text):
        vt, ns, ew = group.groups()
        data[idx, 0] = float(vt)
        data[idx, 1] = float(ns)
        data[idx, 2] = float(ew)
        idx += 1
        
    if idx != npts:
        msg = f"Points listed in file header ({npts}) does not match the number of points found ({npts_iter})"
        msg += " please report this issue to the hvsrpy developers via GitHub issues"
        msg += " (https://github.com/jpvantassel/hvsrpy/issues)."
        raise ValueError(msg)
        
    data /= gain
    data /= conversion

    vt, ns, ew = data.T

    vt = TimeSeries(vt, dt=dt)
    ns = TimeSeries(ns, dt=dt)
    ew = TimeSeries(ew, dt=dt)

    meta = {"File Name(s)": fname}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

def read_sac(fnames, read_kwargs=None):
    if read_kwargs is None:
        read_kwargs = {"format":"SAC"}

    if not isinstance(fnames, (list, tuple)):
        raise ValueError("Must provide 3 sac files (one per trace); only one provided.")

    trace_list = []
    for fname in fnames:
        for byteorder in ["little", "big"]:
            read_kwargs["byteorder"] = byteorder
            try:
                stream = obspy.read(fname, **read_kwargs)
            except Exception as e:
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

    meta = {"File Name(s)": fnames}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

def read_gcf(fnames, read_kwargs=None):
    if read_kwargs is None:
        read_kwargs = {"format":"GCF"}

    if isinstance(fnames, (list, tuple)):
        fname = fnames[0]
        msg = f"Only 1 gcf file allowed; {len(fnames)} provided. Only taking first."
        warnings.warn(msg, IndexError)
    else:
        fname = fnames

    # one gcf file with all three components.
    if isinstance(fname, (str, pathlib.Path)):
        traces = obspy.read(fname, **read_kwargs)

    if len(traces) != 3:
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

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

    meta = {"File Name(s)": fname}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

def read_peer(fnames, read_kwargs=None):
    if read_kwargs is None:
        read_kwargs = {}

    if not isinstance(fnames, (list, tuple)):
        msg = f"Must provide 3 peer files (one per trace) as list or tuple, not {type(fnames)}."
        raise ValueError(msg)

    component_list = []
    component_keys = []
    for fname in fnames:
        with open(fname, "r") as f:
            text = f.read()
        
        component_keys.append(peer_direction_exec.search(text).groups()[0])

        npts = int(peer_npts_exec.search(text).groups()[0])
        dt = float(peer_dt_exec.search(text).groups()[0])
        
        amplitude = np.empty((npts))
        npts_iter = 0
        for group in peer_sample_exec.finditer(text):
            sample, = group.groups()
            amplitude[npts_iter] = sample
            npts_iter += 1
        
        if npts_iter != npts:
            msg = f"Points listed in file header ({npts}) does not match the number of points found ({npts_iter})"
            msg += " please report this issue to the hvsrpy developers via GitHub issues"
            msg += " (https://github.com/jpvantassel/hvsrpy/issues)."
            raise ValueError(msg)
        
        component_list.append(TimeSeries(amplitude, dt=dt))
        
    # organize components - vertical
    vt_id = component_keys.index("UP")
    vt = component_list[vt_id]
    del component_list[vt_id], component_keys[vt_id]
    # organize components - horizontals
    component_keys_abs = np.array(component_keys, dtype=int)
    component_keys_rel = component_keys_abs.copy()
    component_keys_rel[component_keys_abs>180] -= 360
    ns_id = np.argmin(component_keys_rel)
    ns = component_list[ns_id]
    ew_id = np.argmax(component_keys_rel)
    ew = component_list[ew_id]
    del component_list, component_keys

    meta = {"File Name(s)": fname}
    return SeismicRecording3C(ns, ew, vt, meta=meta)

READ_FUNCTION_DICT = { 
                      "mseed":read_mseed,
                      "saf":read_saf,
                      "minishark":read_minishark,
                      "sac":read_sac,
                      "gcf":read_gcf,
                      "peer":read_peer
                    }

# TODO (jpv): Refactor.
# TODO (jpv): Add logging in-lieu of prints and warnings.

def read_data(fnames, read_kwargs=None):
    """

    """
    print(fnames)
    for ftype, read_function in READ_FUNCTION_DICT.items():
        try:
            srecording_3c = read_function(fnames, read_kwargs=read_kwargs)
        except Exception as e:
            warnings.warn(f"Tried reading as {ftype}, got exception |  {e}")
            pass
        else:
            print(f"File type identified as {ftype}.")
            break
    else:
        raise ValueError("File format not recognized.")
    return srecording_3c
