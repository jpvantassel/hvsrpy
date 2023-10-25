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
import io

import obspy
import numpy as np

from .regex import saf_npts_exec, saf_fs_exec, saf_row_exec, saf_v_ch_exec, saf_n_ch_exec, saf_e_ch_exec, saf_north_rot_exec, saf_version_exec
from .regex import mshark_npts_exec, mshark_fs_exec, mshark_gain_exec, mshark_conversion_exec, mshark_row_exec
from .regex import peer_direction_exec, peer_npts_exec, peer_dt_exec, peer_sample_exec

from .timeseries import TimeSeries
from .seismic_recording_3c import SeismicRecording3C

logger = logging.getLogger(__name__)


def _arrange_traces(traces):
    """Sort ``list`` of 3 ``Trace`` objects according to direction."""
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
        else: # pragma: no cover
            msg = "Missing, duplicate, or incorrectly named components."
            raise ValueError(msg)
    return ns, ew, vt


def _check_npts(npts_header, npts_found):
    if npts_header != npts_found: # pragma: no cover
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


def _read_mseed(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in miniSEED format.

    .. warning::
        Private API is subject to change without warning.

    Parameters
    ----------
    fnames : {str, list}
        If ``str`` then ``fnames`` is the name of the miniSEED file,
        full path may be used if desired. The file should contain
        three traces with the appropriate channel names. Refer to
        the 
        `SEED Manual <https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.
        for specifics.
        If ``list`` then ``fnames`` is a list of length three with the
        names of miniSEED files for each component.
    obspy_read_kwargs : dict, optional
        For passing arguments to the ``obspy.read`` function to
        customize its behavior, default is ``None`` indicating
        not keyword arguments will be passed.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "MSEED"}

    # one miniSEED file with all three components.
    if isinstance(fnames, (str, pathlib.Path, io.BytesIO)):
        traces = _quiet_obspy_read(fnames, **obspy_read_kwargs)
        fnames = str(fnames)
    # three miniSEED files; one per component.
    elif isinstance(fnames, (list, tuple)):
        trace_list = []
        for fname in fnames:
            stream = _quiet_obspy_read(fname, **obspy_read_kwargs)
            if len(stream) != 1: # pragma: no cover
                msg = f"File {fname} contained {len(stream)}"
                msg += "traces, rather than 1 as was expected."
                raise IndexError(msg)
            trace = stream[0]
            trace_list.append(trace)
        traces = obspy.Stream(trace_list)
        fnames = [str(fname) for fname in fnames]
    else: # pragma: no cover
        msg = "`fnames` must be either `str` or `list`"
        msg += f"cannot be {type(fnames)}."
        raise ValueError(msg)

    if len(traces) != 3: # pragma: no cover
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"file name(s)": fnames}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def _read_saf(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in SESAME ASCII format (SAF).

    .. warning::
        Private API is subject to change without warning.

    Parameters
    ----------
    fnames : str
        Name of the SESAME ASCII format file, full path may be used if
        desired. The file should contain three traces with the
        appropriate channel names. See  
        `SESAME standard <http://sesame.geopsy.org/Delivrables/D09-03_Texte.pdf>`_.
    obspy_read_kwargs : dict, optional
        Ignored, kept only to maintain consistency with other read
        functions.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if isinstance(fnames, (list, tuple)):
        msg = f"Only 1 saf file allowed; {len(fnames)} provided. "
        raise ValueError(msg)
    elif isinstance(fnames, (io.StringIO,)):
        fname = fnames
        fname.seek(0, 0)
        text = fname.read()
    else:
        fname = fnames
        with open(fname, "r") as f:
            text = f.read()

    # ensure the file is saf format.
    _ = saf_version_exec.search(text).groups()[0]

    npts_header = int(saf_npts_exec.search(text).groups()[0])
    dt = 1/float(saf_fs_exec.search(text).groups()[0])
    v_ch = int(saf_v_ch_exec.search(text).groups()[0])
    n_ch = int(saf_n_ch_exec.search(text).groups()[0])
    e_ch = int(saf_e_ch_exec.search(text).groups()[0])
    if degrees_from_north is None:
        try:
            north_rot = float(saf_north_rot_exec.search(text).groups()[0])
        except: # pragma: no cover
            msg = f"The provided saf file {fname} does not include the "
            msg += "NORTH_ROT keyword, assuming equal to zero."
            warnings.warn(msg, UserWarning)
            degrees_from_north = 0.
        else:
            if n_ch == 1:
                degrees_from_north = north_rot
            elif e_ch == 1:
                degrees_from_north = north_rot + 90.
            else: # pragma: no cover
                msg = f"The provided saf file {fname} is not properly formatted."
                msg += " CH1 must be vertical; CH2 & CH3 the horizontals."
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

    vt = TimeSeries(vt, dt_in_seconds=dt)
    ns = TimeSeries(ns, dt_in_seconds=dt)
    ew = TimeSeries(ew, dt_in_seconds=dt)

    meta = {"file name(s)": str(fname)}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def _read_minishark(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in MiniShark format.

    .. warning::
        Private API is subject to change without warning.
    
    Parameters
    ----------
    fnames : str
        Name of the MiniShark format file, full path may be used if
        desired. The file should contain three traces with the
        appropriate channel names.
    obspy_read_kwargs : dict, optional
        Ignored, kept only to maintain consistency with other read
        functions.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if isinstance(fnames, (list, tuple)):
        msg = f"Only 1 minishark file allowed; {len(fnames)} provided."
        raise ValueError(msg)
    elif isinstance(fnames, io.StringIO):
        fnames.seek(0, 0)
        text = fnames.read()
        fname = fnames
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

    vt = TimeSeries(vt, dt_in_seconds=dt)
    ns = TimeSeries(ns, dt_in_seconds=dt)
    ew = TimeSeries(ew, dt_in_seconds=dt)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"file name(s)": str(fname)}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def _read_sac(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in Seismic Analysis Code format.

    .. warning::
        Private API is subject to change without warning.

    Parameters
    ----------
    fnames : list
        List of length three with the names of the Seismic Analysis
        Code (SAC) format files; one per component. Files can be little
        endian or big endian. Each file should the appropriate channel
        names. See 
        `SAC manual <https://ds.iris.edu/files/sac-manual/sac_manual.pdf>`_.
    obspy_read_kwargs : dict, optional
        For passing arguments to the ``obspy.read`` function to
        customize its behavior, default is ``None`` indicating
        no keyword arguments will be passed.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "SAC"}

    if not isinstance(fnames, (list, tuple)):
        msg = "Must provide 3 sac files (one per trace); only one provided."
        raise ValueError(msg)

    trace_list = []
    for fname in fnames:
        for byteorder in ["little", "big"]:
            if isinstance(fname, io.BytesIO):
                fname.seek(0,0)
            obspy_read_kwargs["byteorder"] = byteorder
            try:
                stream = _quiet_obspy_read(fname, **obspy_read_kwargs)
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

    if len(traces) != 3: # pragma: no cover
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"file name(s)": [str(fname) for fname in fnames]}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def _read_gcf(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in Guralp Compressed Format (GCF).

    .. warning::
        Private API is subject to change without warning.

    Parameters
    ----------
    fnames : str
        Name of the MiniShark Guralp Compressed Format (GCF) file, full
        path may be used if desired. The file should contain three
        traces with the appropriate channel names.
    obspy_read_kwargs : dict, optional
        For passing arguments to the ``obspy.read`` function to
        customize its behavior, default is ``None`` indicating
        no keyword arguments will be passed.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "GCF"}

    if isinstance(fnames, (list, tuple)):
        msg = f"Only 1 gcf file allowed; {len(fnames)} provided. "
        raise ValueError(msg)
    else:
        fname = fnames

    # one gcf file with all three components.
    if isinstance(fname, (str, pathlib.Path, io.BytesIO)):
        traces = _quiet_obspy_read(fname, **obspy_read_kwargs)

    if len(traces) != 3: # pragma: no cover
        msg = f"Provided {len(traces)} traces, but must only provide 3."
        raise ValueError(msg)

    ns, ew, vt = _arrange_traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.

    meta = {"file name(s)": str(fname)}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


def _read_peer(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data from file(s) in PEER format.

    .. warning::
        Private API is subject to change without warning.

    Parameters
    ----------
    fnames : list
        List of length three with the names of the Pacific Earthquake
        Engineering Research (PEER) format files; one per component.
        Each file should have appropriate channel names. Some
        information on the PEER format is provided by
        `SCEC <https://strike.scec.org/scecpedia/PEER_Data_Format>`_.
    obspy_read_kwargs : dict, optional
        Ignored, kept only to maintain consistency with other read
        functions.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is 0.0
        indicating the sensor's north component is aligned with
        magnetic north.

    Returns
    -------
    SeismicRecording3C
        Initialized 3-component seismic recording object.

    """
    if not isinstance(fnames, (list, tuple)): # pragma: no cover
        msg = "Must provide 3 peer files (one per trace) as list or tuple, "
        msg += f"not {type(fnames)}."
        raise ValueError(msg)

    component_list = []
    component_keys = []
    dts = []
    for fname in fnames:
        if isinstance(fname, io.StringIO):
            fname.seek(0, 0)
            text = fname.read()
        else:
            with open(fname, "r") as f:
                text = f.read()

        component_keys.append(peer_direction_exec.search(text).groups()[0])

        npts_header = int(peer_npts_exec.search(text).groups()[0])
        dt = float(peer_dt_exec.search(text).groups()[0])
        dts.append(dt)

        amplitude = np.empty((npts_header))
        idx = 0
        for group in peer_sample_exec.finditer(text):
            sample, = group.groups()
            amplitude[idx] = sample
            idx += 1

        _check_npts(npts_header, idx)

        component_list.append(TimeSeries(amplitude, dt_in_seconds=dt))

    # check the dt from all files are equal
    for idx, dt in enumerate(dts):
        if dt != dts[0]:
            msg = "All time steps must be equal. "
            msg += f"Time step of file {idx} is {dt} s which is not equal to "
            msg += f"that of file 0 that is {dts[0]} s."
            raise ValueError(msg)

    # organize components - vertical
    orientation_is_numeric = False
    try:
        vt_id = component_keys.index("UP")
        orientation_is_numeric = True
    except ValueError:
        try:
            vt_id = component_keys.index("VER")
            orientation_is_numeric = True
        except ValueError:
            for vt_id, _key in enumerate(component_keys):
                if _key[-1].lower() == "z":
                    break
            else:
                msg = f"Components {component_keys} in header are not recognized. "
                msg += "If you believe this is an error please contact the developer."
                raise ValueError(msg)
    vt = component_list[vt_id]
    del component_list[vt_id], component_keys[vt_id]

    # organize components - horizontals
    if orientation_is_numeric:
        component_keys_abs = np.array(component_keys, dtype=int)
        component_keys_rel = component_keys_abs.copy()
        component_keys_rel[component_keys_abs > 180] -= 360
        ns_id = np.argmin(abs(component_keys_rel))
        ns = component_list[ns_id]
        ew_id = np.argmax(abs(component_keys_rel))
        ew = component_list[ew_id]
        del component_list, component_keys
    else:
        component_keys_abs = np.zeros(3, dtype=int)
        for _id, _key in enumerate(component_keys):
            if _key[-1] == "N":
                ns_id = _id
                ns = component_list[ns_id]
            elif _key[-1] == "E":
                ew = component_list[_id]
            else:
                msg = f"Components {component_keys} in header are not recognized. "
                msg += "If you believe this is an error please contact the developer."
                raise ValueError(msg)

    # set rotation iff degrees_from_north is not already set.
    if degrees_from_north is None:
        degrees_from_north = component_keys_abs[ns_id]
        degrees_from_north = float(degrees_from_north - 360*(degrees_from_north // 360))

    # peer does not require all components to be the same length.
    # therefore trim all records to the shortest time length.
    npts = [component.n_samples for component in [ns, ew,vt]]
    ns.amplitude = ns.amplitude[:min(npts)] 
    ew.amplitude = ew.amplitude[:min(npts)] 
    vt.amplitude = vt.amplitude[:min(npts)] 

    meta = {"file name(s)": [str(fname) for fname in fnames]}
    return SeismicRecording3C(ns, ew, vt,
                              degrees_from_north=degrees_from_north, meta=meta)


READ_FUNCTION_DICT = {
    "mseed": _read_mseed,
    "saf": _read_saf,
    "minishark": _read_minishark,
    "sac": _read_sac,
    "gcf": _read_gcf,
    "peer": _read_peer
}


def read_single(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read file(s) associated with a single recording.

    Parameters
    ----------
    fnames: {list, str}
        File name(s) to be read.

        If ``str``, name of file to be read, may include a relative or
        the full path. The file should contain all three components
        (2 horizontal and 1 vertical).

        If ``list``, names of files to be read, each may be a relative
        or the full path. Each file should contain only one component.
    obspy_read_kwargs : dict, optional
        Keyword arguments to be passed directly to ``obspy.read``, in
        general this should not be needed, default is ``None`` indicating
        no custom arguments will be passed to ``obspy.read``.
    degrees_from_north : float, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive. Default is ``None``
        indicating either the metadata in the file denoting the sensor's
        orientation is correct and should be used or (if the sensor's
        orientation is not listed in the file) the sensor's north
        component is aligned with magnetic north
        (i.e., ``degrees_from_north=0``).

    Returns
    -------
    SeismicRecording3C
        Initialized three-component seismic recording object.

    """
    logger.info(f"Attempting to read {fnames}")
    for ftype, read_function in READ_FUNCTION_DICT.items():
        try:
            srecording_3c = read_function(fnames,
                                          obspy_read_kwargs=obspy_read_kwargs,
                                          degrees_from_north=degrees_from_north)
        except Exception as e:
            logger.info(f"Tried reading as {ftype}, got exception |  {e}")

            if ftype == "peer":
                raise e

            pass
        else:
            logger.info(f"File type identified as {ftype}.")
            break
    else: # pragma: no cover
        msg = "File format not recognized. Only the following are supported: "
        msg += f"{READ_FUNCTION_DICT.keys()}."
        raise ValueError(msg)
    return srecording_3c


def read(fnames, obspy_read_kwargs=None, degrees_from_north=None):
    """Read seismic data file(s).

    Parameters
    ----------
    fnames : iterable of iterable of str or iterable of str
        Collection of file name(s) to be read. All entries should be
        readable by the function ``hvsrpy.read_single()``.
    obspy_read_kwargs : dict or iterable of dicts, optional
        Keyword arguments to be passed directly to
        ``hvsrpy.read_single()``.

        If ``dict``, keyword argument will be repeated for all file
        names provided.

        If ``iterable of dicts`` each keyword arguments will be provided
        in order.

        Default is ``None`` indicating standard read behavior will be
        used.
    degrees_from_north : float or iterable of floats, optional
        Rotation in degrees of the sensor's north component relative to
        magnetic north; clock wise positive.

        If ``float``, it will be repeated for all file names provided.

        If ``iterable of floats`` each ``float`` will be provided
        in order.

        Default is ``None`` indicating either the metadata in the file
        denoting the sensor's orientation is correct and should be used
        or (if the sensor's orientation is not listed in the file) the
        sensor's north component is aligned with magnetic north
        (i.e., ``degrees_from_north=0``).

    Returns
    -------
    list
        Of initialized ``SeismicRecording3C`` objects, one for each each
        iterable entry provided.

    """
    # if only string provided put it in a list and warn user.
    if not isinstance(fnames, (list, tuple)):
        msg = "fnames should be iterable of str or iterable of "
        msg += "iterable of str."
        warnings.warn(msg)
        fnames = [fnames]

    # scale obspy_read_kwargs as needed to match fnames.
    if isinstance(obspy_read_kwargs, (dict, type(None))):
        read_kwargs_iter = itertools.repeat(obspy_read_kwargs)
    else:
        read_kwargs_iter = obspy_read_kwargs

    # scale degrees_from_north as needed to match fnames.
    if isinstance(obspy_read_kwargs, (dict, type(None))):
        degrees_from_north_iter = itertools.repeat(degrees_from_north)
    else:
        degrees_from_north_iter = degrees_from_north

    seismic_recordings = []
    for fname, read_kwargs, degrees_from_north in zip(fnames, read_kwargs_iter, degrees_from_north_iter):

        # if entry is a list with only a single entry, remove the list.
        if isinstance(fname, (list, tuple)):
            if len(fname) == 1:
                fname = fname[0]

        seismic_recordings.append(read_single(fname,
                                              obspy_read_kwargs=read_kwargs,
                                              degrees_from_north=degrees_from_north))

    return seismic_recordings
