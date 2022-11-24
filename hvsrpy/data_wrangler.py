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

from sigpropy import TimeSeries

from .seismic_recording_3c import SeismicRecording3C

def read_mseed(fnames, obspy_read_kwargs=None):
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
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {}

    # one miniSEED file with all three components.
    if isinstance(fnames, str):
        traces = obspy.read(fnames, **obspy_read_kwargs)
    # three miniSEED files, one per component.
    elif isinstance(fnames, (list, tuple)):
        trace_list = []
        for fname in fnames:
            stream = obspy.read(fname, **obspy_read_kwargs)
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

def read_saf():
    raise NotImplementedError

def read_minishark():
    raise NotImplementedError

def read_sac():
    raise NotImplementedError

def read_gcf():
    raise NotImplementedError

def read_peer():
    raise NotImplementedError

READ_FUNCTIONS = [read_mseed, read_saf, read_minishark, read_sac, read_gcf,
                  read_peer
                 ]

# TODO (jpv): Not correctly raising error with wrong file.
def read_data(fnames, obspy_read_kwargs=None):
    """

    """
    raise NotImplementedError
#     for function in READ_FUNCTIONS:
#         # try:
#         srecording_3c = function(fnames, obspy_read_kwargs=obspy_read_kwargs)
#         # except:
#         #     pass
#         # else:
#         #     break
#     else:
#         raise ValueError("File format not recognized.")
#     return srecording_3c

