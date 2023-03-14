import numpy as np

from .hvsr_diffuse_field import HvsrDiffuseField
from .hvsr_traditional import HvsrTraditional
from .hvsr_azimuthal import HvsrAzimuthal


def _nested_dictionary_to_lines(data, key=None):
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            _lines = _nested_dictionary_to_lines(value, key=key)
            lines.append(f"{key}:\n")
            _lines = [f"\t{line}" for line in _lines]
            lines.extend(_lines)
        else:
            if key == "frequency_resampling_in_hz":
                continue
            line = f"{key}: {value}\n"
            lines.append(line)
    return lines

# TODO(jpv): Remove distirubtion_mean_curve.
def write_hvsr_to_file(hvsr, fname, distribution_mean_curve="lognormal"):
    """Writes HVSR object to text-based file.
    
    Parameters
    ----------
    hvsr : {HvsrTraditional, HvsrAzimuthal, HvsrDiffuseField}
        HVSR object that should be archived to a file on disk.
    fname : str
        Name of output file where the contents of the HVSR object are
        to be stored. May be a relative or the full path.

    Returns
    -------
    None
        Instead writes HVSR object to disk.

    """
    header_lines = _nested_dictionary_to_lines(hvsr.meta)
    header = "".join(header_lines)
    
    if isinstance(hvsr, HvsrDiffuseField):
        categories = ["frequency (Hz)", "hvsr curve 1", f"mean curve", f"mean curve std"]
        last_header_line = ",".join(categories)
        array = np.empty((len(hvsr.frequency), len(categories)))
        array[:, 0] = hvsr.frequency
        array[:, 1] = hvsr.amplitude
        array[:, -2] = hvsr.mean_curve(distribution=distribution_mean_curve)
        array[:, -1] = 0
    elif isinstance(hvsr, HvsrTraditional):
        categories = ["frequency (Hz)"]
        categories.extend([f"hvsr curve {x}" for x in range(1, hvsr.n_curves+1)])
        categories.extend([f"mean curve ({distribution_mean_curve})", f"mean curve std ({distribution_mean_curve})"])
        last_header_line = ",".join(categories)
        array = np.empty((len(hvsr.frequency), len(categories)))
        array[:, 0] = hvsr.frequency
        array[:, 1:-2] = hvsr.amplitude.T
        array[:, -2] = hvsr.mean_curve(distribution=distribution_mean_curve)
        array[:, -1] = hvsr.std_curve(distribution=distribution_mean_curve)
    elif isinstance(hvsr, HvsrAzimuthal):
        categories = ["frequency (Hz)"]
        # Note: HvsrAzimuthal does not require each HvsrTraditional to have the same number of curves.
        _categories = []
        for azimuth, _hvsr in zip(hvsr.azimuths, hvsr.hvsrs):
            for curve_idx in range(1, _hvsr.n_curves+1):
                _categories.append(f"azimuth {azimuth}|hvsr curve {curve_idx}")
        categories.extend(_categories)
        categories.extend([f"mean curve ({distribution_mean_curve})", f"mean curve std ({distribution_mean_curve})"])
        last_header_line = ",".join(categories)
        array = np.empty((len(hvsr.frequency), len(categories)))
        array[:, 0] = hvsr.frequency
        start_index = 1
        for hvsr in hvsr.hvsrs:
            stop_index = start_index + hvsr.n_curves
            array[:, start_index:stop_index] = hvsr.amplitude.T
            start_index = stop_index
        array[:, -2] = hvsr.mean_curve(distribution=distribution_mean_curve)
        array[:, -1] = hvsr.std_curve(distribution=distribution_mean_curve)
    else:
        raise NotImplementedError
    
    header = "".join([header, last_header_line])

    np.savetxt(fname, array, delimiter=",", header=header, encoding="utf-8")


def read_hvsr_from_file(fname):
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
    pass
    # return hvsr















#     # def print_stats(self, distribution_f0, places=2):  # pragma: no cover
#     #     """Print basic statistics of `Hvsr` instance."""
#     #     display(self._stats(distribution_f0=distribution_f0).round(places))

#     def _geopsy_style_lines(self, distribution_f0, distribution_mc):
#         """Lines for Geopsy-style file."""
#         # f0 from windows
#         mean = self.mean_f0_frq(distribution_f0)
#         lower = self.nstd_f0_frq(-1, distribution_f0)
#         upper = self.nstd_f0_frq(+1, distribution_f0)

#         # mean curve
#         mc = self.mean_curve(distribution_mc)
#         mc_peak_frq = self.mc_peak_frq(distribution_mc)
#         mc_peak_amp = self.mc_peak_amp(distribution_mc)
#         _min = self.nstd_curve(-1, distribution_mc)
#         _max = self.nstd_curve(+1, distribution_mc)

#         def fclean(number, decimals=4):
#             return np.round(number, decimals=decimals)

#         lines = [
#             f"# hvsrpy output version {__version__}",
#             f"# Number of windows = {sum(self.valid_window_boolean_mask)}",
#             f"# f0 from average\t{fclean(mc_peak_frq)}",
#             f"# Number of windows for f0 = {sum(self.valid_window_boolean_mask)}",
#             f"# f0 from windows\t{fclean(mean)}\t{fclean(lower)}\t{fclean(upper)}",
#             f"# Peak amplitude\t{fclean(mc_peak_amp)}",
#             f"# Position\t{0} {0} {0}",
#             "# Category\tDefault",
#             "# Frequency\tAverage\tMin\tMax",
#         ]

#         _lines = []
#         for line in lines:
#             _lines.append(line+"\n")

#         for f_i, a_i, n_i, x_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
#             _lines.append(f"{f_i}\t{a_i}\t{n_i}\t{x_i}\n")

#         return _lines

#     def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
#         """Lines for hvsrpy-style file."""
#         # Correct distribution
#         distribution_f0 = self.correct_distribution(distribution_f0)
#         distribution_mc = self.correct_distribution(distribution_mc)

#         # f0 from windows
#         mean_f = self.mean_f0_frq(distribution_f0)
#         sigm_f = self.std_f0_frq(distribution_f0)
#         ci_68_lower_f = self.nstd_f0_frq(-1, distribution_f0)
#         ci_68_upper_f = self.nstd_f0_frq(+1, distribution_f0)

#         # mean curve
#         mc = self.mean_curve(distribution_mc)
#         mc_peak_frq = self.mc_peak_frq(distribution_mc)
#         mc_peak_amp = self.mc_peak_amp(distribution_mc)
#         _min = self.nstd_curve(-1, distribution_mc)
#         _max = self.nstd_curve(+1, distribution_mc)

#         n_rejected = self.nseries - sum(self.valid_window_boolean_mask)
#         rejection = "False" if self.meta.get(
#             'Performed Rejection') is None else "True"
#         lines = [
#             f"# hvsrpy output version {__version__}",
#             f"# File Name (),{self.meta.get('File Name')}",
#             f"# Method (),{self.meta.get('method')}",
#             f"# Azimuth (),{self.meta.get('azimuth')}",
#             f"# Window Length (s),{self.meta.get('Window Length')}",
#             f"# Total Number of Windows (),{self.nseries}",
#             f"# Frequency Domain Window Rejection Performed (),{rejection}",
#             f"# Lower frequency limit for peaks (Hz),{self.f_low}",
#             f"# Upper frequency limit for peaks (Hz),{self.f_high}",
#             f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
#             f"# Number of Accepted Windows (),{self.nseries-n_rejected}",
#             f"# Number of Rejected Windows (),{n_rejected}",
#             f"# Distribution of f0 (),{distribution_f0}"]

#         def fclean(number, decimals=4):
#             return np.round(number, decimals=decimals)

#         distribution_f0 = self.correct_distribution(distribution_f0)
#         distribution_mc = self.correct_distribution(distribution_mc)

#         if distribution_f0 == "lognormal":
#             mean_t = 1/mean_f
#             sigm_t = sigm_f
#             ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
#             ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

#             lines += [
#                 f"# Median f0 (Hz) [LMf0],{fclean(mean_f)}",
#                 f"# Lognormal standard deviation f0 () [SigmaLNf0],{fclean(sigm_f)}",
#                 f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
#                 f"# Median T0 (s) [LMT0],{fclean(mean_t)}",
#                 f"# Lognormal standard deviation T0 () [SigmaLNT0],{fclean(sigm_t)}",
#                 f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
#             ]

#         else:
#             lines += [
#                 f"# Mean f0 (Hz),{fclean(mean_f)}",
#                 f"# Standard deviation f0 (Hz) [Sigmaf0],{fclean(sigm_f)}",
#                 f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
#                 "# Mean T0 (s) [LMT0],NA",
#                 "# Standard deviation T0 () [SigmaT0],NA",
#                 "# 68 % Confidence Interval T0 (s),NA",
#             ]

#         c_type = "Median" if distribution_mc == "lognormal" else "Mean"
#         lines += [
#             f"# {c_type} Curve Distribution (),{distribution_mc}",
#             f"# {c_type} Curve Peak Frequency (Hz) [f0mc],{fclean(mc_peak_frq)}",
#             f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
#             f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
#         ]

#         _lines = []
#         for line in lines:
#             _lines.append(line+"\n")

#         for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
#             _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

#         return _lines

#     def to_file(self, fname, distribution_f0, distribution_mc, data_format="hvsrpy"):
#         """Save HVSR data to summary file.

#         Parameters
#         ----------
#         fname : str
#             Name of file to save the results, may be a full or
#             relative path.
#         distribution_f0 : {'lognormal', 'normal'}, optional
#             Assumed distribution of `f0` from the time windows, the
#             default is 'lognormal'.
#         distribution_mc : {'lognormal', 'normal'}, optional
#             Assumed distribution of mean curve, the default is
#             'lognormal'.
#         data_format : {'hvsrpy', 'geopsy'}, optional
#             Format of output data file, default is 'hvsrpy'.

#         Returns
#         -------
#         None
#             Writes file to disk.

#         """
#         if data_format == "geopsy":
#             lines = self._geopsy_style_lines(distribution_f0, distribution_mc)
#         elif data_format == "hvsrpy":
#             lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)
#         else:
#             raise NotImplementedError(f"data_format={data_format} is unknown.")

#         with open(fname, "w") as f:
#             for line in lines:
#                 f.write(line)


#     def _stats(self, distribution_f0):
#         distribution_f0 = self.correct_distribution(distribution_f0)

#         if distribution_f0 == "lognormal":
#             columns = ["Lognormal Median", "Lognormal Standard Deviation"]
#             data = np.array([[self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)],
#                              [1/self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)]])

#         elif distribution_f0 == "normal":
#             columns = ["Mean", "Standard Deviation"]
#             data = np.array([[self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)],
#                              [np.nan, np.nan]])
#         else:
#             msg = f"`distribution_f0` of {distribution_f0} is not implemented."
#             raise NotImplementedError(msg)

#         df = DataFrame(data=data, columns=columns,
#                        index=["Fundamental Site Frequency, f0",
#                               "Fundamental Site Period, T0"])
#         return df


# ### HvsrAzimuthal

#     def _stats(self, distribution_f0):
#         distribution_f0 = Hvsr.correct_distribution(distribution_f0)

#         if distribution_f0 == "lognormal":
#             columns = ["Lognormal Median", "Lognormal Standard Deviation"]
#             data = np.array([[self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)],
#                              [1/self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)]])

#         elif distribution_f0 == "normal":
#             columns = ["Means", "Standard Deviation"]
#             data = np.array([[self.mean_f0_frq(distribution_f0),
#                               self.std_f0_frq(distribution_f0)],
#                              [np.nan, np.nan]])
#         else:
#             msg = f"`distribution_f0` of {distribution_f0} is not implemented."
#             raise NotImplementedError(msg)

#         df = DataFrame(data=data, columns=columns,
#                        index=["Fundamental Site Frequency, f0,AZ",
#                               "Fundamental Site Period, T0,AZ"])
#         return df

#     # def print_stats(self, distribution_f0, places=2):  # pragma: no cover
#     #     """Print basic statistics of `Hvsr` instance."""
#     #     display(self._stats(distribution_f0=distribution_f0).round(places))

#     def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
#         """Lines for hvsrpy-style file."""
#         # Correct distribution
#         distribution_f0 = Hvsr.correct_distribution(distribution_f0)
#         distribution_mc = Hvsr.correct_distribution(distribution_mc)

#         # `f0` from windows
#         mean_f = self.mean_f0_frq(distribution_f0)
#         sigm_f = self.std_f0_frq(distribution_f0)
#         ci_68_lower_f = self.nstd_f0_frq(-1, distribution_f0)
#         ci_68_upper_f = self.nstd_f0_frq(+1, distribution_f0)

#         # mean curve
#         mc = self.mean_curve(distribution_mc)
#         mc_peak_frq = self.mc_peak_frq(distribution_mc)
#         mc_peak_amp = self.mc_peak_amp(distribution_mc)
#         _min = self.nstd_curve(-1, distribution_mc)
#         _max = self.nstd_curve(+1, distribution_mc)

#         rejection = "False" if self.meta.get('Performed Rejection') is None else "True"

#         nseries = self.hvsrs[0].nseries
#         n_accepted = sum([sum(hvsr.valid_window_indices) for hvsr in self.hvsrs])
#         n_rejected = self.azimuth_count*nseries - n_accepted
#         lines = [
#             f"# hvsrpy output version {__version__}",
#             f"# File Name (),{self.meta.get('File Name')}",
#             f"# Window Length (s),{self.meta.get('Window Length')}",
#             f"# Total Number of Windows per Azimuth (),{nseries}",
#             f"# Total Number of Azimuths (),{self.azimuth_count}",
#             f"# Total Number of Windows (),{nseries*self.azimuth_count}",
#             f"# Frequency Domain Window Rejection Performed (),{rejection}",
#             f"# Lower frequency limit for peaks (Hz),{self.hvsrs[0].f_low}",
#             f"# Upper frequency limit for peaks (Hz),{self.hvsrs[0].f_high}",
#             f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
#             f"# Number of Accepted Windows (),{n_accepted}",
#             f"# Number of Rejected Windows (),{n_rejected}",
#             f"# Distribution of f0 (),{distribution_f0}"]

#         def fclean(number, decimals=4):
#             return np.round(number, decimals=decimals)

#         if distribution_f0 == "lognormal":
#             mean_t = 1/mean_f
#             sigm_t = sigm_f
#             ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
#             ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

#             lines += [
#                 f"# Median f0 (Hz) [LMf0AZ],{fclean(mean_f)}",
#                 f"# Lognormal standard deviation f0 () [SigmaLNf0AZ],{fclean(sigm_f)}",
#                 f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
#                 f"# Median T0 (s) [LMT0AZ],{fclean(mean_t)}",
#                 f"# Lognormal standard deviation T0 () [SigmaLNT0AZ],{fclean(sigm_t)}",
#                 f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
#             ]

#         else:
#             lines += [
#                 f"# Mean f0 (Hz) [f0AZ],{fclean(mean_f)}",
#                 f"# Standard deviation f0 (Hz) [Sigmaf0AZ],{fclean(sigm_f)}",
#                 f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
#                 "# Mean T0 (s) [LMT0AZ],NAN",
#                 "# Standard deviation T0 () [SigmaT0AZ],NAN",
#                 "# 68 % Confidence Interval T0 (s),NAN",
#             ]

#         c_type = "Median" if distribution_mc == "lognormal" else "Mean"
#         lines += [
#             f"# {c_type} Curve Distribution (),{distribution_mc}",
#             f"# {c_type} Curve Peak Frequency (Hz) [f0mcAZ],{fclean(mc_peak_frq)}",
#             f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
#             f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
#         ]

#         _lines = []
#         for line in lines:
#             _lines.append(line+"\n")

#         for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
#             _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

#         return _lines

#     def to_file(self, fname, distribution_f0, distribution_mc, data_format="hvsrpy"):
#         """Save HVSR data to file.

#         Parameters
#         ----------
#         fname : str
#             Name of file to save the results, may be the full or a
#             relative path.
#         distribution_f0 : {'lognormal', 'normal'}, optional
#             Assumed distribution of `f0` from the time windows, the
#             default is 'lognormal'.
#         distribution_mc : {'lognormal', 'normal'}, optional
#             Assumed distribution of mean curve, the default is
#             'lognormal'.
#         data_format : {'hvsrpy'}, optional
#             Format of output data file, default is 'hvsrpy'.

#         Returns
#         -------
#         None
#             Writes file to disk.

#         """
#         if data_format not in ["hvsrpy"]:
#             raise ValueError(f"data_format {data_format} unknown.")

#         lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)

#         with open(fname, "w") as f:
#             for line in lines:
#                 f.write(line)

# def parse_hvsrpy_output(fname):
#     """Parse an hvsrpy output file for revalent information.

#     Parameters
#     ----------
#     fname : str
#         Name of file to be parsed may include a relative or full path.

#     Returns
#     -------
#     dict
#         With revalent information as key value pairs.

#     """
#     data = {}
#     frqs, meds, lows, higs = [], [], [], []

#     lookup = {"# Window Length (s)": ("windowlength", float),
#               "# Total Number of Windows ()": ("total_windows", int),
#               "# Frequency Domain Window Rejection Performed ()": ("rejection_bool", bool),
#               "# Lower frequency limit for peaks (Hz)": ("f_low", lambda x: None if x == "None" else float(x)),
#               "# Upper frequency limit for peaks (Hz)": ("f_high", lambda x: None if x == "None" else float(x)),
#               "# Number of Standard Deviations Used for Rejection () [n]": ("n_for_rejection", float),
#               "# Number of Accepted Windows ()": ("accepted_windows", int),
#               "# Distribution of f0 ()": ("distribution_f0", lambda x: x),
#               "# Mean f0 (Hz)": ("mean_f0", float),
#               "# Standard deviation f0 (Hz) [Sigmaf0]": ("std_f0", float),
#               "# Median Curve Distribution ()": ("distribution_mc", lambda x: x),
#               "# Median Curve Peak Frequency (Hz) [f0mc]": ("f0_mc", float),
#               "# Median Curve Peak Amplitude ()": ("amplitude_f0_mc", float)
#               }

#     with open(fname, "r") as f:
#         for line in f:
#             if line.startswith("#"):
#                 try:
#                     key, value = line.split(",")
#                 except ValueError:
#                     continue

#                 try:
#                     subkey, operation = lookup[key]
#                     data[subkey] = operation(value.rstrip())
#                 except KeyError:
#                     continue
#             else:
#                 frq, med, low, hig = line.split(",")

#                 frqs.append(frq)
#                 meds.append(med)
#                 lows.append(low)
#                 higs.append(hig)

#     data["frequency"] = np.array(frqs, dtype=np.double)
#     data["curve"] = np.array(meds, dtype=np.double)
#     data["lower"] = np.array(lows, dtype=np.double)
#     data["upper"] = np.array(higs, dtype=np.double)

#     return data

# # """This file contains the class Hvsr for organizing data
# # related to the horizontal-to-vertical spectral ratio method."""

# # import os
# # import glob
# # import re
# # import logging
# # logger = logging.getLogger(__name__)


# class Hvsr():
#     def __init__(self, frequency, amplitude, identifier):
#         self.frq = [frequency]
#         self.amp = [amplitude]
#         self.idn = identifier

#     def append(self, frequency, amplitude):
#         for cfreq, nfreq in zip(self.frq[0], frequency):
#             if cfreq!=nfreq:
#                 raise ValueError(f"appended f {cfreq} != existing f{nfreq}")
#         self.frq.append(frequency)
#         self.amp.append(amplitude)

#     @classmethod
#     def from_geopsy_folder(cls, dirname, identifier):
#         logging.info(f"Reading .hv files from {dirname}")
#         fnames = glob.glob(dirname+"/*.hv")
#         logging.debug(f"File names to load are {fnames}")
#         logging.info(f"Starting file {fnames[0]}")
#         obj = cls.from_geopsy_file(fnames[0], identifier)
#         for fname in fnames[1:]:
#             logging.info(f"Starting file {fname}")
#             tmp_obj = cls.from_geopsy_file(fname, "temp")
#             obj.append(tmp_obj.frq[0], tmp_obj.amp[0])
#         return obj

#     @classmethod
#     def from_geopsy_file(cls, fname, identifier):
#         with open(fname, "r") as f:
#             lines = f.read().splitlines()

#         for num, line in enumerate(lines):
#             if line.startswith("# Frequency"):
#                 start_line = num + 1
#                 break

#         frq, amp = [], []
#         for line in lines[start_line:]:
#             fr, am = re.findall(r"^(\d+.?\d*)\t(\d+.?\d*)\t\d+.?\d*\t\d+.?\d*$", line)[0]
#             frq.append(float(fr))
#             amp.append(float(am))

#         return cls(frq, amp, identifier)

 
# # def resonance_identification_manual(self,
# #                                     distribution_mc='lognormal',
# #                                     find_peaks_kwargs=None,
# #                                     ylims=None):
# #     if find_peaks_kwargs is None:
# #         raise NotImplementedError
# #     (valid_idxs, _) = self.find_peaks(self.amp, **find_peaks_kwargs)

# #     frqs, amps = [], []
# #     for amp, col_ids in zip(self.amp, valid_idxs):
# #         frqs.extend(self.frq[col_ids])
# #         amps.extend(amp[col_ids])
# #     peaks_valid = (frqs, amps)
# #     peaks_invalid = ([], [])

# #     fig, ax = single_plot(
# #         self, peaks_valid, peaks_invalid, distribution_mc, ylims=ylims)
# #     ax.autoscale(enable=False)
# #     fig.show()
# #     pxlim = ax.get_xlim()
# #     pylim = ax.get_ylim()

# #     # continue button
# #     upper_right_corner = (0.05, 0.95)
# #     _xc, _yc = upper_right_corner
# #     box_size = 0.1
# #     scale_x = (np.log10(max(pxlim)) - np.log10(min(pxlim)))
# #     scale_y = max(pylim) - min(pylim)
# #     x_lower, x_upper = np.exp(_xc*scale_x + np.log10(min(pxlim))
# #                                 ), np.exp((_xc+box_size)*scale_x + np.log10(min(pxlim)))
# #     y_lower, y_upper = (_yc - box_size)*scale_y + \
# #         min(pylim), _yc*scale_y + min(pylim)

# #     def draw_continue_box(ax):
# #         ax.fill([x_lower, x_upper, x_upper, x_lower],
# #                 [y_upper, y_upper, y_lower, y_lower], color="lightgreen")
# #         ax.text(_xc, _yc-box_size/2, "continue?", ha="left",
# #                 va="center", transform=ax.transAxes)
# #     draw_continue_box(ax)

# #     f_lows, f_highs = [], []
# #     while True:
# #         xs, ys = ginput_session(fig, ax, initial_adjustment=False,
# #                                 npts=2, ask_to_confirm_point=False, ask_to_continue=False)

# #         in_continue_box = False
# #         for _x, _y in zip(xs, ys):
# #             if (_x < x_upper) and (_x > x_lower) and (_y > y_lower) and (_y < y_upper):
# #                 in_continue_box = True
# #                 break

# #         if in_continue_box:
# #             plt.close()
# #             break
# #         else:
# #             f_lows.append(min(xs))
# #             f_highs.append(max(xs))
# #             continue

# #     f_lows.sort()
# #     f_highs.sort()
# #     limits = [[fl, fh] for fl, fh in zip(f_lows, f_highs)]
# #     limits = np.array(limits)

# #     return limits