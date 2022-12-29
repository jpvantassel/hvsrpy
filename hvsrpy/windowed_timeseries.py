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
    
"""TimeSeries class definition."""

import warnings
import logging

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt, detrend

logger = logging.getLogger("hvsrpy.timeseries")

__all__ = ['WindowedTimeSeries']


class WindowedTimeSeries():
    """Essentially a list of TimeSeries."""

    def __init__(self, windows):
        """
        Collection of TimeSeries representing windows.

        TODO(jpv): Finish documentation

        """
        self.windows = []
        example = windows[0]
        for idx, window in enumerate(windows):

            if not example.is_similar(window):
                msg = f"windows[0] and windows[{idx}] are not similar;"
                msg += "all windows must be similar."
                raise ValueError(msg)

            self.windows.append(example.from_timeseries(window))

    def join(self):
            # def join(self):
    #     """
    #     Rejoin a split `TimeSeries`.
    #     Returns
    #     -------
    #     None
    #         Updates the object's internal attributes
    #         (e.g., `amplitude`).
    #     """
    #     nth = self.nsamples_per_window
    #     keep_ids = np.ones(self._amp.size, dtype=bool)
    #     keep_ids[nth::nth] = False
    #     self._amp = np.expand_dims(self._amp.flatten()[keep_ids], axis=0)

        # return TimeSeries
        pass

    # def trim(self):
        # need to join, trim, and split ... maybe dont need this?
            # nseries_before_join = int(self.nseries)
        # if self.nseries > 1:
        #     windowlength = self.windowlength
        #     warnings.warn("nseries > 1, so joining before splitting.")
        #     self.join()


        # if nseries_before_join > 1:
        #     self.split(windowlength)


    @property
    def nsamples_per_window(self):
        return self.windows[0].nsamples

    @property
    def windowlength(self):
        return (self.nsamples_per_window-1)*self.dt

    # @property
    # def n_windows(self):
    #     warnings.warn("`n_windows` is deprecated, use `nwindows` instead",
    #                   DeprecationWarning)
    #     return self.nwindows

    # @property
    # def nwindows(self):
    #     warnings.warn("`nwindows` is deprecated, use `nseries` instead",
    #                   DeprecationWarning)
    #     return self.nseries

    # @property
    # def nseries(self):
    #     return self._amp.shape[0]


