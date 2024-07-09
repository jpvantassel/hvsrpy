# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2024 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
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

"""Class definition for HvsrDiffuseField object."""

import logging

from .hvsr_curve import HvsrCurve

logger = logging.getLogger(__name__)

__all__ = ["HvsrDiffuseField"]

class HvsrDiffuseField(HvsrCurve):
    
    def mean_curve(self, distribution=None):
        """Mean HVSR curve according to diffuse field theory.

        Parameters
        ----------
        distribution : None, optional
            Not used only kept for symmetry with other HVSR objects.

        Returns
        -------
        ndarray
            Mean HVSR curve according to diffuse field theory.

        """
        return self.amplitude

    def mean_curve_peak(self, distribution=None,
                        search_range_in_hz=(None, None),
                        find_peaks_kwargs=None):
        """Frequency and amplitude of the peak of the mean HVSR curve.

        Parameters
        ----------
        distribution : None, optional
            Not used only kept for symmetry with other HVSR objects.
        search_range_in_hz : tuple, optional
            Frequency range to be searched for peaks.
            Half open ranges can be specified with ``None``, default is
            ``(None, None)`` indicating the full frequency range will be
            searched.
        find_peaks_kwargs : dict
            Keyword arguments for the ``scipy`` function
            `find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
            see ``scipy`` documentation for details, default is ``None``
            indicating defaults will be used.

        Returns
        -------
        tuple
            Frequency and amplitude associated with the peak of the mean
            HVSR curve of the form
            ``(mean_curve_peak_frequency, mean_curve_peak_amplitude)``.

        """
        amplitude = self.mean_curve()
        f_peak, a_peak = HvsrCurve._find_peak_bounded(self.frequency,
                                                      amplitude,
                                                      search_range_in_hz=search_range_in_hz,
                                                      find_peaks_kwargs=find_peaks_kwargs)

        if f_peak is None or a_peak is None:
            msg = "Mean curve does not have a peak in the specified range."
            raise ValueError(msg)

        return (f_peak, a_peak)
