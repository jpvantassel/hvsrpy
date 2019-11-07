"""File for Horizontal-to-Vertical Spectral Ratio (Hvsr) class."""

import numpy as np
import scipy.signal as sg
import logging
logging.getLogger()


class Hvsr():
    """Class for creating and manipulating horizontal-to-vertical
    spectral ratio objects.

    Attributes:
        amp : ndarray
            Array of horitzontal-to-vertical amplitudes. Each row
            represents an individual curve and each column a
            particular frequency.
        frq : ndarray
            Vector of frequencies, corresponding to each column.
        n_windows : int
            Number of windows in hvsr object.
        valid_windows : ndarray
            Array of booleans of length `n_windows` which determines
            which windows are valid.
        valid_window_indices : ndarray
            Array of indices for valid windows.
        peaks : ndarray

    """
    @staticmethod
    def check_input(name, value):
        pass

    def __init__(self, amplitude, frequency, find_peaks=True):
        """Initialize a Hvsr oject from amplitude and frequency vector.

        Args:
            amplitude : np.array
                See class attribute documentation.
            frequency : np.array
                See class attribute documentation.

        Returns:
            Initialized Hvsr object.
        """
        # TODO (jpv): Add check, see method above.
        self.amp = amplitude
        self.frq = frequency
        self.n_windows = self.amp.shape[0] if len(self.amp.shape) > 1 else 1
        self.valid_window_indices = np.arange(self.n_windows)
        self.master_peak_frq = np.zeros(self.n_windows)
        self.master_peak_amp = np.zeros(self.n_windows)
        self.initialized_peaks = find_peaks
        if find_peaks:
            self.update_peaks()

    @property
    def peak_frq(self):
        """Return valid peaks frequency vector."""
        if not self.initialized_peaks:
            self.update_peaks()
        return self.master_peak_frq[self.valid_window_indices]

    @property
    def peak_amp(self):
        """Return valid peaks ampltiude vector."""
        if not self.initialized_peaks:
            self.update_peaks()
        return self.master_peak_amp[self.valid_window_indices]

    @staticmethod
    def find_peaks(amp, **kwargs):
        """Returns index of all peaks in `amp`.

        Wrapper method for scipy.signal.find_peaks function.

        Args:
            amp : ndarray
                Vector or array of amplitudes.
            **kwargs : various
                Refer to `scipy.signal.find_peaks` documentation:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        Returns:
            peaks : np.array or list
                np.array or list of np.arrays (one per window) of peak
                indices.

            properties : various
                Refer to `scipy.signal.find_peaks` documentation.
        """
        if len(amp.shape) == 1:
            peaks, settings = sg.find_peaks(amp, **kwargs)
            return (peaks, settings)
        else:
            peaks = []
            for c_amp in amp:
                peak, settings = sg.find_peaks(c_amp, **kwargs)
                peaks.append(peak)
            return (peaks, settings)

    def update_peaks(self, **kwargs):
        """Update `peaks` attribute using the lowest frequency, highest
        amplitude peak."""
        if not self.initialized_peaks:
            self.initialized_peaks=True

        if self.n_windows == 1:
            peak_indices, _ = self.find_peaks(self.amp, **kwargs)
            c_index = np.where(self.amp == np.max(self.amp[peak_indices]))
            self.master_peak_amp = self.amp[c_index]
            self.master_peak_frq = self.frq[c_index]
            return

        peak_indices, _ = self.find_peaks(self.amp[self.valid_window_indices],
                                          **kwargs)
        valid_indices = []
        for c_window, c_window_peaks in zip(self.valid_window_indices, peak_indices):
            try:
                c_index = np.where(self.amp[c_window] == np.max(self.amp[c_window, c_window_peaks]))
                self.master_peak_amp[c_window] = self.amp[c_window, c_index]
                self.master_peak_frq[c_window] = self.frq[c_index]
                valid_indices.append(c_window)
            except:
                assert(c_window_peaks.size == 0)
                logging.warning(f"No peak found in window #{c_window}.")
        self.valid_window_indices = np.array(valid_indices)

    def mean_f0(self, distribution='log-normal'):
        """Return mean value of f0 of valid timewindows using different
        distribution.

        Args:
            distribution : {'normal', 'log-normal'}

        Return:
            mean : float
                Mean value according to the distribution specified.
        """
        if distribution == "normal":
            return np.mean(self.peak_frq)
        elif distribution == "log-normal":
            return np.exp(np.mean(np.log(self.peak_frq)))
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def std_f0(self, distribution='log-normal'):
        """Return std_deviation value of f0 using different
        distributions.

        Args:
            distribution : {'normal', 'log-normal'}

        Return:
            std : float
                Sample standard deviation value according to the
                distribution specified.
        """
        if distribution == "normal":
            return np.std(self.peak_frq, ddof=1)
        elif distribution == "log-normal":
            return np.std(np.log(self.peak_frq), ddof=1)
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def mean_curve(self, distribution='log-normal'):
        """Return mean hvsr curve.

        Args:
            distribution : {'normal', 'log-normal'}

        Return:
            mean : ndarray
                Mean hvsr curve according to the distribution specified.
        """
        if self.n_windows == 1:
            return self.amp

        if distribution == "normal":
            return np.mean(self.amp[self.valid_window_indices], axis=0)
        elif distribution == "log-normal":
            return np.exp(np.mean(np.log(self.amp[self.valid_window_indices]), axis=0))
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def std_curve(self, distribution='log-normal'):
        """Return the standard deviation of mean hvsr curve.

        Args:
            distribution : {'normal', 'log-normal'}

        Return:
            std : ndarray
                Standard deviation of hvsr curve according to the
                distribution specified.
        """
        if self.n_windows == 1:
            raise ValueError(
                f"The standard deviation of the mean curve is not defined for a single window.")

        if distribution == "normal":
            return np.std(self.amp[self.valid_window_indices], axis=0, ddof=1)
        elif distribution == "log-normal":
            return np.std(np.log(self.amp[self.valid_window_indices]), axis=0, ddof=1)
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def mc_peak(self, distribution='log-normal'):
        mc = self.mean_curve(distribution)
        return self.frq[np.where(mc == np.max(mc[self.find_peaks(mc)[0]]))]

    def reject_windows(self, n=2, max_iterations=50, distribution='log-normal'):
        """Perform rejection of H/V windows using the method proposed by
        Chen et al. (2020).

        Args:
            n : float, optional
                Number of standard deviations from the mean (default
                value is 2).
            max_iterations : int, optional
                Maximum number of rejection iterations (default value is
                50).

        Returns:
            window_ids : np.array
                Index for each window.
        """
        if not self.initialized_peaks:
            self.update_peaks()

        if distribution == 'log-normal':
            def calulate_range(mean, std, n):
                upper = np.exp(np.log(mean)+n*std)
                lower = np.exp(np.log(mean)-n*std)
                return (lower, upper)
        elif distribution == 'normal':
            def calulate_range(mean, std, n):
                upper = mean+n*std
                lower = mean-n*std
                return (lower, upper)
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

        for c_iteration in range(max_iterations):

            logging.debug(f"c_iteration: {c_iteration}")
            logging.debug(f"valid_window_indices: {self.valid_window_indices}")

            mean_f0 = self.mean_f0(distribution)
            logging.debug(f"mean_f0: {mean_f0}")
            std_f0 = self.std_f0(distribution)
            logging.debug(f"std_f0: {std_f0}")
            mc_peak = self.mc_peak(distribution)
            logging.debug(f"mc_peaks: {mc_peak}")

            d_before = abs(mean_f0 - mc_peak)

            lower_bound, upper_bound = calulate_range(mean_f0, std_f0, n)
            rejected_windows = 0
            keep_indices = []
            for c_window, c_peak in zip(self.valid_window_indices, self.peak_frq):
                if c_peak < lower_bound or c_peak > upper_bound:
                    rejected_windows += 1
                else:
                    keep_indices.append(c_window)
            self.valid_window_indices = np.array(keep_indices)

            new_mean_f0 = self.mean_f0(distribution)
            new_mc_peak = self.mc_peak(distribution)
            new_std_f0 = self.std_f0(distribution)

            d_after = abs(new_mean_f0 - new_mc_peak)

            if std_f0 == 0 or new_std_f0 == 0 or d_before == 0:
                logging.info(f"Performed {c_iteration} iterations, returning b/c 0 values.")
                return c_iteration

            logging.debug(
                f"d relative difference: {(abs(d_after - d_before)/d_before)}")
            logging.debug(
                f"std relative difference: {(abs(std_f0 - new_std_f0)/std_f0)}")

            if ((abs(d_after - d_before)/d_before) < 0.01) and ((abs(std_f0 - new_std_f0)/std_f0) < 0.01):
                logging.info(f"Performed {c_iteration} iterations, returning b/c rejection converged.")
                return c_iteration
