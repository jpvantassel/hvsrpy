"""File for Horizontal-to-Vertical Spectral Ratio (Hvsr) class."""

import numpy as np
import scipy.signal as sg
import logging
logging.getLogger()


class Hvsr():
    """Class for creating and manipulating horizontal-to-vertical
    spectral ratio (H/V) objects.

    Attributes:
        amp : ndarray
            Array of H/V amplitudes. Each row represents an individual
            curve and each column a frequency.
        frq : ndarray
            Vector of frequencies, corresponding to each column.
        n_windows : int
            Number of windows in Hvsr object.
        valid_window_indices : ndarray
            Array of indices indicating valid windows.

    """
    @staticmethod
    def _check_input(name, value):
        """Basic check on input values.
        
        Specifically;
            1. `value` must be of type `list`, `tuple`, `ndarray`.
            2. If `value` is not `ndarray`, convert to `ndarray`.
            3. `value` must be >=0. 

        Args:
            name : str
                Name of `value` to be checked, used solely for
                meaningful error messages.
            value : any
                Value to be checked.

        Returns:
            `value` which may have been cast to a `ndarray`.

        Raises:
            TypeError:
                If `value` is not one of those specified.
            ValueError:
                If `value` has negative values.
        """

        if type(value) not in [list, tuple, np.ndarray]:
            msg = f"{name} must be of type ndarray, not {type(value)}."
            raise TypeError(msg)
        if type(value) in [list, tuple]:
            value = np.array(value)
        if np.sum(value<0):
            print(value)
            raise ValueError(f"{name} must be >= 0.")
        return value

    def __init__(self, amplitude, frequency, find_peaks=True):
        """Initialize a Hvsr oject from amplitude and frequency vector.

        Args:
            amplitude : ndarray
                Array of H/V amplitudes. Each row represents an individual
                curve and each column a frequency.
            frequency : ndarray
                Vector of frequencies, corresponding to each column.

        Returns:
            Initialized Hvsr object.
        """
        self.amp = self._check_input("amplitude", amplitude)
        self.frq = self._check_input("frequency", frequency)
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
                Vector or array of amplitudes. See `amp` attribute for 
                details.
            **kwargs : dict
                Refer to `scipy.signal.find_peaks` documentation:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        Returns:
            peaks : ndarray or list
                `ndarray` or `list` of `ndarrays` (one per window) of
                peak indices.

            properties : dict
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
        """Update `peaks` attribute with the lowest frequency, highest
        amplitude peak.
        
        Args:
            **kwargs:
                Refer to `find_peaks` documentation.
            
        Returns:
            `None`, update `peaks` attribute.
        """
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
        """Return mean value of `f0` of valid timewindows.

        Args:
            distribution : {'normal', 'log-normal'}
                Assumed distribution of `f0`, default is `log-normal`.

        Returns:
            Mean value of `f0` according to the distribution specified.

        Raises:
            KeyError:
                If `distribution` does not match the available options.
        """
        if distribution == "normal":
            return np.mean(self.peak_frq)
        elif distribution == "log-normal":
            return np.exp(np.mean(np.log(self.peak_frq)))
        else:
            msg = f"distribution type {distribution} not recognized."
            raise KeyError(msg)

    def std_f0(self, distribution='log-normal'):
        """Return sample standard deviation of `f0` of valid timewindows.

        Args:
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of `f0`, default is `log-normal`.

        Returns:
            std : float
                Sample standard deviation value according to the
                distribution specified.
        
        Raises:
            KeyError:
                If `distribution` does not match the available options.
        """
        if distribution == "normal":
            return np.std(self.peak_frq, ddof=1)
        elif distribution == "log-normal":
            return np.std(np.log(self.peak_frq), ddof=1)
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def mean_curve(self, distribution='log-normal'):
        """Return mean H/V curve.

        Args:
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of mean curve, default is 
                `log-normal`.

        Returns:
            Mean H/V curve as `ndarray` according to the distribution
            specified.
        
        Raises:
            KeyError:
                If `distribution` does not match the available options.
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
        """Sample standard deviation associate with the mean H/V curve.

        Args:
            distribution : {'normal', 'log-normal'}, optional
                Assumed distribution of H/V curve, default is
                `log-normal`.

        Returns:
            Sample standard deviation of H/V curve as `ndarray`
            according to the distribution specified.

        Raises:
            ValueError:
                If only single time window is defined.
            KeyError:
                If `distribution` does not match the available options.
        """
        if self.n_windows == 1:
            msg = "The standard deviation of the mean curve is not defined for a single window."
            raise ValueError(msg)

        if distribution == "normal":
            return np.std(self.amp[self.valid_window_indices], axis=0, ddof=1)
        elif distribution == "log-normal":
            return np.std(np.log(self.amp[self.valid_window_indices]), axis=0, ddof=1)
        else:
            raise KeyError(f"distribution type {distribution} not recognized.")

    def mc_peak(self, distribution='log-normal'):
        """Peak of mean H/V curve.

        Args:
            distribution : {'normal', 'log-normal'}, optional
                Refer to method `mean_curve` for details.
        
        Returns:
            Frequency associated with the peak of the mean H/V curve.
        """
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
