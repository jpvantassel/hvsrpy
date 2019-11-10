"""Tests for Hvsr object."""

import unittest
import numpy as np
import hvsrpy as hv
import logging
logging.basicConfig(level=logging.DEBUG)


class Test(unittest.TestCase):
    def test_init(self):
        # amp as 1d array
        frq = np.linspace(1, 10, 20)
        amp = np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual(frq.tolist(), myhvsr.frq.tolist())
        self.assertListEqual(amp.tolist(), myhvsr.amp.tolist())

        # # amp as 2d array
        frq = np.linspace(1, 10, 20)
        amp = (np.sin(2*np.pi*5*np.linspace(0, 10, 20))+10)*np.ones((20, 20))
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual(frq.tolist(), myhvsr.frq.tolist())
        self.assertListEqual(amp.tolist(), myhvsr.amp.tolist())

    def test_find_peaks(self):
        # amp as 1d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 0, 1, 0, 0])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual([2], hv.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - single peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual([[2], [1], [2]], hv.Hvsr.find_peaks(myhvsr.amp)[0])

        # amp as 1d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0, 1, 0, 1, 0])
        myhvsr = hv.Hvsr(amp, frq)
        self.assertListEqual([1, 3], hv.Hvsr.find_peaks(myhvsr.amp)[0].tolist())

        # amp as 2d array - multiple peak
        frq = np.array([1, 2, 3, 4, 5, 6, 7])
        amp = np.array([[0, 1, 0, 1, 0, 5, 0],
                        [0, 2, 6, 5, 0, 0, 0],
                        [0, 0, 7, 6, 8, 0, 0]])
        myhvsr = hv.Hvsr(amp, frq)
        for known, test in zip([[1, 3, 5], [2], [2, 4]], hv.Hvsr.find_peaks(myhvsr.amp)[0]):
            self.assertListEqual(known, test.tolist())

    def test_update_peaks(self):
        frq = np.arange(0, 1, 0.1)
        amp = np.zeros((10, 10))
        col = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 9])
        amp[np.arange(10), col] = 1
        myhv = hv.Hvsr(amp, frq)
        self.assertListEqual(myhv.peak_frq.tolist(), frq[col[:-1]].tolist())

    def test_mean_std_f0(self):
        frq = np.arange(0, 10, 1)
        amp = np.zeros((10, 10))
        col = np.array([1, 2, 4, 6, 8, 1, 3, 5, 7, 6])
        amp[np.arange(10), col] = 1
        myhv = hv.Hvsr(amp, frq)
        self.assertEqual(myhv.mean_f0(distribution='log-normal'),
                         np.exp(np.mean(np.log(col))))
        self.assertEqual(myhv.mean_f0(distribution='normal'), np.mean(col))
        self.assertEqual(myhv.std_f0(distribution='log-normal'),
                         np.std(np.log(col), ddof=1))
        self.assertEqual(myhv.std_f0(distribution='normal'),
                         np.std(col, ddof=1))

    def test_mean_std_curve(self):
        frq = np.array([0, 1])
        amp = np.array([[1, 1],
                        [3, 4],
                        [5, 7]])
        myhv = hv.Hvsr(amp, frq, find_peaks=False)

        # Log-normal
        mean_curve = myhv.mean_curve(distribution='log-normal')
        std_curve = myhv.std_curve(distribution='log-normal')
        for col in range(amp.shape[1]):
            self.assertEqual(
                np.exp(np.mean(np.log(amp[:, col]))), mean_curve[col])
            self.assertEqual(
                np.std(np.log(amp[:, col]), ddof=1), std_curve[col])

        # Normal
        mean_curve = myhv.mean_curve(distribution='normal')
        std_curve = myhv.std_curve(distribution='normal')
        for col in range(amp.shape[1]):
            self.assertEqual(np.mean(amp[:, col]), mean_curve[col])
            self.assertEqual(np.std(amp[:, col], ddof=1), std_curve[col])

    def test_mc_peak(self):
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        self.assertEqual(1., myhv.mc_peak())

    def test_reject_windows(self):
        # Reject single window, end due to zero stdev
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 7])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        myhv.reject_windows(n=2)
        self.assertEqual(myhv.mean_f0(), 1.0)

        # Reject single window, end due to convergence criteria
        frq = np.arange(0, 10, 1)
        amp = np.ones((10, 10))
        col = np.array([1, 2, 2, 2, 2, 1, 2, 2, 1, 9])
        amp[np.arange(10), col] = 2
        myhv = hv.Hvsr(amp, frq)
        myhv.reject_windows(n=2)
        self.assertListEqual(myhv.peak_frq.tolist(), frq[col[:-1]].tolist())

if __name__ == "__main__":
    unittest.main()
