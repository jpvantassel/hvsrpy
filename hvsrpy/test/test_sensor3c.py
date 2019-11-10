"""Tests for Sensor3c."""

import unittest
import hvsrpy as hv
import logging
logging.basicConfig(level=logging.DEBUG)


class Test(unittest.TestCase):
    def test_init(self):
        # Successful init
        ns = hv.TimeSeries([1, 1, 1], dt=1)
        ew = hv.TimeSeries([1, 1, 1], dt=1)
        vt = hv.TimeSeries([1, 1, 1], dt=1)
        hv.Sensor3c(ns, ew, vt)

        # Bad ns, should be TimeSeries
        ns = [1, 1, 1]
        self.assertRaises(TypeError, hv.Sensor3c, ns, ew, vt)

        # Bad dt, should be 1
        ns = hv.TimeSeries([1, 1, 1], dt=2)
        self.assertRaises(ValueError, hv.Sensor3c, ns, ew, vt)

        # Bad length, will trim
        ns = hv.TimeSeries([1, 1], dt=1)
        hv.Sensor3c(ns, ew, vt)
        self.assertEqual(ew.amp.tolist(), [1, 1])
        self.assertEqual(vt.amp.tolist(), [1, 1])

    


if __name__ == "__main__":
    unittest.main()
