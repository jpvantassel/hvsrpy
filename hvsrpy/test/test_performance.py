"""This file contains a speed test for the main hvsr calculation."""

import hvsrpy
import cProfile
import pstats

def main():
    windowlength = 60
    flow = 0.1
    fhigh = 45
    forder = 5
    width = 0.05
    bandwidth = 40
    fmin = 0.1
    fmax = 50
    fn = 256
    res_type = 'log'
    ratio_type = 'geometric-mean'
    n = 2
    sensor = hvsrpy.Sensor3c.from_mseed("test/data/a2/UT.STN11.A2_C50.miniseed")
    hv = sensor.hv(windowlength, flow, fhigh, forder, width, bandwidth, fmin, fmax, fn, res_type, ratio_type)
    hv.reject_windows(n, max_iterations=10)
    
fname = "test/.tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)
