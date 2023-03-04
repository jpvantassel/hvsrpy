        #     def test_stat_factories(self):
        #         distribution = "exponential"
        #         self.assertRaises(NotImplementedError, self.hv._mean_factory,
        #                           distribution, np.array([1, 2, 3, 4]))
        #         self.assertRaises(NotImplementedError, self.hv._std_factory,
        #                           distribution, np.array([1, 2, 3, 4]))
        #         self.assertRaises(NotImplementedError, self.hv._nth_std_factory,
        #                           1, distribution, 0, 0)


#     @unittest.skip("Ignore for now")
#     def test_io(self):
#         fname = self.full_path + "data/a2/UT.STN11.A2_C150.miniseed"
#         windowlength = 60
#         bp_filter = {"flag": False, "flow": 0.1, "maxf": 30, "order": 5}
#         width = 0.1
#         bandwidth = 40
#         resampling = {"minf": 0.2, "maxf": 20, "nf": 128, "res_type": "log"}
#         method = "multiple-azimuths"
#         azimuthal_interval = 15
#         azimuth = np.arange(0, 180+azimuthal_interval, azimuthal_interval)
#         sensor = hvsrpy.Sensor3c.from_mseed(fname)
#         sensor.meta["File Name"] = "UT.STN11.A2_C150.miniseed"
#         hv = sensor.hv(windowlength, bp_filter, width,
#                        bandwidth, resampling, method, azimuth=azimuth)
#         distribution_f0 = "lognormal"
#         distribution_mc = "lognormal"

#         n = 2
#         n_iteration = 50
#         hv.reject_windows(n=n, max_iterations=n_iteration,
#                           distribution_f0=distribution_f0,
#                           distribution_mc=distribution_mc)

#         # Post-rejection
#         df = hv._stats(distribution_f0)
#         returned = np.round(df.to_numpy(), 2)
#         expected = np.array([[0.67, 0.18], [1.50, 0.18]])
#         self.assertArrayEqual(expected, returned)

#         # data_format == "hvsrpy"
#         returned = hv._hvsrpy_style_lines(distribution_f0, distribution_mc)
#         with open(self.full_path+"data/output/example_output_hvsrpy_az.hv") as f:
#             expected = f.readlines()
#         self.assertListEqual(expected, returned)
