    # def print_stats(self, distribution_f0, places=2):  # pragma: no cover
    #     """Print basic statistics of `Hvsr` instance."""
    #     display(self._stats(distribution_f0=distribution_f0).round(places))

    def _geopsy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for Geopsy-style file."""
        # f0 from windows
        mean = self.mean_f0_frq(distribution_f0)
        lower = self.nstd_f0_frq(-1, distribution_f0)
        upper = self.nstd_f0_frq(+1, distribution_f0)

        # mean curve
        mc = self.mean_curve(distribution_mc)
        mc_peak_frq = self.mc_peak_frq(distribution_mc)
        mc_peak_amp = self.mc_peak_amp(distribution_mc)
        _min = self.nstd_curve(-1, distribution_mc)
        _max = self.nstd_curve(+1, distribution_mc)

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        lines = [
            f"# hvsrpy output version {__version__}",
            f"# Number of windows = {sum(self.valid_window_boolean_mask)}",
            f"# f0 from average\t{fclean(mc_peak_frq)}",
            f"# Number of windows for f0 = {sum(self.valid_window_boolean_mask)}",
            f"# f0 from windows\t{fclean(mean)}\t{fclean(lower)}\t{fclean(upper)}",
            f"# Peak amplitude\t{fclean(mc_peak_amp)}",
            f"# Position\t{0} {0} {0}",
            "# Category\tDefault",
            "# Frequency\tAverage\tMin\tMax",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, a_i, n_i, x_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i}\t{a_i}\t{n_i}\t{x_i}\n")

        return _lines

    def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for hvsrpy-style file."""
        # Correct distribution
        distribution_f0 = self.correct_distribution(distribution_f0)
        distribution_mc = self.correct_distribution(distribution_mc)

        # f0 from windows
        mean_f = self.mean_f0_frq(distribution_f0)
        sigm_f = self.std_f0_frq(distribution_f0)
        ci_68_lower_f = self.nstd_f0_frq(-1, distribution_f0)
        ci_68_upper_f = self.nstd_f0_frq(+1, distribution_f0)

        # mean curve
        mc = self.mean_curve(distribution_mc)
        mc_peak_frq = self.mc_peak_frq(distribution_mc)
        mc_peak_amp = self.mc_peak_amp(distribution_mc)
        _min = self.nstd_curve(-1, distribution_mc)
        _max = self.nstd_curve(+1, distribution_mc)

        n_rejected = self.nseries - sum(self.valid_window_boolean_mask)
        rejection = "False" if self.meta.get(
            'Performed Rejection') is None else "True"
        lines = [
            f"# hvsrpy output version {__version__}",
            f"# File Name (),{self.meta.get('File Name')}",
            f"# Method (),{self.meta.get('method')}",
            f"# Azimuth (),{self.meta.get('azimuth')}",
            f"# Window Length (s),{self.meta.get('Window Length')}",
            f"# Total Number of Windows (),{self.nseries}",
            f"# Frequency Domain Window Rejection Performed (),{rejection}",
            f"# Lower frequency limit for peaks (Hz),{self.f_low}",
            f"# Upper frequency limit for peaks (Hz),{self.f_high}",
            f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
            f"# Number of Accepted Windows (),{self.nseries-n_rejected}",
            f"# Number of Rejected Windows (),{n_rejected}",
            f"# Distribution of f0 (),{distribution_f0}"]

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        distribution_f0 = self.correct_distribution(distribution_f0)
        distribution_mc = self.correct_distribution(distribution_mc)

        if distribution_f0 == "lognormal":
            mean_t = 1/mean_f
            sigm_t = sigm_f
            ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
            ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

            lines += [
                f"# Median f0 (Hz) [LMf0],{fclean(mean_f)}",
                f"# Lognormal standard deviation f0 () [SigmaLNf0],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Median T0 (s) [LMT0],{fclean(mean_t)}",
                f"# Lognormal standard deviation T0 () [SigmaLNT0],{fclean(sigm_t)}",
                f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
            ]

        else:
            lines += [
                f"# Mean f0 (Hz),{fclean(mean_f)}",
                f"# Standard deviation f0 (Hz) [Sigmaf0],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                "# Mean T0 (s) [LMT0],NA",
                "# Standard deviation T0 () [SigmaT0],NA",
                "# 68 % Confidence Interval T0 (s),NA",
            ]

        c_type = "Median" if distribution_mc == "lognormal" else "Mean"
        lines += [
            f"# {c_type} Curve Distribution (),{distribution_mc}",
            f"# {c_type} Curve Peak Frequency (Hz) [f0mc],{fclean(mc_peak_frq)}",
            f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
            f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

        return _lines

    def to_file(self, fname, distribution_f0, distribution_mc, data_format="hvsrpy"):
        """Save HVSR data to summary file.

        Parameters
        ----------
        fname : str
            Name of file to save the results, may be a full or
            relative path.
        distribution_f0 : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0` from the time windows, the
            default is 'lognormal'.
        distribution_mc : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'lognormal'.
        data_format : {'hvsrpy', 'geopsy'}, optional
            Format of output data file, default is 'hvsrpy'.

        Returns
        -------
        None
            Writes file to disk.

        """
        if data_format == "geopsy":
            lines = self._geopsy_style_lines(distribution_f0, distribution_mc)
        elif data_format == "hvsrpy":
            lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)
        else:
            raise NotImplementedError(f"data_format={data_format} is unknown.")

        with open(fname, "w") as f:
            for line in lines:
                f.write(line)


    def _stats(self, distribution_f0):
        distribution_f0 = self.correct_distribution(distribution_f0)

        if distribution_f0 == "lognormal":
            columns = ["Lognormal Median", "Lognormal Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [1/self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)]])

        elif distribution_f0 == "normal":
            columns = ["Mean", "Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [np.nan, np.nan]])
        else:
            msg = f"`distribution_f0` of {distribution_f0} is not implemented."
            raise NotImplementedError(msg)

        df = DataFrame(data=data, columns=columns,
                       index=["Fundamental Site Frequency, f0",
                              "Fundamental Site Period, T0"])
        return df


### HvsrAzimuthal

    def _stats(self, distribution_f0):
        distribution_f0 = Hvsr.correct_distribution(distribution_f0)

        if distribution_f0 == "lognormal":
            columns = ["Lognormal Median", "Lognormal Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [1/self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)]])

        elif distribution_f0 == "normal":
            columns = ["Means", "Standard Deviation"]
            data = np.array([[self.mean_f0_frq(distribution_f0),
                              self.std_f0_frq(distribution_f0)],
                             [np.nan, np.nan]])
        else:
            msg = f"`distribution_f0` of {distribution_f0} is not implemented."
            raise NotImplementedError(msg)

        df = DataFrame(data=data, columns=columns,
                       index=["Fundamental Site Frequency, f0,AZ",
                              "Fundamental Site Period, T0,AZ"])
        return df

    # def print_stats(self, distribution_f0, places=2):  # pragma: no cover
    #     """Print basic statistics of `Hvsr` instance."""
    #     display(self._stats(distribution_f0=distribution_f0).round(places))

    def _hvsrpy_style_lines(self, distribution_f0, distribution_mc):
        """Lines for hvsrpy-style file."""
        # Correct distribution
        distribution_f0 = Hvsr.correct_distribution(distribution_f0)
        distribution_mc = Hvsr.correct_distribution(distribution_mc)

        # `f0` from windows
        mean_f = self.mean_f0_frq(distribution_f0)
        sigm_f = self.std_f0_frq(distribution_f0)
        ci_68_lower_f = self.nstd_f0_frq(-1, distribution_f0)
        ci_68_upper_f = self.nstd_f0_frq(+1, distribution_f0)

        # mean curve
        mc = self.mean_curve(distribution_mc)
        mc_peak_frq = self.mc_peak_frq(distribution_mc)
        mc_peak_amp = self.mc_peak_amp(distribution_mc)
        _min = self.nstd_curve(-1, distribution_mc)
        _max = self.nstd_curve(+1, distribution_mc)

        rejection = "False" if self.meta.get('Performed Rejection') is None else "True"

        nseries = self.hvsrs[0].nseries
        n_accepted = sum([sum(hvsr.valid_window_indices) for hvsr in self.hvsrs])
        n_rejected = self.azimuth_count*nseries - n_accepted
        lines = [
            f"# hvsrpy output version {__version__}",
            f"# File Name (),{self.meta.get('File Name')}",
            f"# Window Length (s),{self.meta.get('Window Length')}",
            f"# Total Number of Windows per Azimuth (),{nseries}",
            f"# Total Number of Azimuths (),{self.azimuth_count}",
            f"# Total Number of Windows (),{nseries*self.azimuth_count}",
            f"# Frequency Domain Window Rejection Performed (),{rejection}",
            f"# Lower frequency limit for peaks (Hz),{self.hvsrs[0].f_low}",
            f"# Upper frequency limit for peaks (Hz),{self.hvsrs[0].f_high}",
            f"# Number of Standard Deviations Used for Rejection () [n],{self.meta.get('n')}",
            f"# Number of Accepted Windows (),{n_accepted}",
            f"# Number of Rejected Windows (),{n_rejected}",
            f"# Distribution of f0 (),{distribution_f0}"]

        def fclean(number, decimals=4):
            return np.round(number, decimals=decimals)

        if distribution_f0 == "lognormal":
            mean_t = 1/mean_f
            sigm_t = sigm_f
            ci_68_lower_t = np.exp(np.log(mean_t) - sigm_t)
            ci_68_upper_t = np.exp(np.log(mean_t) + sigm_t)

            lines += [
                f"# Median f0 (Hz) [LMf0AZ],{fclean(mean_f)}",
                f"# Lognormal standard deviation f0 () [SigmaLNf0AZ],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                f"# Median T0 (s) [LMT0AZ],{fclean(mean_t)}",
                f"# Lognormal standard deviation T0 () [SigmaLNT0AZ],{fclean(sigm_t)}",
                f"# 68 % Confidence Interval T0 (s),{fclean(ci_68_lower_t)},to,{fclean(ci_68_upper_t)}",
            ]

        else:
            lines += [
                f"# Mean f0 (Hz) [f0AZ],{fclean(mean_f)}",
                f"# Standard deviation f0 (Hz) [Sigmaf0AZ],{fclean(sigm_f)}",
                f"# 68 % Confidence Interval f0 (Hz),{fclean(ci_68_lower_f)},to,{fclean(ci_68_upper_f)}",
                "# Mean T0 (s) [LMT0AZ],NAN",
                "# Standard deviation T0 () [SigmaT0AZ],NAN",
                "# 68 % Confidence Interval T0 (s),NAN",
            ]

        c_type = "Median" if distribution_mc == "lognormal" else "Mean"
        lines += [
            f"# {c_type} Curve Distribution (),{distribution_mc}",
            f"# {c_type} Curve Peak Frequency (Hz) [f0mcAZ],{fclean(mc_peak_frq)}",
            f"# {c_type} Curve Peak Amplitude (),{fclean(mc_peak_amp)}",
            f"# Frequency (Hz),{c_type} Curve,1 STD Below {c_type} Curve,1 STD Above {c_type} Curve",
        ]

        _lines = []
        for line in lines:
            _lines.append(line+"\n")

        for f_i, mean_i, bel_i, abv_i in zip(fclean(self.frq), fclean(mc), fclean(_min), fclean(_max)):
            _lines.append(f"{f_i},{mean_i},{bel_i},{abv_i}\n")

        return _lines

    def to_file(self, fname, distribution_f0, distribution_mc, data_format="hvsrpy"):
        """Save HVSR data to file.

        Parameters
        ----------
        fname : str
            Name of file to save the results, may be the full or a
            relative path.
        distribution_f0 : {'lognormal', 'normal'}, optional
            Assumed distribution of `f0` from the time windows, the
            default is 'lognormal'.
        distribution_mc : {'lognormal', 'normal'}, optional
            Assumed distribution of mean curve, the default is
            'lognormal'.
        data_format : {'hvsrpy'}, optional
            Format of output data file, default is 'hvsrpy'.

        Returns
        -------
        None
            Writes file to disk.

        """
        if data_format not in ["hvsrpy"]:
            raise ValueError(f"data_format {data_format} unknown.")

        lines = self._hvsrpy_style_lines(distribution_f0, distribution_mc)

        with open(fname, "w") as f:
            for line in lines:
                f.write(line)
