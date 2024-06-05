Introduction to `hvsrpy`
========================

About `hvsrpy`
--------------

`hvsrpy` is an open-source Python package for performing
horizontal-to-vertical spectral ratio (HVSR) processing of microtremor
and earthquake recordings. `hvsrpy` was developed by
`Joseph P. Vantassel <https://www.jpvantassel.com/>`_ with
contributions from Dana M. Brannon under the supervision of Professor
Brady R. Cox at The University of Texas at Austin. `hvsrpy` continues to
be developed and maintained by `Joseph P. Vantassel and his research
group at Virginia Tech <https://geoimaging-research.org/>`_.

Citation
--------

If you use `hvsrpy` in your research or consulting, we ask you please
cite the following:

    Joseph Vantassel. (2020). jpvantassel/hvsrpy: latest (Concept).
    Zenodo. http://doi.org/10.5281/zenodo.3666956

.. TODO(JPV): Add citatation to hvsrpy paper when published.

.. note:: 
   For software, version specific citations should be preferred to
   general concept citations, such as that listed above. To generate a
   version specific citation for `hvsrpy`, please use the citation tool
   on the `hvsrpy` `archive <http://doi.org/10.5281/zenodo.3666956>`_.


.. hvsrpy would not exist without the help of many others. As a small
.. display of gratitude, we thank them individually here.

.. Why use hvsrpy

.. hvsrpy contains features not currently available in any other commercial or open-source software, including:

..     A lognormal distribution for the fundamental site frequency (f0) so the uncertainty in f0 can be represented consistently in frequency or period.
..     Ability to use the geometric-mean, squared-average, or any azimuth of your choice.
..     Easy access to the HVSR data from each time window (and azimuth in the case of azimuthal calculations), not only the mean/median curve.
..     A method to calculate statistics on f0 that incorporates azimuthal variability.
..     A method for developing rigorous and unbiased spatial statistics.
..     A fully-automated frequency-domain window-rejection algorithm.
..     Automatic checking of the SESAME (2004) peak reliability and clarity criteria.
..     A command line interface for highly performant batch-style processing.

.. Example output from hvsrpy when considering the geometric-mean of the horizontal components

.. 	Lognormal Median 	Lognormal Standard Deviation
.. Fundamental Site Frequency, f0,GM 	0.72 	0.11
.. Fundamental Site Period, T0,GM 	1.40 	0.11
.. Example output from hvsrpy when considering azimuthal variability

.. 	Lognormal Median 	Lognormal Standard Deviation
.. Fundamental Site Frequency, f0,AZ 	0.68 	0.18
.. Fundamental Site Period, T0,AZ 	1.48 	0.18
.. Example output from hvsrpy when considering spatial variability

.. 	Lognormal Median 	Lognormal Standard Deviation
.. Fundamental Site Frequency, f0,XY 	0.58 	0.15
.. Fundamental Site Period, T0,XY 	1.74 	0.15

References
----------

The references below provide background on the calculations performed
by `hvsrpy`. We ask you please cite them appropriately.

    Cox, B. R., Cheng, T., Vantassel, J. P., and Manuel, L. (2020).
    "A statistical representation and frequency-domain window-rejection
    algorithm for single-station HVSR measurements. Geophysical Journal
    International, 221(3), 2170-2183. https://doi.org/10.1093/gji/ggaa119

    Cheng, T., Cox, B. R., Vantassel, J. P., and Manuel, L. (2020). "A
    statistical approach to account for azimuthal variability in
    single-station HVSR measurements." Geophysical Journal International,
    223(2), 1040-1053. https://doi.org/10.1093/gji/ggaa342

    Cheng, T., Hallal, M. M., Vantassel, J. P., and Cox, B. R., (2021).
    "Estimating Unbiased Statistics for Fundamental Site Frequency Using
    Spatially Distributed HVSR Measurements and Voronoi Tessellation. J.
    Geotech. Geoenviron. Eng. 147, 04021068.
    https://doi.org/10.1061/(ASCE)GT.1943-5606.0002551

    SESAME. (2004). Guidelines for the Implementation of the H/V
    Spectral Ratio Technique on Ambient Vibrations Measurements,
    Processing, and Interpretation. European Commission - Research
    General Directorate, 62, European Commission - Research General
    Directorate.

    Welch, P., (1967). The use of fast Fourier transform for the
    estimation of power spectra: a method based on time averaging over
    short, modified periodograms. IEEE Transactions on audio and
    electroacoustics, 15(2), pp.70-73.

.. toctree::
   :maxdepth: 1
   :hidden:

   license
   install
   api
   cli
