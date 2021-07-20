# Additional Information

> Joseph P. Vantassel, The University of Texas at Austin

## I am interested in learning more about the horizontal-to-vertical spectral ratio method. What resources would you recommend?

An excellent starting reference for HVSR method is the SESAME Guidelines
(SESAME 2004). The guidelines thoroughly documents field acquisition,
processing, and interpretation of HVSR data. The citation is
below:

SESAME. (2004). Guidelines for the Implementation of the H/V Spectral Ratio
Technique on Ambient Vibrations Measurements, Processing, and Interpretation.
European Commission - Research General Directorate, 62.

## How do the time-domain, frequency-domain, and HVSR settings affect the result?

The effect of the processing settings will vary on a case-by-case basis. Some
simple examples showing how the processing settings affect a single time window
are shown in the
[README](https://github.com/jpvantassel/hvsrpy/blob/main/README.md).
For most sites the default/recommended settings (as detailed in the
Jupyter notebook) should be sufficient. Refer to the literature including
Cox et al. (2020) and the SESAME Guidelines (2004) specifically for more
information.

## What types of data formats can _hvsrpy_ accept?

Currently _hvsrpy_ is able to easily (i.e., no coding required) process
miniSEED files which conform to the _SEED_ standard. These can either be single
3-component files (i.e., all three components are in a single file) or three
1-component files. There are plans to extend this functionality to other common
data formats in the near future. If you would like to see support for a
particular format please read and comment on
[this open issue](https://github.com/jpvantassel/hvsrpy/issues/8).

## Do I need to pre-process my ambient noise recording before using _hvsrpy_?

In general, good quality ambient noise data (i.e., the sensor was allowed to
record undisturbed for the full record length) does not need to be
pre-processed, and can be used directly.
