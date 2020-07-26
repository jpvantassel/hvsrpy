# _hvsrpy_ - A Python package for horizontal-to-vertical spectral ratio processing

> Joseph P. Vantassel, The University of Texas at Austin

[![DOI](https://zenodo.org/badge/219310971.svg)](https://zenodo.org/badge/latestdoi/219310971)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/jpvantassel/hvsrpy/blob/master/LICENSE.txt)
[![CircleCI](https://circleci.com/gh/jpvantassel/hvsrpy.svg?style=svg)](https://circleci.com/gh/jpvantassel/hvsrpy)
[![Documentation Status](https://readthedocs.org/projects/hvsrpy/badge/?version=latest)](https://hvsrpy.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jpvantassel/hvsrpy.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jpvantassel/hvsrpy/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/528737ade629492e8652be369528c756)](https://www.codacy.com/manual/jpvantassel/hvsrpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jpvantassel/hvsrpy&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/jpvantassel/hvsrpy/branch/master/graph/badge.svg)](https://codecov.io/gh/jpvantassel/hvsrpy)

## Table of Contents

---

-   [About _hvsrpy_](#About-hvsrpy)
-   [Why use _hvsrpy_](#Why-use-hvsrpy)
-   [A Comparison of _hvsrpy_ with _Geopsy_](#A-comparison-of-hvsrpy-with-Geopsy)
-   [Getting Started](#Getting-Started)
-   [Additional Comparisons between _hvsrpy_ and _Geopsy_](#Additional-Comparisons-between-hvsrpy-and-Geopsy)
    -   [Multiple Windows](#Multiple-Windows)
    -   [Single Window](#Single-Window)

## About _hvsrpy_

---

`hvsrpy` is a Python package for performing horizontal-to-vertical spectral
ratio (H/V) processing. `hvsrpy` was developed by Joseph P. Vantassel with
contributions from Dana M. Brannon under the supervision of Professor Brady R.
Cox at The University of Texas at Austin. The automated frequency-domain
window-rejection algorithm and log-normal statistics implemented in `hvsrpy`
are detailed in Cox et al. (2020). The statistical approach to incorporate
azimuth variability implemented in `hvsrpy` are detailed in Cheng et al. (2020).

If you use `hvsrpy` in your research or consulting, we ask you please cite the
following:

> Joseph Vantassel. (2020). jpvantassel/hvsrpy: latest (Concept). Zenodo.
> [http://doi.org/10.5281/zenodo.3666956](http://doi.org/10.5281/zenodo.3666956)

_Note: For software, version specific citations should be preferred to
general concept citations, such as that listed above. To generate a version
specific citation for `hvsrpy`, please use the citation tool for that specific
version on the `hvsrpy` [archive](http://doi.org/10.5281/zenodo.3666956)._

These works provide background for the calculations performed by `hvsrpy`.

> Cox, B. R., Cheng, T., Vantassel, J. P., and Manuel, L. (2020). “A statistical
> representation and frequency-domain window-rejection algorithm for
> single-station HVSR measurements.” _Geophysical Journal International_, 221(3),
> 2170-2183.

> Cheng, T., Cox, B. R., Vantassel, J. P., and Manuel, L. (2020). "A
> statistical approach to account for azimuthal variability in single-station
> HVSR measurements." _Geophysical Journal International_, Accepted.

> Cheng, T., Hallal, M., Vantassel, J. P., and Cox, B. R. (2020). "Estimating
> Unbiased Statistics for Fundamental Site Frequency Using Spatially Distributed
> HVSR Measurements and Voronoi Tessellation." Submitted.

> SESAME. (2004). Guidelines for the Implementation of the H/V Spectral Ratio
> Technique on Ambient Vibrations Measurements, Processing, and Interpretation.
> European Commission - Research General Directorate, 62, European Commission -
> Research General Directorate.

`hvsrpy` would not exist without the help of many others. As a small display of
gratitude, we thank them individually
[here](https://github.com/jpvantassel/hvsrpy/blob/master/thanks.md).

## Why use _hvsrpy_

---

`hvsrpy` contains features not currently available in any other commercial or
open-source software, including:

-   A log-normal distribution for the fundamental site frequency (`f0`) so the
uncertainty in `f0` can be represented consistently in frequency or period.
-   Ability to use the geometric-mean, squared-average, or any azimuth of your
choice.
-   Access to the H/V data from each time window (and azimuth in the case of
azimuthal calculations), and not only the mean/median curve.
-   A method to calculate statistics on `f0` that incorporates azimuthal
variability.
-   A method for developing rigorous and unbiased spatial statistics.
-   A fully-automated frequency-domain window-rejection algorithm.
-   Automatic checking of the SESAME (2004) peak reliability and clarity
criteria.
-   A performant framework for batch-style processing.

### Example output from `hvsrpy` when considering the geometric-mean of the horizontal components

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/example_hvsr_figure.png?raw=true" width="775">

### Example output from `hvsrpy` when considering azimuthal variability

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/example_hvsr_figure_az.png?raw=true" width="775">

### Example output from `hvsrpy` when considering spatial variability

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/example_hvsr_figure_sp.png?raw=true" width="775">

## A comparison of _hvsrpy_ with _Geopsy_

---

Some of the functionality available in `hvsrpy` overlaps with the popular
open-source software `Geopsy`. And so to encourage standardization, wherever
their functionality coincides we have sought to ensure consistency. Two such
comparisons are shown below. One for a single time window (left) and one for
multiple time windows (right). Additional examples and the information
necessary to reproduce them are provided at the end of this document.

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_a.png?raw=true" width="425"> <img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN11_c050.png?raw=true" width="425">

## Getting Started

---

### Installing or Upgrading _hvsrpy_

1.  If you do not have Python 3.6 or later installed, you will need to do
so. A detailed set of instructions can be found
[here](https://jpvantassel.github.io/python3-course/#/intro/installing_python).

2.  If you have not installed `hvsrpy` previously use `pip install hvsrpy`.
If you are not familiar with `pip`, a useful tutorial can be found
[here](https://jpvantassel.github.io/python3-course/#/intro/pip). If you have
an earlier version and would like to upgrade to the latest version of `hvsrpy`
use `pip install hvsrpy --upgrade`.

3.  Confirm that `hvsrpy` has installed/updated successfully by examining the
last few lines of the text displayed in the console.

### Using _hvsrpy_

1.  Download the contents of the [examples](https://github.com/jpvantassel/hvsrpy/tree/master/examples)
  directory to any location of your choice.

2.  Launch the Jupyter notebook (`simple_hvsrpy_interface.ipynb`) in the examples
  directory for a no-coding-required introduction to the basics of the `hvsrpy`
  package. If you have not installed `Jupyter`, detailed instructions can be
  found [here](https://jpvantassel.github.io/python3-course/#/intro/installing_jupyter).

3.  Launch the Jupyter notebook (`azimuthal_hvsrpy_interface.ipynb`) in the
  examples directory to perform more rigorous calculations which incorporate
  azimuthal variability.

4.  Enjoy!

### Looking for more information

More information regarding HVSR processing and `hvsrpy` can be found
[here](https://github.com/jpvantassel/hvsrpy/blob/master/additional_information.md).

## Additional Comparisons between _hvsrpy_ and _Geopsy_

---

### Multiple Windows

The examples in this section use the same settings applied to different
noise records. The settings are provided in the __Settings__ section and the
name of each file is provided above the corresponding figure in the __Results__
section. The noise records (i.e., _.miniseed_ files) are provided in the
[examples](https://github.com/jpvantassel/hvsrpy/tree/master/examples) directory
and also as part of a large published data set
[(Cox and Vantassel, 2018)](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published//PRJ-2075/Thorndon%20Warf%20(A2)/Unprocessed%20Data/Microtremor%20Array%20Measurements%20(MAM)).

#### Settings

-   __Window Length:__ 60 seconds
-   __Bandpass Filter Boolean:__ False
-   __Cosine Taper Width:__ 10% (i.e., 5% in Geopsy)
-   __Konno and Ohmachi Smoothing Coefficient:__ 40
-   __Resampling:__
    -   __Minimum Frequency:__ 0.3 Hz
    -   __Maximum Frequency:__ 40 Hz
    -   __Number of Points:__ 2048
    -   __Sampling Type:__ 'log'
-   __Method for Combining Horizontal Components:__ 'squared-average'
-   __Distribution for f0 from Time Windows:__ 'normal'
-   __Distribution for Mean Curve:__ 'log-normal'

#### Multiple Window Results

__File Name:__ _UT.STN11.A2_C50.miniseed_

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN11_c050.png?raw=true" width="425">

__File Name:__ _UT.STN11.A2_C150.miniseed_

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN11_c150.png?raw=true" width="425">

__File Name:__ _UT.STN12.A2_C50.miniseed_

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN12_c050.png?raw=true" width="425">

__File Name:__ _UT.STN12.A2_C150.miniseed_

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN12_c150.png?raw=true" width="425">

### Single Window

The examples in this section apply different settings to the same noise
record (_UT.STN11.A2_C50.miniseed_). For brevity, the default settings are
listed in the __Default Settings__ section, with only the variations from
these settings noted for each example.

#### Default Settings

- __Window Length:__ 60 seconds
- __Bandpass Filter Boolean:__ False
- __Cosine Taper Width:__ 10% (i.e., 5% in Geopsy)
- __Konno and Ohmachi Smoothing Coefficient:__ 40
- __Resampling:__
    - __Minimum Frequency:__ 0.3 Hz
    - __Maximum Frequency:__ 40 Hz
    - __Number of Points:__ 2048
    - __Sampling Type:__ 'log'
- __Method for Combining Horizontal Components:__ 'squared-average'
- __Distribution for f0 from Time Windows:__ 'normal'
- __Distribution for Mean Curve:__ 'log-normal'

#### Single Window Results

__Default Case:__ No variation from those settings listed above.

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_a.png?raw=true" width="425">

__Window Length:__ 120 seconds.

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_b.png?raw=true" width="425">

__Cosine Taper Width:__ 20 % (i.e., 10 % in Geopsy)

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_e.png?raw=true" width="425">

__Cosine Taper Width:__ 0.2 % (i.e., 0.1 % in Geopsy)

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_f.png?raw=true" width="425">

__Konno and Ohmachi Smoothing Coefficient:__ 10

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_c.png?raw=true" width="425">

__Konno and Ohmachi Smoothing Coefficient:__ 80

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_d.png?raw=true" width="425">

__Number of Points:__ 512

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_g.png?raw=true" width="425">

__Number of Points:__ 4096

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_h.png?raw=true" width="425">
