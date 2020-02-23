# _hvsrpy_ - A Python package for horizontal-to-vertical spectral ratio processing

> Joseph P. Vantassel, The University of Texas at Austin

[![DOI](https://zenodo.org/badge/219310971.svg)](https://zenodo.org/badge/latestdoi/219310971)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/jpvantassel/hvsrpy/blob/master/LICENSE.txt)
[![CircleCI](https://circleci.com/gh/jpvantassel/hvsrpy.svg?style=svg)](https://circleci.com/gh/jpvantassel/hvsrpy)
[![Documentation Status](https://readthedocs.org/projects/hvsrpy/badge/?version=latest)](https://hvsrpy.readthedocs.io/en/latest/?badge=latest)

## Table of Contents

---

- [About _hvsrpy_](#About-hvsrpy)
- [Why use _hvsrpy_](#Why-use-hvsrpy)
- [A Comparison of _hvsrpy_ with _Geopsy_](#A-comparison-of-hvsrpy-with-Geopsy)
- [Getting Started](#Getting-Started)
- [Additional Comparisons between _hvsrpy_ and _Geopsy_](#Additional-Comparisons-between-hvsrpy-and-Geopsy)
  - [Multiple Windows](#Multiple-Windows)
  - [Single Window](#Single-Window)

## About _hvsrpy_

---

`hvsrpy` is a Python package for performing horizontal-to-vertical spectral
ratio (H/V) processing. `hvsrpy` was developed by Joseph P. Vantassel with
contributions from Dana M. Brannon under the supervision of Professor Brady R.
Cox at The University of Texas at Austin. The automated frequency-domain
window-rejection algorithm and log-normal statistics implemented in `hvsrpy`
were developed by Tianjian Cheng under the supervision of Professor Brady R. Cox
at The University of Texas at Austin and detailed in Cox et al. (2020), citation
below.

If you use _hvsrpy_ in your research, we ask you please cite the following:

>Cox, B. R., Cheng, T., Vantassel, J. P., and Manuel, L. (2020). “A Statistical
Representation and Frequency-Domain Window-Rejection Algorithm for
Single-Station HVSR Measurements.” _Geophysical Journal International._
In review.

>Joseph Vantassel. (2020). jpvantassel/hvsrpy: latest (Concept). Zenodo.
[http://doi.org/10.5281/zenodo.3666956](http://doi.org/10.5281/zenodo.3666956)

_Note: For software, version specific citations should be preferred to
general concept citations, such as that listed above. To generate a version
specific citation for `hvsrpy`, please use the citation tool for that specific
version on the `hvsrpy` [archive](http://doi.org/10.5281/zenodo.3666956)._

## Why use _hvsrpy_

---

`hvsrpy` contains features not currently available in any other commercial or
open-source software, including:

- A fully-automated frequency-domain window-rejection algorithm, which allows
spurious time windows to be removed in a repeatable and expedient manner.
- A log-normal distribution for the fundamental site frequency (`f0`) so the
uncertainty in `f0` can be represented consistently regardless of whether it is
described in terms of frequency or period.
- Combining the two horizontal components using the geometric mean.
- Access to the H/V data from each time window, not only the
mean/median curve.
- A performant framework for batch-style processing.

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/example_hvsr_figure.png?raw=true" width="775">

## A comparison of _hvsrpy_ with _Geopsy_

---

To illustrate that `hvsrpy` can exactly reproduce the results from the popular
open-source software `Geopsy` two comparisons are shown below. One for a single
time window (left) and one for multiple time windows (right). Additional
examples and the information necessary to reproduce them are provided at the end
of this document.

<img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/singlewindow_a.png?raw=true" width="425"> <img src="https://github.com/jpvantassel/hvsrpy/blob/master/figs/multiwindow_STN11_c050.png?raw=true" width="425">

## Getting Started

---

### Installing _hvsrpy_

1. If you do not have Python 3.6 or later installed, you will need to do
so. A detailed set of instructions can be found [here](https://github.com/jpvantassel/python3-course/blob/master/0_Getting_Started/installing_python.md).

2. `pip install hvsrpy`. If you are not familiar with `pip`, a useful tutorial
can be found [here](https://github.com/jpvantassel/python3-course/blob/master/1_Installing_Packages/pip.md).

3. Confirm that `hvsrpy` has installed successfully by examining the last few
lines of the text displayed in the console.

### Using _hvsrpy_

1. Download the contents of the [examples](https://github.com/jpvantassel/hvsrpy/tree/master/examples)
  directory to any location of your choice.

2. Launch the Jupyter notebook (`file with .ipynb extension`) in the examples
  directory for a no-coding-required introduction to the `hvsrpy` package. If
  you have not installed `Jupyter`, detailed instructions can be found
  [here](https://github.com/jpvantassel/python3-course/blob/master/0_Getting_Started/installing_jupyter.md).

3. Enjoy!

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
