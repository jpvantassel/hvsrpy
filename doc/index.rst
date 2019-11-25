.. hvsrpy documentation master file, created by sphinx-quickstart on
   Tue Nov 12 10:00:56 2019. You can adapt this file completely to your
   liking, but it should at least contain the root `toctree` directive.

hvsrpy Documentation
====================

Summary
-------
hvsrpy is a Python module for horizontal-to-vertical spectral ratio
processing. hvsrpy was developed by Joseph P. Vantassel with contributions
from Dana M. Brannon under the supervision of Professor Brady R. Cox at the
University of Texas at Austin. The fully-automated frequency-domain rejection
algorithm implemented in `hvsrpy` was developed by Tianjian Cheng under the
supervision of Professor Brady R. Cox at the University of Texas at Austin and
detailed in Cox et al. (in review).

The module includes two main class definitons `Sensor3c` and `Hvsr`. These
classes include various methods for creating and manipulating 3-component
sensor and horizontal-to-vertical spectral ratio objects, respectively.

License Information
-------------------

   Copyright (C) 2019 Joseph P. Vantassel (jvantassel@utexas.edu)

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https: //www.gnu.org/licenses/>.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Sensor3c Class
==============
.. automodule:: hvsrpy.Sensor3c
   :attributes:
   :members:
   :noindex:

.. autoclass:: hvsrpy.Sensor3c
   :members:

   .. automethod:: __init__

Hvsr Class
==========
.. automodule:: hvsrpy.Hvsr
   :attributes:
   :members:
   :noindex:

.. autoclass:: hvsrpy.Hvsr
   :members:

   .. automethod:: __init__

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
