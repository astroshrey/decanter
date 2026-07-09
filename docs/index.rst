decanter
========

A fast, pure-Python reduction pipeline for **WINERED** near-infrared echelle
spectra. ``decanter`` reproduces WARP — the reference IRAF-based WINERED
pipeline — to float32 precision across all three instrument modes (HIRES-Y,
HIRES-J, WIDE), while running much faster and parallelizing trivially over
frames.

Install
-------

.. code-block:: bash

   pip install -e .        # requires numpy, scipy, astropy

Quickstart
----------

Load a calibration set, then reduce your ``(object, sky)`` frame pairs:

.. code-block:: python

   import decanter

   # Point at any calibration-set directory, or a WARP reduction root.
   calib = decanter.Calibration.from_dir("/path/to/calibration_set")

   r = decanter.reduce("WINA00045837.fits", "WINA00045838.fits", calib)

   spec = r.obj[(1.30, 163)]        # OrderSpectrum, keyed by (FSR cut, order)
   spec.wavelength, spec.flux       # vacuum-Å grid + flux (counts)

A whole transit is just a loop — each reduction is independent:

.. code-block:: python

   calib = decanter.Calibration.from_dir(calib_dir)
   spectra = [decanter.reduce(obj, sky, calib) for obj, sky in frame_pairs]

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
