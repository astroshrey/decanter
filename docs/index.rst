decanter
========

A fast, pure-Python reduction pipeline for **WINERED** near-infrared echelle
spectra. ``decanter`` runs the reduction of WARP — the reference IRAF-based
WINERED pipeline — and, by default, calibrates every frame's wavelengths on the
OH airglow and telluric H2O lines (:doc:`wavelength`). Its ``warp-clone`` mode
is bit-identical to WARP across all three instrument modes (HIRES-Y, HIRES-J,
WIDE).

Install
-------

.. code-block:: bash

   pip install -e .        # requires numpy, scipy, astropy, jax

Quickstart
----------

Load a calibration set, then reduce your ``(object, sky)`` frame pairs:

.. code-block:: python

   import decanter

   # Point at any calibration-set directory, or a WARP reduction root.
   calib = decanter.Calibration.from_dir("/path/to/calibration_set")

   r = decanter.reduce("WINA00045837.fits", calib, sky="WINA00045838.fits")

   spec = r.obj[(1.30, 163)]        # OrderSpectrum, keyed by (FSR cut, order)
   spec.wavelength, spec.flux       # vacuum-Å grid + flux (counts)

A whole night, every frame on its true telluric wavelength scale:

.. code-block:: python

   series = decanter.reduce_many(frame_pairs, calib)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   wavelength
   api
