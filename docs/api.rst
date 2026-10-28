API reference
=============

Reduction
---------

.. autofunction:: decanter.reduce

.. autofunction:: decanter.reduce_many

.. autofunction:: decanter.reduce_to_warp_tree

.. autofunction:: decanter.combine

Calibration
-----------

.. autoclass:: decanter.Calibration
   :members: from_dir, assert_matches

.. autoclass:: decanter.InstrumentConfig
   :members:

.. autoexception:: decanter.CalibrationMismatch

Results
-------

.. autoclass:: decanter.Reduction
   :members:

.. autoclass:: decanter.OrderSpectrum
   :members:

Configuration
-------------

.. autoclass:: decanter.Config
   :members:

Wavelength calibration
----------------------

.. autofunction:: decanter.wavecal.anchor

.. autofunction:: decanter.wavecal.anchor_reduction

.. autofunction:: decanter.wavecal.telluric_check

.. autofunction:: decanter.wavecal.build_static_surface

.. autofunction:: decanter.wavecal.export_resample
