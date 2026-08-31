API reference
=============

Reduction
---------

.. autofunction:: decanter.reduce

.. autofunction:: decanter.reduce_many

.. autofunction:: decanter.calibrate_wavelengths

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

.. autoclass:: decanter.TransitSeries
   :members:

Configuration
-------------

.. autoclass:: decanter.Config
   :members:

.. autoclass:: decanter.WavecalConfig
   :members:

.. autoclass:: decanter.WavecalSolution
   :members:
