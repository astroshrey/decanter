"""Legacy stage shim — phase-2 back-compat layer.

Real work has moved to:

    decanter.image2d     # was s01..s05
    decanter.rectify     # was s06
    decanter.extract     # was s07..s09
    decanter.waveshift   # was s10..s11
    decanter.wavelength  # was s12..s13

This shim re-exports each of those modules under their historical
``sNN_*`` names so existing ``from decanter.stages import s01_sky_subtract``
imports continue to work for tests and bench scripts. New code should
prefer :func:`decanter.reduce` and the task-grouped modules directly.

We use :func:`importlib.import_module` (not ``import as``) because
each target package's ``__init__.py`` re-exports the same-named
function (e.g. ``from decanter.image2d.sky_subtract import sky_subtract``),
which shadows the submodule attribute on the package. ``import_module``
bypasses the package namespace and returns the actual module object so
legacy ``module.run(...)`` callers keep working.

The s00 calibration loader and s14..s19 downstream stubs have been
deleted as part of the Phase-2 cleanup. ``Calibration.from_dir``
replaces s00; the s14..s19 work (continuum / vac2air / SNR /
plotting / organize / report) is handled downstream by the user with
different techniques.
"""

import importlib

s01_sky_subtract = importlib.import_module("decanter.image2d.sky_subtract")
s02_cosmic_ray = importlib.import_module("decanter.image2d.cosmic_ray")
s03_apscatter = importlib.import_module("decanter.image2d.apscatter")
s04_flatfield = importlib.import_module("decanter.image2d.flatfield")
s05_badpix_interp = importlib.import_module("decanter.image2d.fixpix")
s06_transform_cut = importlib.import_module("decanter.rectify.transform")
s07_psf_center = importlib.import_module("decanter.extract.psf_center")
s08_extract_1d = importlib.import_module("decanter.extract.box_extract_1d")
s09_extract_2d = importlib.import_module("decanter.extract.strip_extract_2d")
s10_waveshift_measure = importlib.import_module("decanter.waveshift.measure")
s11_waveshift_apply = importlib.import_module("decanter.waveshift.apply")
s12_dispcor = importlib.import_module("decanter.wavelength.dispcor")
s13_fsr_truncate = importlib.import_module("decanter.wavelength.fsr")
