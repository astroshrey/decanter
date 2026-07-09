"""Legacy listfile-driven pipeline orchestrator.

Historically called the per-stage ``run()`` functions in order; that
shape survives here as a thin back-compat wrapper for bench scripts
and tests that still iterate the active 13 stages explicitly. New
code should use :func:`decanter.reduce`.

The s00 calib loader and s14..s19 downstream stubs have been deleted;
:class:`decanter.Calibration` replaces s00, and continuum / vac→air /
SNR / plotting / organize are handled downstream by the user.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from decanter.config import Config
from decanter.stages import (
    s01_sky_subtract,
    s02_cosmic_ray,
    s03_apscatter,
    s04_flatfield,
    s05_badpix_interp,
    s06_transform_cut,
    s07_psf_center,
    s08_extract_1d,
    s09_extract_2d,
    s10_waveshift_measure,
    s11_waveshift_apply,
    s12_dispcor,
    s13_fsr_truncate,
)

# Active stages in WARP's call order. 13 entries (down from 20 — s00
# replaced by Calibration; s14..s19 deleted).
STAGES: tuple[ModuleType, ...] = (
    s01_sky_subtract,
    s02_cosmic_ray,
    s03_apscatter,
    s04_flatfield,
    s05_badpix_interp,
    s06_transform_cut,
    s07_psf_center,
    s08_extract_1d,
    s09_extract_2d,
    s10_waveshift_measure,
    s11_waveshift_apply,
    s12_dispcor,
    s13_fsr_truncate,
)


def run_pipeline(config: Config, workdir: Path, listfile: Path) -> None:
    """Run every active stage's ``run()`` in order against ``listfile``.

    Legacy entry point; new code should prefer :func:`decanter.reduce`,
    which composes the pure task functions directly and returns an
    in-memory :class:`Reduction` instead of writing per-stage FITS.
    """
    for stage in STAGES:
        stage.run(config=config, workdir=workdir, listfile=listfile)
