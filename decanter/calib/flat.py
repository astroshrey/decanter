"""Master flat construction.

WARP equivalent: ``Warp_calib.py:296-318`` (the calibration mode
pipeline). Phase 1 of decanter **consumes** an existing WARP master flat
from disk; reimplementing the calibration pipeline is Phase 2 / 3 work.
This module exists as a placeholder so the eventual flat builder has a
home.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_master_flat(flaton_paths: list[Path], flatoff_paths: list[Path]) -> np.ndarray:
    """Construct a master flat from raw flat-on and flat-off frames.

    Deferred to Phase 2 — out of scope for science-side parity testing.

    Raises:
        NotImplementedError: deferred to Phase 2.
    """
    raise NotImplementedError("calib.flat.build_master_flat: deferred to Phase 2")
