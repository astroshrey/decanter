"""Parse ThAr linelists used by the wavelength calibration.

Source files: e.g. ``reference/HIRES-Y/winered_hiresy_linelist_20220713.txt``.
Format is ASCII; one row per line with the central wavelength (vacuum
Å) and optional metadata.

These linelists feed s12 (dispcor) — decanter Phase 1 consumes WARP's
pre-fit dispersion solutions rather than refitting from the linelist,
so this loader is currently informational rather than load-bearing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    """Read a ThAr linelist, returning an array of wavelengths in vacuum Å.

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("io.linelists.load: not yet implemented")
