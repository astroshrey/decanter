"""Per-machine roots for WINERED data and the WARP checkout.

decanter is mirrored between machines whose usernames differ (laptop
``/Users/shrey``, Mac mini ``/Users/shreyasvissapragada``) but whose
``~/Disk`` layout is identical. All reference paths resolve from
``$HOME`` so the same checkout works on both; override with the
``WINERED_ROOT`` / ``WARP_ROOT`` environment variables when needed.

One layout difference survives $HOME-relativization: the laptop keeps
raw nights and calibration sets together under ``winered/data/``
(nights named ``YYYYMMDD``), while the mini uses
``winered/science_data/<YYYY_MM_DD>`` for nights and
``winered/calibration_data/`` for calibration sets. Use
:data:`CAL_DATA_ROOTS` / :func:`raw_night_dir` instead of hardcoding
either convention.
"""
from __future__ import annotations

import os
from pathlib import Path

WINERED_ROOT = Path(
    os.environ.get("WINERED_ROOT", str(Path.home() / "Disk" / "winered"))
)
WARP_ROOT = Path(
    os.environ.get("WARP_ROOT", str(Path.home() / "Disk" / "codes" / "WARP"))
)

# Roots that may hold calibration_* set directories on either machine.
CAL_DATA_ROOTS = (
    WINERED_ROOT / "data",              # laptop layout
    WINERED_ROOT / "calibration_data",  # mac-mini layout
)


def raw_night_dir(laptop_name: str, mini_name: str) -> Path:
    """First existing raw-night dir given the two per-machine dir names."""
    laptop = WINERED_ROOT / "data" / laptop_name
    mini = WINERED_ROOT / "science_data" / mini_name
    for cand in (laptop, mini):
        if cand.is_dir():
            return cand
    return laptop
