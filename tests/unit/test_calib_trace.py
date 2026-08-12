"""Tests for aperture tracing from the multihole frame (calibration stage 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decanter.calib.trace import _REF_DIR, _REF_NPZ, trace_apertures

_RAW = Path("/Users/shreyasvissapragada/Disk/winered/science_data/2025_08_07")
_MULTIHOLE = [_RAW / f"WINA000535{n}.fits" for n in (61, 62, 63)]
_APDB = Path("/Users/shreyasvissapragada/Disk/winered/reductions/TOI2109/"
             "calibration_data/database/apmultihole_HIRES-Y100_20250806")
_LIVE = all(p.is_file() for p in _MULTIHOLE) and _APDB.is_file()


def test_reference_templates_bundled():
    for name in _REF_NPZ.values():
        z = np.load(_REF_DIR / name, allow_pickle=True)
        assert {"centers", "orders", "traceid"} <= set(z.keys())


@pytest.mark.skipif(not _LIVE, reason="live WINERED multihole frames not available")
def test_trace_matches_warp_apmultihole():
    """Traced detector aperture x(y) matches WARP's apmultihole DB to <0.1 px."""
    from decanter.calib.aperture import ApertureSet
    from decanter.calib.trace import average_frames

    img = average_frames(_MULTIHOLE)
    res = trace_apertures(img, "HIRES-Y")
    aps = ApertureSet.load(_APDB, array_length=img.shape[0])
    rms = []
    for m in range(159, 185):
        if m not in res.traces or m not in aps.apertures:
            continue
        mine = np.asarray(res.traces[m], float)
        warp = np.asarray(aps.apertures[m].trace_x_clamped, float)
        n = min(mine.size, warp.size)
        rms.append(np.std((mine[:n] - warp[:n])[100:n - 100]))
    assert len(rms) >= 24
    assert np.median(rms) < 0.1, f"trace RMS vs WARP too high: {np.median(rms):.3f} px"
