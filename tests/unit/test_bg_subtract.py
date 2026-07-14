"""box_extract background subtraction: a flat background under the aperture
is removed exactly, leaving the stellar flux."""
from __future__ import annotations

import numpy as np

from decanter.extract.box_extract_1d import box_extract


def _strip(background: float, star_amp: float = 500.0):
    H, W = 40, 120
    x = np.arange(1, W + 1)
    star_x = 60.0
    prof = star_amp * np.exp(-0.5 * ((x - star_x) / 2.0) ** 2)  # narrow star
    img = np.tile(prof, (H, 1)) + background                     # + flat background
    trace = np.full(H, star_x, dtype=float)
    return img.astype(np.float32), trace


def test_flat_background_removed_exactly():
    img, trace = _strip(background=10.0)
    raw = box_extract(img, trace, ap_low=-6.0, ap_high=6.0)
    sub = box_extract(img, trace, ap_low=-6.0, ap_high=6.0, subtract_background=True)
    # aperture width ~12 px * background 10 = ~120 ct removed per row
    removed = np.median(raw - sub)
    assert 115 < removed < 125
    # a zero-background strip is untouched
    img0, trace0 = _strip(background=0.0)
    assert np.allclose(box_extract(img0, trace0, ap_low=-6, ap_high=6),
                       box_extract(img0, trace0, ap_low=-6, ap_high=6,
                                   subtract_background=True), atol=1e-3)


def test_background_default_off_is_noop():
    img, trace = _strip(background=25.0)
    assert np.array_equal(
        box_extract(img, trace, ap_low=-6, ap_high=6),
        box_extract(img, trace, ap_low=-6, ap_high=6, subtract_background=False))
