"""Unit tests for :mod:`decanter.utils.cosmic_ray`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from decanter.utils.cosmic_ray import (
    _make_slant_footprint,
    _median_filter_pair,
    detect_cosmic_rays,
    ndr_from_header,
)


# ---- helper smoke tests --------------------------------------------------


def test_slant_footprint_is_square() -> None:
    fp = _make_slant_footprint(angle_deg=85.0, window=10)
    assert fp.shape == (10, 10)
    assert fp.dtype.kind in ("i", "u")
    assert fp.sum() > 0
    # Symmetric about the center because we use a single-pixel band.
    assert fp[5, 5] == 1


def test_median_filter_pair_zeroes_constant() -> None:
    """A constant frame has zero residual after median-filter pair."""
    img = np.full((20, 20), 5.0, dtype=np.float32)
    resid = _median_filter_pair(img, medsize=5)
    assert np.allclose(resid, 0.0)


def test_median_filter_pair_spikes_an_isolated_outlier() -> None:
    """An isolated bright pixel survives the median-filter pair largely intact."""
    img = np.zeros((40, 40), dtype=np.float32)
    img[20, 20] = 1000.0
    resid = _median_filter_pair(img, medsize=5)
    assert resid[20, 20] > 900.0  # most of the spike still there
    # Other pixels are ~0 (filter removed flat background).
    far_mask = np.ones_like(resid, dtype=bool)
    far_mask[18:23, 18:23] = False
    assert np.allclose(resid[far_mask], 0.0, atol=1e-5)


# ---- NDR header parsing -----------------------------------------------------


@pytest.mark.parametrize(
    "exptime,expected",
    [(3.0, 1), (10.0, 2), (20.0, 4), (60.0, 8), (300.0, 8), (1000.0, 16)],
)
def test_ndr_staircase_for_32_outputs(exptime: float, expected: int) -> None:
    """The NDR table for NOUTPUTS=32 is a staircase keyed on EXPTIME."""
    assert ndr_from_header({"EXPTIME": exptime, "NOUTPUTS": 32}) == expected


def test_ndr_returns_header_value_when_present() -> None:
    """If the FITS header carries an explicit NDR, the staircase isn't used."""
    assert ndr_from_header({"NDR": 4, "EXPTIME": 1000.0, "NOUTPUTS": 32}) == 4


def test_ndr_default_for_non_32_output_modes() -> None:
    """NOUTPUTS != 32 always falls back to NDR=1."""
    assert ndr_from_header({"EXPTIME": 1.0, "NOUTPUTS": 1}) == 1
    assert ndr_from_header({"EXPTIME": 1000.0, "NOUTPUTS": 8}) == 1


# ---- detection on synthetic data -------------------------------------------


def _make_synthetic_inputs(
    *,
    H: int = 200,
    W: int = 200,
    n_orders: int = 2,
    rng_seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Build a tiny synthetic dataset for fast unit-testing.

    Two parallel apertures running along Y at x=70 and x=130, each ±20 pixels
    wide. Slitcoord = x - center for in-aperture pixels, sentinel outside.
    Static bp mask is all zeros.

    Returns ``(diff, raw1, raw2, apmask, slitcoord, static_bp)``.
    """
    rng = np.random.default_rng(rng_seed)
    # Background: ~zero (already sky-subtracted-ish) in diff; positive in raws.
    raw1 = rng.normal(loc=1000.0, scale=10.0, size=(H, W)).astype(np.float32)
    raw2 = rng.normal(loc=1000.0, scale=10.0, size=(H, W)).astype(np.float32)
    diff = raw1 - raw2
    apmask = np.zeros((H, W), dtype=np.int32)
    slitcoord = np.full((H, W), -10000.0, dtype=np.float64)
    centers = [70, 130]
    for i, cx in enumerate(centers, start=1):
        # ±20 px window per aperture
        x_idx = np.arange(W)
        in_x = np.abs(x_idx - cx) < 20
        apmask[:, in_x] = 159 + i  # orders 160 and 161
        residue = (x_idx[None, :] - cx).astype(np.float64)
        residue = np.broadcast_to(residue, (H, W))
        slitcoord = np.where(apmask == 159 + i, residue, slitcoord)
    static_bp = np.zeros((H, W), dtype=np.int16)
    return diff, raw1, raw2, apmask, slitcoord, static_bp


def test_clean_data_has_zero_cosmic_rays() -> None:
    """No injected CRs → detector returns an empty mask (or near-empty)."""
    diff, raw1, raw2, apmask, slitcoord, bp = _make_synthetic_inputs(rng_seed=11)
    result = detect_cosmic_rays(
        diff=diff,
        raw1=raw1,
        raw2=raw2,
        apmask=apmask,
        slitcoord=slitcoord,
        static_bp=bp,
        ndr1=8,
        ndr2=8,
        abba=True,
        echelle_orders=(160, 161),
        array_length=200,
        ystep=50,
        bins=2,
        xlim1=-20.0,
        xlim2=20.0,
    )
    # With sigma=10 threshold and clean Gaussian data, expect zero or a tiny
    # handful of false positives.
    assert result.n_cosmic_rays < 5, f"too many false positives: {result.n_cosmic_rays}"
    assert result.mask.dtype == np.int16
    assert result.mask.shape == diff.shape


def test_injected_cosmic_rays_are_recovered() -> None:
    """Inject 20 bright CRs inside the apertures; recall must be 100%."""
    diff, raw1, raw2, apmask, slitcoord, bp = _make_synthetic_inputs(rng_seed=22)
    # Inject CRs into the obj frame at known (y, x) locations inside aperture 1.
    rng = np.random.default_rng(seed=99)
    cr_positions: list[tuple[int, int]] = []
    # H=200, ystep=50 → n_y_tiles=3 → valid Y in (0, 150] (1-indexed).
    # Inject CRs strictly inside this range so the y-tile gate doesn't drop them.
    while len(cr_positions) < 20:
        y = int(rng.integers(20, 145))
        x = int(rng.integers(55, 85))  # inside aperture 1 (center 70, ±20)
        pos = (y, x)
        if pos not in cr_positions:
            cr_positions.append(pos)
    for y, x in cr_positions:
        raw1[y, x] += 5000.0  # huge, well above noise
    # diff = raw1 - raw2, so the CR shows up positively in diff.
    diff = raw1 - raw2

    result = detect_cosmic_rays(
        diff=diff,
        raw1=raw1,
        raw2=raw2,
        apmask=apmask,
        slitcoord=slitcoord,
        static_bp=bp,
        ndr1=8,
        ndr2=8,
        abba=True,
        echelle_orders=(160, 161),
        array_length=200,
        ystep=50,
        bins=2,
        xlim1=-20.0,
        xlim2=20.0,
        # Loosen the position-cluster check — synthetic CRs are inside the
        # aperture by construction, so the ABBA position ratio fires; bypass.
        slitposratio=100.0,
    )
    recovered = sum(1 for y, x in cr_positions if result.mask[y, x] == 1)
    assert recovered == len(cr_positions), (
        f"recovered {recovered}/{len(cr_positions)} injected CRs"
    )
    assert result.n_cosmic_rays >= len(cr_positions)


def test_static_bp_pixels_kept_in_saved_mask_but_excluded_from_histogram() -> None:
    """WARP saves the *unfiltered* maskarray (badpixmask.py:295). Static-BP
    pixels that also trigger as CR DO appear in the saved mask. They only
    get filtered out of the var/ave histogram check via
    ``reqmask = (maskarray == 1) & (bpfdata == 0)`` (badpixmask.py:249).

    This is a regression test for the parity bug where decanter was excluding
    static-BP pixels from the saved mask, costing us ~10% recall vs WARP
    (133 of 141 "WARP-only" pixels on TOI2109 frame 1 were exactly this
    case).
    """
    diff, raw1, raw2, apmask, slitcoord, bp = _make_synthetic_inputs(rng_seed=33)
    # Inject a CR inside aperture 1; also flag it as static bad.
    raw1[100, 70] += 5000.0
    diff = raw1 - raw2
    bp[100, 70] = 1

    result = detect_cosmic_rays(
        diff=diff, raw1=raw1, raw2=raw2,
        apmask=apmask, slitcoord=slitcoord, static_bp=bp,
        ndr1=8, ndr2=8, abba=True,
        echelle_orders=(160, 161), array_length=200,
        ystep=50, bins=2, xlim1=-20.0, xlim2=20.0,
        slitposratio=100.0,
    )
    assert result.mask[100, 70] == 1, "static-BP CR should be kept in the saved mask (matches WARP)"


def test_threshold_clamped_to_max_sigma() -> None:
    """If threshold > max_sigma, threshold is lowered to max_sigma (WARP line 119-120)."""
    diff, raw1, raw2, apmask, slitcoord, bp = _make_synthetic_inputs()
    result = detect_cosmic_rays(
        diff=diff, raw1=raw1, raw2=raw2,
        apmask=apmask, slitcoord=slitcoord, static_bp=bp,
        ndr1=8, ndr2=8, abba=True,
        echelle_orders=(160, 161), array_length=200,
        ystep=50, bins=2, xlim1=-20.0, xlim2=20.0,
        threshold=99.0,
        max_sigma=15.0,
    )
    assert result.final_threshold <= 15.0


def test_negative_diff_attributes_cr_to_sky_frame() -> None:
    """A CR injected in raw2 (sky) → negative spike in diff → still detected."""
    diff, raw1, raw2, apmask, slitcoord, bp = _make_synthetic_inputs(rng_seed=44)
    raw2[80, 130] += 5000.0  # CR in sky frame, aperture 2 (center 130)
    diff = raw1 - raw2  # negative spike at (80, 130)

    result = detect_cosmic_rays(
        diff=diff, raw1=raw1, raw2=raw2,
        apmask=apmask, slitcoord=slitcoord, static_bp=bp,
        ndr1=8, ndr2=8, abba=True,
        echelle_orders=(160, 161), array_length=200,
        ystep=50, bins=2, xlim1=-20.0, xlim2=20.0,
        slitposratio=100.0,
    )
    assert result.mask[80, 130] == 1, "negative-diff CR should still be flagged"
