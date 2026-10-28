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


# ---- WARP factor-map semantics (badpixmask.py:186-235) ------------------------
#
# Regression for the Qatar6 NO9/NO10 + HD207496 NO35 s02 mismatches: decanter's
# factor map used 30 half-open bins [-30,-28) ... [28,30), float64 bincount
# statistics and nanmax. WARP measures 28 bins (r_s, r_s+2] for r_s=-28..26
# with float32 np.std, and takes Python's builtin max (a NaN in the first bin
# makes the factor 1).

from decanter.utils.cosmic_ray import (  # noqa: E402
    _warp_window_factor,
    warp_slit_bins,
    warp_y_windows,
)


def test_warp_slit_bins_and_windows() -> None:
    b = warp_slit_bins(-30, 30, 2)
    assert len(b) == 28 and b[0] == (-28, -26) and b[-1] == (26, 28)
    assert warp_y_windows(2048, 100) == [(100 * j, 100 * (j + 1)) for j in range(19)]
    assert warp_y_windows(400, 100) == [(0, 100), (100, 200), (200, 300)]


def _one_order_frame(noisy_slit: float | None, *, first_slit: float = -29.0, seed: int = 5):
    """400x400 frame, one order (160) along y at x=200, integer slit coordinates
    (so pixels sit exactly on WARP's bin edges), constant raws, unit-noise diff,
    one column of huge diff noise at ``noisy_slit`` and a 600-ADU cosmic ray in
    the object frame at (150, 210)."""
    H = W = 400
    rng = np.random.default_rng(seed)
    x = np.arange(W)
    slit = np.broadcast_to((x - 200).astype(np.float64), (H, W))
    inside = (np.abs(slit) < 30) & (slit >= first_slit)
    apmask = np.where(inside, 160, 0).astype(np.int32)
    slitcoord = np.where(inside, slit, -10000.0)
    raw1 = np.full((H, W), 1000.0, dtype=np.float32)
    raw2 = np.full((H, W), 1000.0, dtype=np.float32)
    diff = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    if noisy_slit is not None:
        col = 200 + int(noisy_slit)
        diff[:, col] = rng.normal(0.0, 2000.0, H).astype(np.float32)
    raw1[150, 210] += 600.0
    diff[150, 210] += 600.0
    bp = np.zeros((H, W), dtype=np.int16)
    return diff, raw1, raw2, apmask, slitcoord, bp


def _detect(diff, raw1, raw2, apmask, slitcoord, bp):
    return detect_cosmic_rays(
        diff=diff, raw1=raw1, raw2=raw2, apmask=apmask, slitcoord=slitcoord,
        static_bp=bp, ndr1=8, ndr2=8, abba=True, echelle_orders=(160,),
        array_length=400)


@pytest.mark.parametrize("noisy_slit,found", [
    (-29.0, True),   # edge strip (-30, -28]: never measured by WARP
    (-28.0, True),   # bin edge: -28 belongs to (-30, -28], not (-28, -26]
    (29.0, True),    # edge strip (28, 30)
    (28.0, False),   # in (26, 28]: measured -> factor ~ 50 -> the CR is missed
    (-27.0, False),  # in (-28, -26]
])
def test_factor_map_uses_warp_slit_bins(noisy_slit: float, found: bool) -> None:
    res = _detect(*_one_order_frame(noisy_slit))
    assert bool(res.mask[150, 210]) is found


def test_factor_nan_first_bin_gives_factor_one() -> None:
    """Order without pixels in (-28, -26]: WARP's first std is NaN, so the
    builtin ``max`` keeps NaN and ``max(1., nan)`` is 1 even though another
    bin is very noisy; nanmax-based code missed this CR."""
    res = _detect(*_one_order_frame(0.0, first_slit=-25.0))
    assert res.mask[150, 210] == 1


def test_window_factor_python_max_semantics() -> None:
    sc = np.array([-27.0, -27.0, -25.0, -25.0, -23.0, -23.0])
    n = np.full(sc.shape, 2.0, dtype=np.float32)
    kw = dict(xlim1=-30, xlim2=30, bins=2, clipsigma=5., iteration=3)
    # First bin all zero: std 0 -> |v| < 0 keeps nothing -> NaN -> factor 1.
    d = np.array([0, 0, 50, -50, 0, 0], dtype=np.float32)
    assert _warp_window_factor(sc, d, n, **kw) == 1.0
    # NaN in a later bin is skipped by max: factor = std(second bin) / 2.
    d = np.array([1, -1, 50, -50, 0, 0], dtype=np.float32)
    f = _warp_window_factor(sc, d, n, **kw)
    assert f == np.float32(50.0) / np.float32(2.0) and isinstance(f, np.float32)


def test_noise_image_is_float32_left_to_right() -> None:
    from decanter.utils.cosmic_ray import warp_noise_image
    rng = np.random.default_rng(3)
    r1 = rng.uniform(0, 5e4, (30, 30)).astype(">f4")
    r2 = rng.uniform(0, 5e4, (30, 30)).astype(">f4")
    got = warp_noise_image(r1, r2, 8, 16)
    import scipy.ndimage
    from decanter.utils.cosmic_ray import _make_slant_footprint
    mf = scipy.ndimage.median_filter(np.absolute(r1 + r2), footprint=_make_slant_footprint())
    want = (mf * 2.27 + 8. ** 2. + 6. ** 2.) ** 0.5 / 2.27
    assert got.dtype == np.float32 and np.array_equal(got, want)
