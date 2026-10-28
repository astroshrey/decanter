"""Unit tests for :mod:`decanter.utils.fixpix` (bit-exact IRAF fixpix port).

Expected values were checked against IRAF's compiled ``xtfixpix.o`` /
``xtfp.o`` (``scripts/iraf_oracle/fixpix_oracle.c``) and against
``iraf.fixpix`` run through pyraf.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from decanter.utils.fixpix import fixpix, iraf_bad_pixels

f32 = np.float32


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=f32).view(np.int32)


def test_single_pixel_along_line() -> None:
    """One bad pixel: line/column runs tie (1 == 1) -> interpolate along the line."""
    img = np.array([[0, 0, 0, 0, 0],
                    [1, 2, 99, 4, 5],
                    [0, 0, 0, 0, 0]], dtype=f32)
    mask = np.zeros(img.shape, bool)
    mask[1, 2] = True
    out = fixpix(img, mask)
    assert out[1, 2] == f32(3.0)


def test_uses_narrower_axis() -> None:
    """A long line run with a short column run is interpolated along the column."""
    img = np.zeros((5, 5), dtype=f32)
    img[:, 2] = [10.0, 20.0, 99.0, 40.0, 50.0]
    img[2, :] = [100.0, 99.0, 99.0, 99.0, 500.0]
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 1:4] = True                       # line run 3, column runs 1
    out = fixpix(img, mask)
    assert out[2, 2] == f32(30.0)             # (20 + 40) / 2 along column 2


def test_square_block_uses_line() -> None:
    """Equal line and column run lengths interpolate along the line."""
    img = np.arange(25, dtype=f32).reshape(5, 5) ** 2
    mask = np.zeros((5, 5), bool)
    mask[1:4, 1:4] = True
    out = fixpix(img, mask)
    a, c = img[2, 0], img[2, 4]
    b = (c - a) / f32(4)
    assert out[2, 1] == f32(a + b)


def test_fused_multiply_add() -> None:
    """``val = a + b*(i-c1)`` is one fused multiply-add, as clang compiled it."""
    row = np.array([-836.89, 710.45, 722.57, 753.07, -56.18], dtype=f32)
    img = np.zeros((5, 5), dtype=f32)
    img[1] = row
    mask = np.zeros((5, 5), bool)
    mask[1:4, 1:4] = True
    out = fixpix(img, mask)
    b = (row[4] - row[0]) / f32(4)
    unfused = f32(row[0] + f32(b * f32(3)))
    assert out[1, 3] == f32(-251.3574981689453)
    assert out[1, 3] != unfused


def test_edges_extrapolate_constant() -> None:
    """Runs touching an edge take the one available end point."""
    img = np.array([[99.0, 2.0, 3.0, 4.0, 99.0]], dtype=f32)
    img = np.vstack([img, img + 10, img + 20])
    mask = np.zeros(img.shape, bool)
    mask[1, 0] = mask[1, 4] = True
    out = fixpix(img, mask)
    assert out[1, 0] == f32(12.0)
    assert out[1, 4] == f32(14.0)


def test_negative_zero_through_fma() -> None:
    """Left/right-edge line runs go through the FMA with b = 0 (-0 -> +0);
    bottom-edge column runs assign the end point directly (-0 stays -0)."""
    img = np.zeros((3, 5), dtype=f32)
    img[1, 3] = f32(-0.0)
    mask = np.zeros(img.shape, bool)
    mask[1, 4] = True                       # line interp, right edge: fma(0, 1, -0)
    out = fixpix(img, mask)
    assert _bits(out)[1, 4] == _bits(np.array(0.0, f32))
    img2 = np.zeros((3, 5), dtype=f32)
    img2[1, 2] = f32(-0.0)
    mask2 = np.zeros(img2.shape, bool)
    mask2[2, 0:3] = True                    # line run 3 > column run 1
    out2 = fixpix(img2, mask2)
    assert _bits(out2)[2, 2] == _bits(np.array(-0.0, f32))


def test_unchanged_when_no_mask() -> None:
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 20)).astype(f32)
    out = fixpix(img, np.zeros((20, 20), dtype=bool))
    assert np.array_equal(_bits(out), _bits(img))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        fixpix(np.ones((10, 10), f32), np.zeros((10, 11), dtype=bool))


def test_mask_image_semantics() -> None:
    """Mask images are read as INTEGER (truncated); values > 0 are bad."""
    mask = np.array([[0.0, 0.5, 0.999, 1.0, 2.7, -1.0, -3.5, 3.0]], dtype=f32)
    assert iraf_bad_pixels(mask).tolist() == [[False, False, False, True, True,
                                                False, False, True]]
    assert iraf_bad_pixels(np.array([[0, 1, -2, 3]], np.int16)).tolist() == [
        [False, True, False, True]]


def test_only_bad_pixels_change() -> None:
    rng = np.random.default_rng(3)
    img = rng.normal(size=(40, 50)).astype(f32)
    mask = rng.random((40, 50)) < 0.2
    out = fixpix(img, mask)
    assert np.array_equal(_bits(out)[~mask], _bits(img)[~mask])
    # Output never depends on the values at bad pixels.
    img2 = img.copy()
    img2[mask] = 1e6
    assert np.array_equal(_bits(fixpix(img2, mask)), _bits(out))


@pytest.mark.regression
def test_warp_parity_toi2109(warp_reference_dir) -> None:
    """0 differing float32 pixels vs WARP's ``_sscfm`` and sky ``_fm``."""
    d = warp_reference_dir / "TOI2109_NO1" / "intermediate_files"
    mask = fits.getdata(d / "OBJ/4-OBJ_mask/maskflat_TOI2109_NO1_s.fits")
    for src, exp in (("OBJ/3-OBJ_flat/TOI2109_NO1_sscf.fits",
                      "OBJ/4-OBJ_mask/TOI2109_NO1_sscfm.fits"),
                     ("SKY/1-SKY_flat/TOI2109_skyNO1_f.fits",
                      "SKY/2-SKY_mask/TOI2109_skyNO1_fm.fits")):
        out = fixpix(fits.getdata(d / src), mask)
        assert int((_bits(out) != _bits(fits.getdata(d / exp))).sum()) == 0
