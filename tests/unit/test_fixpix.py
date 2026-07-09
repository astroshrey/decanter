"""Unit tests for :mod:`decanter.utils.fixpix`."""

from __future__ import annotations

import numpy as np
import pytest

from decanter.utils.fixpix import _axis_linear_interp, _axis_run_lengths, fixpix


def test_run_lengths_horizontal() -> None:
    """Row of [0,1,1,1,0,0,1,0] → run lengths [0,3,3,3,0,0,1,0]."""
    mask = np.array([[False, True, True, True, False, False, True, False]])
    runs = _axis_run_lengths(mask, axis=1)
    assert (runs[0] == [0, 3, 3, 3, 0, 0, 1, 0]).all()


def test_run_lengths_vertical() -> None:
    """Column-wise run lengths."""
    mask = np.array(
        [
            [False, True],
            [True, True],
            [True, False],
            [False, False],
        ]
    )
    runs = _axis_run_lengths(mask, axis=0)
    # Column 0: runs of length [0,2,2,0]; column 1: [2,2,0,0].
    assert (runs[:, 0] == [0, 2, 2, 0]).all()
    assert (runs[:, 1] == [2, 2, 0, 0]).all()


def test_axis_interp_single_pixel_along_row() -> None:
    """One bad pixel in the middle of a row interpolates as the average."""
    img = np.array([[1.0, 2.0, 99.0, 4.0, 5.0]], dtype=np.float32)
    mask = np.array([[False, False, True, False, False]])
    out = _axis_linear_interp(img, mask, axis=1)
    assert out[0, 2] == pytest.approx(3.0)


def test_axis_interp_two_run_along_row() -> None:
    """Two consecutive bad pixels: linear interp between the bracketing goods."""
    img = np.array([[1.0, 99.0, 99.0, 4.0]], dtype=np.float32)
    mask = np.array([[False, True, True, False]])
    out = _axis_linear_interp(img, mask, axis=1)
    assert out[0, 1] == pytest.approx(2.0)
    assert out[0, 2] == pytest.approx(3.0)


def test_fixpix_uses_shorter_axis() -> None:
    """A bad pixel with a long row-run and short col-run uses the column axis."""
    # Build a 5x5 image where row 2 is entirely bad except endpoints,
    # and col 2 has only the center pixel bad.
    img = np.zeros((5, 5), dtype=np.float32)
    img[:, 2] = np.array([10.0, 20.0, 99.0, 40.0, 50.0])  # col 2 with a bump
    img[2, :] = np.array([100.0, 99.0, 99.0, 99.0, 500.0])  # row 2 long bad-run
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 1:4] = True  # row 2 has a run of 3
    # Col 2: the masked pixel at (2,2) is a run of 1 along the column.
    out = fixpix(img, mask)
    # At (2, 2): row-run-length=3, col-run-length=1. Should use column interp.
    # Col 2 values: [10, 20, ?, 40, 50] → linear interp at 2 is 30.
    assert out[2, 2] == pytest.approx(30.0)


def test_fixpix_unchanged_when_no_mask() -> None:
    """All-zero mask returns input unchanged."""
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 20)).astype(np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    out = fixpix(img, mask)
    assert np.array_equal(out, img)


def test_fixpix_shape_mismatch_raises() -> None:
    img = np.ones((10, 10), dtype=np.float32)
    mask = np.zeros((10, 11), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        fixpix(img, mask)


def test_fixpix_int_mask_accepted() -> None:
    """Mask can be int (treats nonzero as bad)."""
    img = np.array([[1.0, 99.0, 3.0]], dtype=np.float32)
    mask = np.array([[0, 1, 0]], dtype=np.int16)
    out = fixpix(img, mask)
    assert out[0, 1] == pytest.approx(2.0)


def test_fixpix_edge_pixel_uses_constant_extrapolation() -> None:
    """A bad pixel at the row edge is filled with the nearest good value."""
    img = np.array([[99.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False, False, False]])
    out = fixpix(img, mask)
    # np.interp constant-extrapolates → nearest good = 2.0
    assert out[0, 0] == pytest.approx(2.0)
