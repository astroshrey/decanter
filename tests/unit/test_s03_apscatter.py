"""Unit tests for :mod:`decanter.image2d.apscatter`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.calib.aperture import Aperture, ApertureSet
from decanter.config import Config
from decanter.io.apdb import ApertureEntry, FUNCTION_TYPE_CHEBYSHEV
from decanter.image2d.apscatter import apscatter_model, build_sample_mask, run


def _make_constant_entry(order: int, center_x: float) -> ApertureEntry:
    """Aperture with a constant trace at ``center_x``."""
    return ApertureEntry(
        order=order,
        center_x=center_x,
        center_y=1024.0,
        low=-20.0,
        high=20.0,
        function_type=FUNCTION_TYPE_CHEBYSHEV,
        poly_order=2,
        y_min=1.0,
        y_max=2048.0,
        coefficients=(0.0, 0.0),
        background_sample="INDEF",
    )


def _make_apset(centers: list[float] = [500.0, 1500.0], L: int = 200) -> ApertureSet:
    apertures = {
        159 + i: Aperture(entry=_make_constant_entry(159 + i, cx), array_length=L)
        for i, cx in enumerate(centers)
    }
    return ApertureSet(apertures=apertures, array_length=L)


def test_sample_mask_excludes_apertures_and_edges() -> None:
    """Mask is False inside apertures and outside the x-sample range."""
    apset = _make_apset(centers=[100.0], L=200)
    mask = build_sample_mask(apset, sample_x_lo=9, sample_x_hi=1999)
    # Inside aperture at x=100 (1-indexed → 0-indexed ~99): mask should be False
    # (apmask_array uses 1-indexed columns; the aperture spans 0-indexed ~80..120).
    assert not mask[100, 99]
    # Far from aperture: True (still inside sample range; for L=200, sample_x_hi=1999
    # means "everything", so only the inner edge zeros matter).
    assert mask[100, 50] == True  # noqa: E712


def test_sample_mask_honors_asymmetric_edge_aperture() -> None:
    """An aperture with ``low=-500, high=32`` excludes ``[trace-500, trace+32]``.

    Regression test for the apscatter parity bug: WARP's apsc_maskfile uses
    the asymmetric ``low=-500`` convention on the leftmost aperture to mean
    "extend to the image edge". Using ``|residue| < high`` instead of the
    signed bounds would incorrectly leave the leftmost ~95 columns as
    valid scatter samples, biasing the Legendre fit by ~1 count.
    """
    # Synthetic: one wide aperture at center=66 with low=-500, high=+32 (the WARP convention).
    L = 200
    entry = ApertureEntry(
        order=159,
        center_x=66.0,
        center_y=100.0,
        low=-500.0,
        high=32.0,
        function_type=FUNCTION_TYPE_CHEBYSHEV,
        poly_order=2,
        y_min=1.0,
        y_max=float(L),
        coefficients=(0.0, 0.0),  # constant trace at x=66
        background_sample="INDEF",
    )
    apset = ApertureSet(apertures={159: Aperture(entry=entry, array_length=L)}, array_length=L)
    mask = build_sample_mask(apset, sample_x_lo=9, sample_x_hi=L - 11)
    # Cols 0-indexed 33..66 should be excluded (signed: -33 < residue < 32 covers x-66 ∈ (-500, 32),
    # i.e. x < 98). Specifically: x=50 (1-idx) is at residue=50-66=-16 → in (-500, 32) → EXCLUDED.
    # 1-indexed x=50 → 0-indexed col 49.
    assert not mask[100, 49], "col 50 (1-idx) should be inside the wide-edge aperture"
    # But col x=100 (1-idx) → residue=34 → outside (-500, 32) → INCLUDED.
    assert mask[100, 99], "col 100 (1-idx) should be outside the aperture"


def test_apscatter_model_subtracts_flat_background() -> None:
    """A constant-offset frame produces a constant-offset model."""
    H, W = 200, 200
    apset = _make_apset(centers=[100.0], L=H)
    sample_mask = build_sample_mask(apset, sample_x_lo=9, sample_x_hi=W - 11)
    frame = np.full((H, W), 50.0, dtype=np.float64)
    model = apscatter_model(
        frame, sample_mask,
        ap1_niterate=5, ap2_niterate=5,
        sample_y_lo=9, sample_y_hi=H - 11,
    )
    # The fit should recover ~50 everywhere within the sample region.
    inner = model[20:180, 20:180]
    assert np.allclose(inner, 50.0, atol=0.1)


def test_apscatter_model_subtracts_linear_x_gradient() -> None:
    """A linear-in-x scattered light is reproduced by the degree-3 fit."""
    H, W = 200, 200
    apset = _make_apset(centers=[100.0], L=H)
    sample_mask = build_sample_mask(apset, sample_x_lo=9, sample_x_hi=W - 11)
    x = np.arange(W, dtype=np.float64)
    frame = np.broadcast_to(2.0 + 0.05 * x, (H, W)).astype(np.float64)
    model = apscatter_model(
        frame, sample_mask,
        ap1_niterate=5, ap2_niterate=5,
        sample_y_lo=9, sample_y_hi=H - 11,
    )
    inner = model[50:150, 30:170]
    truth = frame[50:150, 30:170]
    # Look at relative error excluding regions near the aperture edge.
    diff = np.abs(inner - truth)
    assert diff.max() < 0.5  # generous; legendre fit + smoothing is approximate


def test_run_gate_copies_through_when_flag_off(tmp_path: Path) -> None:
    """flag_apscatter=False → output is byte-identical to input."""
    objname = "STAR"
    in_path = tmp_path / f"{objname}_NO1_s.fits"
    out_path = tmp_path / f"{objname}_NO1_ssc.fits"
    raw_obj = tmp_path / "raw1.fits"
    raw_sky = tmp_path / "raw2.fits"
    listfile = tmp_path / "list.txt"

    data = np.linspace(0, 100, 100 * 100).reshape(100, 100).astype(np.float32)
    header = fits.Header()
    header["OBJECT"] = objname
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)
    fits.PrimaryHDU(data=data, header=header).writeto(raw_obj)
    fits.PrimaryHDU(data=data, header=header).writeto(raw_sky)
    listfile.write_text("raw1.fits raw2.fits\n")

    cfg = Config(flag_apscatter=False)
    run(cfg, tmp_path, listfile)
    out_data, _ = fits.getdata(out_path), fits.getheader(out_path)
    assert np.array_equal(out_data, data)


def test_run_requires_apdb_when_enabled(tmp_path: Path) -> None:
    """flag_apscatter=True without an apdb path raises a clear error."""
    listfile = tmp_path / "list.txt"
    listfile.write_text("a.fits b.fits\n")
    with pytest.raises(ValueError, match="apdb_path"):
        run(Config(flag_apscatter=True), tmp_path, listfile)
