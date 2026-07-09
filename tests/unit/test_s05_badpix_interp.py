"""Unit tests for :mod:`decanter.image2d.fixpix`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.config import Config
from decanter.image2d.fixpix import run


def _write_frame(path: Path, data: np.ndarray, object_name: str = "STAR") -> None:
    header = fits.Header()
    header["OBJECT"] = object_name
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def test_static_only_interpolates(tmp_path: Path) -> None:
    """flag_bpmask=False uses only the static mask."""
    objname = "STAR"
    H, W = 20, 20
    img = np.ones((H, W), dtype=np.float32) * 5.0
    img[10, 10] = 999.0
    static_mask = np.zeros((H, W), dtype=np.int16)
    static_mask[10, 10] = 1

    _write_frame(tmp_path / "raw1.fits", img, object_name=objname)
    _write_frame(tmp_path / "raw2.fits", img, object_name=objname)
    _write_frame(tmp_path / f"{objname}_NO1_sscf.fits", img, object_name=objname)
    fits.PrimaryHDU(data=static_mask).writeto(tmp_path / "static_bp.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(flag_bpmask=False), tmp_path, listfile,
        static_bp_mask_path=tmp_path / "static_bp.fits")

    out = fits.getdata(tmp_path / f"{objname}_NO1_sscfm.fits")
    assert out[10, 10] == pytest.approx(5.0, abs=1e-5)
    # Untouched pixels stay at 5.0
    assert out[0, 0] == pytest.approx(5.0)


def test_combines_static_and_cr(tmp_path: Path) -> None:
    """flag_bpmask=True ORs the CR mask into the static mask."""
    objname = "STAR"
    H, W = 20, 20
    img = np.ones((H, W), dtype=np.float32) * 5.0
    img[5, 5] = 999.0  # CR pixel (only in CR mask)
    img[15, 15] = -999.0  # static bad pixel (only in static mask)

    static_mask = np.zeros((H, W), dtype=np.int16)
    static_mask[15, 15] = 1
    cr_mask = np.zeros((H, W), dtype=np.int16)
    cr_mask[5, 5] = 1

    _write_frame(tmp_path / "raw1.fits", img, object_name=objname)
    _write_frame(tmp_path / "raw2.fits", img, object_name=objname)
    _write_frame(tmp_path / f"{objname}_NO1_sscf.fits", img, object_name=objname)
    fits.PrimaryHDU(data=static_mask).writeto(tmp_path / "static_bp.fits")
    fits.PrimaryHDU(data=cr_mask).writeto(tmp_path / f"mask_{objname}_NO1_s.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(flag_bpmask=True), tmp_path, listfile,
        static_bp_mask_path=tmp_path / "static_bp.fits")

    out = fits.getdata(tmp_path / f"{objname}_NO1_sscfm.fits")
    assert out[5, 5] == pytest.approx(5.0, abs=1e-5)
    assert out[15, 15] == pytest.approx(5.0, abs=1e-5)


def test_nan_in_input_treated_as_bad(tmp_path: Path) -> None:
    """A NaN pixel (e.g., divide-by-zero from s04) is interpolated."""
    objname = "STAR"
    H, W = 20, 20
    img = np.ones((H, W), dtype=np.float32) * 5.0
    img[10, 10] = np.nan

    static_mask = np.zeros((H, W), dtype=np.int16)
    _write_frame(tmp_path / "raw1.fits", img, object_name=objname)
    _write_frame(tmp_path / "raw2.fits", img, object_name=objname)
    _write_frame(tmp_path / f"{objname}_NO1_sscf.fits", img, object_name=objname)
    fits.PrimaryHDU(data=static_mask).writeto(tmp_path / "static_bp.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(flag_bpmask=False), tmp_path, listfile,
        static_bp_mask_path=tmp_path / "static_bp.fits")

    out = fits.getdata(tmp_path / f"{objname}_NO1_sscfm.fits")
    assert np.isfinite(out).all()
    assert out[10, 10] == pytest.approx(5.0, abs=1e-5)


def test_bp_mask_header_keyword(tmp_path: Path) -> None:
    """BP_MASK header keyword records which mask source was used."""
    objname = "STAR"
    H, W = 10, 10
    img = np.ones((H, W), dtype=np.float32)
    static_mask = np.zeros((H, W), dtype=np.int16)
    cr_mask = np.zeros((H, W), dtype=np.int16)

    _write_frame(tmp_path / "raw1.fits", img, object_name=objname)
    _write_frame(tmp_path / "raw2.fits", img, object_name=objname)
    _write_frame(tmp_path / f"{objname}_NO1_sscf.fits", img, object_name=objname)
    fits.PrimaryHDU(data=static_mask).writeto(tmp_path / "static_bp.fits")
    fits.PrimaryHDU(data=cr_mask).writeto(tmp_path / f"mask_{objname}_NO1_s.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(flag_bpmask=True), tmp_path, listfile,
        static_bp_mask_path=tmp_path / "static_bp.fits")

    h = fits.getheader(tmp_path / f"{objname}_NO1_sscfm.fits")
    assert h["BP_MASK"] == "static_bp.fits"


def test_missing_static_path_raises(tmp_path: Path) -> None:
    listfile = tmp_path / "list.txt"
    listfile.write_text("a.fits b.fits\n")
    with pytest.raises(ValueError, match="static_bp_mask_path"):
        run(Config(), tmp_path, listfile)
