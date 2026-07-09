"""Unit tests for :mod:`decanter.image2d.flatfield`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.config import Config
from decanter.image2d.flatfield import run


def _write_simple_frame(path: Path, data: np.ndarray, object_name: str = "STAR") -> None:
    header = fits.Header()
    header["OBJECT"] = object_name
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def test_divides_by_flat(tmp_path: Path) -> None:
    """Output equals input / flat (within float32 round-trip)."""
    objname = "STAR"
    data = np.full((50, 50), 100.0, dtype=np.float32)
    flat = np.full((50, 50), 2.0, dtype=np.float32)

    raw_path = tmp_path / "raw1.fits"
    sky_path = tmp_path / "raw2.fits"
    flat_path = tmp_path / "flat.fits"
    ssc_path = tmp_path / f"{objname}_NO1_ssc.fits"
    listfile = tmp_path / "list.txt"

    _write_simple_frame(raw_path, data, object_name=objname)
    _write_simple_frame(sky_path, data, object_name=objname)
    _write_simple_frame(ssc_path, data, object_name=objname)
    fits.PrimaryHDU(data=flat).writeto(flat_path)
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(), tmp_path, listfile, flat_path=flat_path)

    out_path = tmp_path / f"{objname}_NO1_sscf.fits"
    out = fits.getdata(out_path)
    assert np.allclose(out, 50.0)


def test_flat_header_added(tmp_path: Path) -> None:
    """FLAT keyword is added to the output header naming the flat used."""
    objname = "STAR"
    data = np.ones((10, 10), dtype=np.float32) * 4.0
    flat = np.ones((10, 10), dtype=np.float32) * 2.0
    _write_simple_frame(tmp_path / "raw1.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / "raw2.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / f"{objname}_NO1_ssc.fits", data, object_name=objname)
    fits.PrimaryHDU(data=flat).writeto(tmp_path / "my_flat.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(), tmp_path, listfile, flat_path=tmp_path / "my_flat.fits")

    out_header = fits.getheader(tmp_path / f"{objname}_NO1_sscf.fits")
    assert "FLAT" in out_header
    assert out_header["FLAT"] == "my_flat.fits"


def test_zero_flat_produces_zero(tmp_path: Path) -> None:
    """Division by zero is set to 0 (matches IRAF imarith behavior)."""
    objname = "STAR"
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    flat = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    _write_simple_frame(tmp_path / "raw1.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / "raw2.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / f"{objname}_NO1_ssc.fits", data, object_name=objname)
    fits.PrimaryHDU(data=flat).writeto(tmp_path / "flat.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    run(Config(), tmp_path, listfile, flat_path=tmp_path / "flat.fits")

    out = fits.getdata(tmp_path / f"{objname}_NO1_sscf.fits")
    assert out[0, 1] == 0.0
    assert np.isfinite(out).all()


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """A flat with the wrong shape produces a clear error."""
    objname = "STAR"
    data = np.ones((50, 50), dtype=np.float32)
    flat = np.ones((10, 10), dtype=np.float32)
    _write_simple_frame(tmp_path / "raw1.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / "raw2.fits", data, object_name=objname)
    _write_simple_frame(tmp_path / f"{objname}_NO1_ssc.fits", data, object_name=objname)
    fits.PrimaryHDU(data=flat).writeto(tmp_path / "flat.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("raw1.fits raw2.fits\n")

    with pytest.raises(ValueError, match="shape mismatch"):
        run(Config(), tmp_path, listfile, flat_path=tmp_path / "flat.fits")


def test_missing_flat_path_raises(tmp_path: Path) -> None:
    """Calling without flat_path raises clearly."""
    listfile = tmp_path / "list.txt"
    listfile.write_text("a.fits b.fits\n")
    with pytest.raises(ValueError, match="flat_path"):
        run(Config(), tmp_path, listfile)
