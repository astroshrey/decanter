"""Unit tests for :mod:`decanter.image2d.sky_subtract`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.config import Config
from decanter.stages import s01_sky_subtract


def _make_frame(path: Path, value: float, **header_kwargs: object) -> None:
    hdr = fits.Header()
    for k, v in header_kwargs.items():
        hdr[k] = v
    data = np.full((10, 10), value, dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_simple_subtraction(tmp_path: Path) -> None:
    _make_frame(tmp_path / "obj.fits", 100.0, OBJECT="MY_TARGET", EXPTIME=300.0)
    _make_frame(tmp_path / "sky.fits", 5.0, OBJECT="SKY")
    listfile = tmp_path / "list.txt"
    listfile.write_text("obj sky\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    out = tmp_path / "MY_TARGET_NO1_s.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert np.allclose(hdul[0].data, 95.0)
        assert hdul[0].header["OBJECT"] == "MY_TARGET"
        assert hdul[0].header["EXPTIME"] == 300.0


def test_multiple_pairs_get_unique_indices(tmp_path: Path) -> None:
    _make_frame(tmp_path / "a.fits", 100.0, OBJECT="TGT")
    _make_frame(tmp_path / "b.fits", 5.0, OBJECT="TGT")
    _make_frame(tmp_path / "c.fits", 200.0, OBJECT="TGT")
    _make_frame(tmp_path / "d.fits", 10.0, OBJECT="TGT")
    listfile = tmp_path / "list.txt"
    listfile.write_text("a b\nc d\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    out1 = tmp_path / "TGT_NO1_s.fits"
    out2 = tmp_path / "TGT_NO2_s.fits"
    assert out1.exists()
    assert out2.exists()
    with fits.open(out1) as hdul:
        assert np.allclose(hdul[0].data, 95.0)
    with fits.open(out2) as hdul:
        assert np.allclose(hdul[0].data, 190.0)


def test_objname_is_sanitized(tmp_path: Path) -> None:
    _make_frame(tmp_path / "a.fits", 1.0, OBJECT="Star Name/With Bad'Chars")
    _make_frame(tmp_path / "b.fits", 0.0, OBJECT="SKY")
    listfile = tmp_path / "list.txt"
    listfile.write_text("a b\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    assert (tmp_path / "Star_Name_With_Bad_Chars_NO1_s.fits").exists()


def test_falls_back_to_frame_name_when_object_header_missing(tmp_path: Path) -> None:
    """If the OBJECT header is absent, the output filename uses the raw frame name."""
    _make_frame(tmp_path / "WINA001.fits", 50.0)  # no OBJECT key
    _make_frame(tmp_path / "WINA002.fits", 1.0)
    listfile = tmp_path / "list.txt"
    listfile.write_text("WINA001 WINA002\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    # s01 passes `default=pair.object_name` to headers.get, so the
    # missing-OBJECT path yields a filename keyed on the raw frame name.
    assert (tmp_path / "WINA001_NO1_s.fits").exists()


def test_subtraction_is_bit_identical_to_numpy(tmp_path: Path) -> None:
    """The pipeline output equals the in-memory subtraction byte for byte."""
    rng = np.random.default_rng(seed=42)
    obj = rng.normal(loc=1000.0, scale=50.0, size=(16, 16)).astype(np.float32)
    sky = rng.normal(loc=100.0, scale=10.0, size=(16, 16)).astype(np.float32)
    fits.PrimaryHDU(data=obj, header=fits.Header({"OBJECT": "T"})).writeto(tmp_path / "obj.fits")
    fits.PrimaryHDU(data=sky, header=fits.Header({"OBJECT": "SKY"})).writeto(tmp_path / "sky.fits")
    listfile = tmp_path / "list.txt"
    listfile.write_text("obj sky\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    with fits.open(tmp_path / "T_NO1_s.fits") as hdul:
        assert np.array_equal(hdul[0].data, obj - sky)


def test_run_is_pure_with_respect_to_inputs(tmp_path: Path) -> None:
    """Running the stage must not touch the input FITS files."""
    _make_frame(tmp_path / "obj.fits", 100.0, OBJECT="T")
    _make_frame(tmp_path / "sky.fits", 5.0)
    obj_before = (tmp_path / "obj.fits").read_bytes()
    sky_before = (tmp_path / "sky.fits").read_bytes()
    listfile = tmp_path / "list.txt"
    listfile.write_text("obj sky\n")

    s01_sky_subtract.run(Config(), workdir=tmp_path, listfile=listfile)

    assert (tmp_path / "obj.fits").read_bytes() == obj_before
    assert (tmp_path / "sky.fits").read_bytes() == sky_before
