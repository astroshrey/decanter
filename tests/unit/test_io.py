"""Unit tests for the ``decanter.io`` subpackage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from decanter.io import fits as pwfits
from decanter.io import headers
from decanter.io.listfile import FramePair, parse as parse_listfile


# ---- decanter.io.fits --------------------------------------------------------


def test_read_write_image_roundtrip(tmp_path: Path) -> None:
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    src = tmp_path / "src.fits"
    hdr = astropy_fits.Header()
    hdr["OBJECT"] = "TARGET"
    hdr["EXPTIME"] = 42.0
    astropy_fits.PrimaryHDU(data=data, header=hdr).writeto(src)

    rd, rh = pwfits.read_image(src)
    assert np.array_equal(rd, data)
    assert rh["OBJECT"] == "TARGET"
    assert rh["EXPTIME"] == 42.0

    out = tmp_path / "out.fits"
    pwfits.write_image(out, rd * 2.0, rh)
    rd2, rh2 = pwfits.read_image(out)
    assert np.array_equal(rd2, data * 2.0)
    assert rh2["OBJECT"] == "TARGET"


def test_read_accepts_path_without_extension(tmp_path: Path) -> None:
    """WARP passes bare names; decanter must accept both forms."""
    data = np.zeros((3, 3), dtype=np.float32)
    astropy_fits.PrimaryHDU(data=data).writeto(tmp_path / "bare.fits")
    rd, _ = pwfits.read_image(tmp_path / "bare")  # no .fits
    assert rd.shape == (3, 3)


def test_write_does_not_add_checksum(tmp_path: Path) -> None:
    """astropy's ``add_checksum`` injects CHECKSUM/DATASUM which bust parity diffs."""
    data = np.ones((4, 4), dtype=np.float32)
    out = tmp_path / "out.fits"
    pwfits.write_image(out, data)
    _, hdr = pwfits.read_image(out)
    assert "CHECKSUM" not in hdr
    assert "DATASUM" not in hdr


def test_write_refuses_overwrite_by_default(tmp_path: Path) -> None:
    out = tmp_path / "exists.fits"
    pwfits.write_image(out, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(OSError):
        pwfits.write_image(out, np.ones((2, 2), dtype=np.float32))


# ---- decanter.io.headers -----------------------------------------------------


def test_headers_get_present_and_missing() -> None:
    hdr = astropy_fits.Header()
    hdr["A"] = 1
    assert headers.get(hdr, "A") == 1
    assert headers.get(hdr, "MISSING") == "N/A"
    assert headers.get(hdr, "MISSING", default="custom") == "custom"


def test_headers_get_on_plain_dict() -> None:
    assert headers.get({"X": 7}, "X") == 7
    assert headers.get({"X": 7}, "Y") == "N/A"


# ---- decanter.io.listfile ----------------------------------------------------


def test_listfile_parse_basic(tmp_path: Path) -> None:
    p = tmp_path / "list.txt"
    p.write_text("WINA001 WINA002\nWINA003 WINA004\n")
    pairs = parse_listfile(p)
    assert pairs == [
        FramePair(object_name="WINA001", sky_name="WINA002"),
        FramePair(object_name="WINA003", sky_name="WINA004"),
    ]


def test_listfile_parse_with_optional_tokens(tmp_path: Path) -> None:
    p = tmp_path / "list.txt"
    p.write_text("obj sky ap=-7:3 bg=-22:-12,8:18 ws=0.5\n")
    [pair] = parse_listfile(p)
    assert pair.aperture_low == -7.0
    assert pair.aperture_high == 3.0
    assert pair.background_region == "-22:-12,8:18"
    assert pair.manual_shift == 0.5


def test_listfile_skips_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "list.txt"
    p.write_text("# header comment\nWINA001 WINA002\n\n# another\nWINA003 WINA004\n")
    pairs = parse_listfile(p)
    assert len(pairs) == 2


def test_listfile_rejects_short_lines(tmp_path: Path) -> None:
    p = tmp_path / "list.txt"
    p.write_text("WINA001\n")
    with pytest.raises(ValueError, match="at least 'OBJECT SKY'"):
        parse_listfile(p)


def test_iraf_id_parses_last_of_duplicate_records(tmp_path):
    """IRAF databases are append-only; a file can hold multiple `begin`
    records for the same aperture (LCO26a HIRES-J archive sets do). IRAF's
    dt_locate keeps the LAST matching record, so parse() must too — and
    must not let the second record's `features` table leak into the
    coefficient tokens of the first."""
    from decanter.io import iraf_id

    rec = """begin\tidentify comp.0131
\timage\tcomp.0131
\tunits\tAngstroms
\tfeatures\t2
\t\t100.0\t13500.0\t13500.0\t4.0\t1\t1
\t\t200.0\t13400.0\t13400.0\t4.0\t1\t1
\tcoefficients\t6
\t\t1.
\t\t2.
\t\t1.000000
\t\t4095.000000
\t\t{c0}
\t\t{c1}
"""
    f = tmp_path / "idcomp.0131"
    f.write_text(rec.format(c0="13000.0", c1="-80.0")
                 + rec.format(c0="13478.9", c1="-87.5"))
    sol = iraf_id.parse(f)
    assert sol.order == 2
    assert sol.coefficients[0] == 13478.9  # last record wins
