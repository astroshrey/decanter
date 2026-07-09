"""Unit tests for :func:`decanter.io.apdb.parse`."""

from __future__ import annotations

from pathlib import Path

import pytest

from decanter._localpaths import WARP_ROOT
from decanter.io.apdb import (
    FUNCTION_TYPE_CHEBYSHEV,
    FUNCTION_TYPE_LEGENDRE,
    ApertureEntry,
    parse,
)


SAMPLE_BLOCK = """# Tue 13:12:07 10-Oct-2017
begin\taperture flat_HIRESY 159 197.116 1024.
\timage\tflat_HIRESY
\taperture\t159
\tbeam\t159
\tcenter\t197.116 1024.
\tlow\t-32. -1023.
\thigh\t30. 1025.
\tbackground
\t\txmin -32.
\t\txmax 30.
\t\tfunction chebyshev
\t\torder 1
\t\tsample -22:-12,8:18
\t\tnaverage -3
\t\tniterate 0
\t\tlow_reject 3.
\t\thigh_reject 3.
\t\tgrow 0.
\taxis\t1
\tcurve\t9
\t\t1.
\t\t5.
\t\t4.
\t\t2044.
\t\t-8.457813
\t\t82.94567
\t\t-8.489762
\t\t-0.01563291
\t\t-0.008452293
"""


def test_parse_single_aperture(tmp_path: Path) -> None:
    p = tmp_path / "apsample"
    p.write_text(SAMPLE_BLOCK)
    entries = parse(p)
    assert list(entries.keys()) == [159]
    ent = entries[159]
    assert ent.order == 159
    assert ent.center_x == 197.116
    assert ent.center_y == 1024.0
    assert ent.low == -32.0
    assert ent.high == 30.0
    assert ent.function_type == FUNCTION_TYPE_CHEBYSHEV
    assert ent.poly_order == 5
    assert ent.y_min == 4.0
    assert ent.y_max == 2044.0
    assert ent.coefficients == pytest.approx(
        (-8.457813, 82.94567, -8.489762, -0.01563291, -0.008452293)
    )
    assert ent.background_sample == "-22:-12,8:18"


def test_parse_multiple_apertures(tmp_path: Path) -> None:
    """Two aperture blocks in one file."""
    block2 = SAMPLE_BLOCK.replace("159", "160").replace("197.116", "278.316")
    p = tmp_path / "apdouble"
    p.write_text(SAMPLE_BLOCK + "\n" + block2)
    entries = parse(p)
    assert set(entries.keys()) == {159, 160}
    assert entries[159].center_x == 197.116
    assert entries[160].center_x == 278.316


def test_legendre_function_type(tmp_path: Path) -> None:
    """function_type==2 reads as Legendre."""
    leg_block = SAMPLE_BLOCK.replace("\t\t1.\n\t\t5.", "\t\t2.\n\t\t5.")
    p = tmp_path / "apleg"
    p.write_text(leg_block)
    ent = parse(p)[159]
    assert ent.function_type == FUNCTION_TYPE_LEGENDRE


def test_last_definition_wins(tmp_path: Path) -> None:
    """If two blocks share an order, the second overwrites the first."""
    block2 = SAMPLE_BLOCK.replace("197.116", "999.999")
    p = tmp_path / "apdup"
    p.write_text(SAMPLE_BLOCK + "\n" + block2)
    entries = parse(p)
    assert entries[159].center_x == 999.999


def test_indef_background_sample(tmp_path: Path) -> None:
    """sample=INDEF parses as the literal string."""
    indef_block = SAMPLE_BLOCK.replace("sample -22:-12,8:18", "sample INDEF")
    p = tmp_path / "apindef"
    p.write_text(indef_block)
    ent = parse(p)[159]
    assert ent.background_sample == "INDEF"


def test_real_warp_hiresy_flat() -> None:
    """Round-trip the real HIRES-Y flat aperture database from WARP."""
    real = WARP_ROOT / "reference/HIRES-Y/database/apflat_HIRESY_20170727_m"
    if not real.exists():
        pytest.skip(f"WARP reference data not present at {real}")
    entries = parse(real)
    # HIRES-Y nominally covers orders 159-184, but the flat DB also includes
    # adjacent orders for tracing context. Check we got a reasonable count.
    assert len(entries) >= 26
    # Each entry sanity check: positive center_x, sensible high, 5 coefficients.
    for order, ent in entries.items():
        assert 100 < order < 200, f"unexpected order {order}"
        assert 0 < ent.center_x < 2048
        assert ent.high > 0
        assert ent.low < 0
        assert len(ent.coefficients) == ent.poly_order
        assert ent.function_type in (FUNCTION_TYPE_CHEBYSHEV, FUNCTION_TYPE_LEGENDRE)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse(tmp_path / "does_not_exist")


def test_malformed_curve_raises(tmp_path: Path) -> None:
    """A truncated curve block fails loudly."""
    bad = SAMPLE_BLOCK.replace("\tcurve\t9", "\tcurve\t99")  # claims 99 but only has 9
    p = tmp_path / "apbad"
    p.write_text(bad)
    with pytest.raises(ValueError, match="expected 99 curve lines"):
        parse(p)
