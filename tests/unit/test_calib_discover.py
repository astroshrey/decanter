"""Tests for the Calibration dataclass + WARP input_files.txt discovery."""
from __future__ import annotations

from pathlib import Path

import pytest

from decanter.calib import Calibration
from decanter.calib.discover import _parse_input_files
from decanter._localpaths import WINERED_ROOT

# Live reduction reference; tests autoskip if absent so CI can run anywhere.
_TOI2109_REDUC = WINERED_ROOT / "reductions" / "TOI2109_decanterref"


def test_parse_input_files_minimal(tmp_path):
    """``_parse_input_files`` lowercases comments and trims whitespace."""
    p = tmp_path / "input_files.txt"
    p.write_text(
        "# Title row\n"
        "flat_HIRES-Y100_20250806.fits   # Flat file\n"
        "mask_flat_HIRES-Y100_20250806.fits  # Mask file\n"
        "comp_HIRES-Y100_20250806_fm_ecall.fits  # Comp file\n"
        "multihole_HIRES-Y100_20250806  # Ap file\n"
        "flat_HIRES-Y100_20250806_mscmn  # Ap file for apscatter\n"
        "multihole_HIRES-Y100_20250806  # Aptrans file\n"
    )
    fields = _parse_input_files(p)
    assert fields["flat file"] == "flat_HIRES-Y100_20250806.fits"
    assert fields["mask file"] == "mask_flat_HIRES-Y100_20250806.fits"
    assert fields["aptrans file"] == "multihole_HIRES-Y100_20250806"


def test_parse_input_files_skips_blank_and_commentless_lines(tmp_path):
    p = tmp_path / "input_files.txt"
    p.write_text(
        "# Title row\n"
        "\n"
        "no_comment_here_skipped\n"
        "value_x  # Comment X\n"
    )
    fields = _parse_input_files(p)
    assert fields == {"comment x": "value_x"}


@pytest.mark.skipif(
    not _TOI2109_REDUC.is_dir(),
    reason="TOI2109_decanterref reduction not available",
)
def test_from_dir_resolves_toi2109_decanterref():
    """``Calibration.from_dir`` on the live reference reduction returns
    paths to every calibration file plus the per-order trans aperture DBs."""
    calib = Calibration.from_dir(_TOI2109_REDUC)
    # Every primary path should exist on disk.
    for attr in ("flat", "static_bp_mask", "apdb_multihole", "apdb_apsc",
                 "comp", "fc_dir", "id_dir", "fsr_table"):
        p = getattr(calib, attr)
        assert p.exists(), f"{attr} ({p}) does not exist"
    # fc/id refnames are basename strings (no fc/id prefix, no .fits).
    assert not calib.fc_refname.startswith("fc")
    assert not calib.id_refname.startswith("id")
    assert not calib.id_refname.endswith(".fits")
    # Trans aperture DBs for the 26 echelle orders (159..184).
    assert calib.trans_apdbs is not None
    assert 163 in calib.trans_apdbs
    assert all(p.exists() for p in calib.trans_apdbs.values())


@pytest.mark.skipif(
    not _TOI2109_REDUC.is_dir(),
    reason="TOI2109_decanterref reduction not available",
)
def test_from_dir_is_frozen():
    """Calibration is frozen — mutating raises FrozenInstanceError."""
    import dataclasses
    calib = Calibration.from_dir(_TOI2109_REDUC)
    with pytest.raises(dataclasses.FrozenInstanceError):
        calib.flat = Path("/dev/null")
