"""Tests for the Calibration dataclass + WARP input_files.txt discovery."""
from __future__ import annotations

from pathlib import Path

import pytest

from decanter.calib import Calibration, LockedAperturesWarning
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
    with pytest.warns(LockedAperturesWarning, match="EVERY frame on NO1"):
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


def test_from_dir_calibration_set_does_not_warn():
    """A calibration-set directory has no per-frame DBs: no lock, no warning."""
    import warnings
    from decanter._localpaths import CAL_DATA_ROOTS
    sets = [p for r in CAL_DATA_ROOTS for p in sorted(r.glob("calibration_*"))
            if (p / "input_files.txt").is_file()]
    if not sets:
        pytest.skip("no calibration set on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LockedAperturesWarning)
        calib = Calibration.from_dir(sets[0])
    assert calib.trans_apdbs is None


def _fake_set(d: Path, flat_bytes: bytes = b"flat") -> Path:
    d.mkdir(parents=True)
    (d / "input_files.txt").write_text(
        "WINERED_pipeline\nflat.fits #flat file\nmask.fits #mask file\n"
        "comp.fits #comp file\nmh #ap file\nmh #aptrans file\nfl #ap file for apscatter\n")
    (d / "flat.fits").write_bytes(flat_bytes)
    (d / "mask.fits").write_bytes(b"mask")
    (d / "comp.fits").write_bytes(b"comp")
    (d / "database").mkdir()
    (d / "database" / "apmh").write_text("x")
    return d


def test_from_dir_own_set_wins_and_fallback_warns(tmp_path, monkeypatch):
    """Files next to input_files.txt are used silently; a file missing there is
    taken from another set with a loud CalibrationFallbackWarning that names
    differing same-name files; extra_cal_roots makes the choice explicit."""
    import warnings
    from decanter.calib import discover
    from decanter.calib import CalibrationFallbackWarning
    pub = tmp_path / "pub"
    a = _fake_set(pub / "calibration_A", b"flat-A")
    _fake_set(pub / "calibration_B", b"flat-B")
    monkeypatch.setattr(discover, "_CAL_DATA_ROOTS", (pub,))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationFallbackWarning)
        assert Calibration.from_dir(a).flat == a / "flat.fits"
    red = tmp_path / "red" / "calibration_data"
    _fake_set(red)
    (red / "flat.fits").unlink()
    with pytest.warns(CalibrationFallbackWarning, match="DIFFERENT files of the same name"):
        c = Calibration.from_dir(red.parent)
    assert c.flat.parent.parent == pub
    assert c.static_bp_mask == red / "mask.fits" and c.comp == red / "comp.fits"
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationFallbackWarning)
        c2 = Calibration.from_dir(red.parent, extra_cal_roots=(pub / "calibration_B",))
    assert c2.flat == pub / "calibration_B" / "flat.fits"


def test_from_dir_finds_frame_dbs_of_an_apscatter_off_reduction(tmp_path):
    """A WARP root reduced with Apscatter: no names its per-frame DBs _sfm_
    (Warp_sci.py:284-297); from_dir finds those like the _sscfm_ ones."""
    red = tmp_path / "red" / "calibration_data"
    _fake_set(red)
    db = red / "database"
    (db / "apX_NO1_sfm_m163trans").write_text("x")
    (db / "apX_NO1_sfm_m164trans").write_text("x")
    (db / "apX_NO10_sfm_m163trans").write_text("x")
    with pytest.warns(LockedAperturesWarning):
        c = Calibration.from_dir(red.parent)
    assert c.trans_apdbs == {163: db / "apX_NO1_sfm_m163trans", 164: db / "apX_NO1_sfm_m164trans"}
