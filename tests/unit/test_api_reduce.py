"""Tests for the top-level decanter.reduce() / combine() API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import decanter
from decanter._localpaths import WINERED_ROOT, raw_night_dir

# Live data for the end-to-end parity test (skipped when not present).
_RAW_NIGHT = raw_night_dir("20250807", "2025_08_07")
_RAW_OBJ = _RAW_NIGHT / "WINA00053571.fits"
_RAW_SKY = _RAW_NIGHT / "WINA00053572.fits"
_TOI_PYWARPREF = WINERED_ROOT / "reductions" / "TOI2109_decanterref"
_NO1 = _TOI_PYWARPREF / "TOI2109_NO1"
_WARP_M163_FSR105 = (
    _NO1 / "onedspec" / "VAC_flux" / "fsr1.05"
    / "TOI2109_NO1_m163_fsr1.05_VAC.fits"
)

_LIVE_DATA_OK = (
    _RAW_OBJ.is_file()
    and _RAW_SKY.is_file()
    and _TOI_PYWARPREF.is_dir()
    and _WARP_M163_FSR105.is_file()
)


def test_combine_rejects_empty() -> None:
    """combine() needs at least one reduction."""
    with pytest.raises(ValueError, match="at least one"):
        decanter.combine([])


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_reduce_many_aligns_and_combines() -> None:
    """reduce_many() aligns frames and combine() stacks them onto one grid.

    Reducing the same pair twice gives a 2-frame series with zero relative
    shift; the combined master matches the per-frame spectra."""
    calib = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    series = decanter.reduce_many([(_RAW_OBJ, _RAW_SKY), (_RAW_OBJ, _RAW_SKY)], calib)
    assert len(series.reductions) == 2
    assert series.shifts.shape == (2,)
    assert abs(float(series.shifts[series.refid])) < 1e-9      # reference shift is 0
    # Identical frames -> shift at the fine-search grid floor (~0.01 px), not
    # necessarily bit-zero.
    cdelt = series.reductions[0].obj[(1.05, 163)].cdelt1
    assert np.all(np.abs(series.shifts) < 0.05 * cdelt)        # < 0.05 px
    master = decanter.combine(series, cut=1.05)
    a = master.obj[(1.05, 163)].flux.astype(float)
    b = series.reductions[0].obj[(1.05, 163)].flux.astype(float)
    assert a.shape == b.shape
    an, bn = a / np.median(a), b / np.median(b)
    assert np.median(np.abs(an - bn)) < 5e-3                   # master ~ per-frame spectrum


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_reduce_matches_warp_on_toi2109_decanterref_m163_fsr105() -> None:
    """End-to-end: decanter.reduce() output matches WARP-saved _fsr1.05_VAC
    for m163 within the s06+s08 algorithmic noise floor (medrel < 0.01%,
    same shape, same CRVAL1)."""
    calib = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    r = decanter.reduce(_RAW_OBJ, calib, sky=_RAW_SKY)
    spec = r.obj[(1.05, 163)]

    warp_data = fits.getdata(_WARP_M163_FSR105).astype(np.float64)
    warp_hdr = fits.getheader(_WARP_M163_FSR105)

    assert spec.flux.shape == warp_data.shape, (
        f"shape mismatch: decanter {spec.flux.shape} vs WARP {warp_data.shape}"
    )
    assert abs(spec.crval1 - float(warp_hdr["CRVAL1"])) < 1e-3, (
        f"CRVAL1 mismatch: decanter {spec.crval1} vs WARP {warp_hdr['CRVAL1']}"
    )
    diff = spec.flux.astype(np.float64) - warp_data
    medrel = float(np.median(np.abs(diff) / np.maximum(np.abs(warp_data), 1.0)))
    assert medrel < 1e-4, f"medrel too high: {medrel * 100:.4f}%"


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_reduce_returns_intermediates_when_requested(tmp_path: Path) -> None:
    """save_intermediates=True populates Reduction.intermediates with the
    per-stage 2D / per-order arrays."""
    calib = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    r = decanter.reduce(
        _RAW_OBJ, calib, sky=_RAW_SKY,
        workdir=tmp_path, save_intermediates=True,
    )
    # 2D intermediates exist:
    assert r.intermediates.obj_s is not None
    assert r.intermediates.obj_sscfm is not None
    assert r.intermediates.cr_mask is not None
    # Per-order intermediates exist:
    assert 163 in r.intermediates.strips_obj
    assert 163 in r.intermediates.spectra_1d
    # And the final files made it to disk:
    out = tmp_path / "TOI2109_NO1_sscfm_m163_fsr1.05_VAC.fits"
    assert out.exists()


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_reduce_orders_property_lists_26_orders() -> None:
    """Default Config processes all 26 orders 159..184."""
    calib = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    r = decanter.reduce(_RAW_OBJ, calib, sky=_RAW_SKY)
    assert r.orders == tuple(range(159, 185))
    assert r.fsr_cuts == (1.05, 1.3)


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_standalone_trace_matches_warp_locked() -> None:
    """The WARP-independent extraction (multihole trans reference + center
    search, used when no per-frame trans DB is present) reproduces the
    WARP-locked path to the float32 parity floor across all orders.

    This guards the standalone trace/aperture: a WARP reduction root ships
    both the per-frame trans DBs (locked path) and the multihole trans
    references (standalone path), so dropping the former forces the standalone
    solve on the very same frame and calibration.
    """
    import dataclasses

    calib_locked = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    assert calib_locked.trans_apdbs, "fixture should carry per-frame trans DBs"
    assert calib_locked.ref_trans_apdbs, "fixture should carry multihole trans refs"
    calib_standalone = dataclasses.replace(calib_locked, trans_apdbs=None)

    r_locked = decanter.reduce(_RAW_OBJ, calib_locked, sky=_RAW_SKY)
    r_standalone = decanter.reduce(_RAW_OBJ, calib_standalone, sky=_RAW_SKY)

    worst = 0.0
    for key, spec in r_locked.obj.items():
        a = r_standalone.obj[key].flux.astype(np.float64)
        b = spec.flux.astype(np.float64)
        medrel = float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
        worst = max(worst, medrel)
    assert worst < 5e-3, f"standalone vs WARP-locked medrel too high: {worst * 100:.3f}%"


@pytest.mark.skipif(not _LIVE_DATA_OK, reason="live WINERED data not available")
def test_reduce_a_only_no_sky() -> None:
    """Reducing a single nod position (sky=None) works end to end.

    Guards the A-only path: with no sky frame the nod pattern must still be
    resolved for the center-search aperture solve (regression for a NameError
    on ``nodpos`` when the CR-mask branch was skipped)."""
    calib = decanter.Calibration.from_dir(_TOI_PYWARPREF)
    r = decanter.reduce(_RAW_OBJ, calib)                       # no sky
    assert (1.05, 163) in r.obj and r.obj[(1.05, 163)].flux.size > 0
    r_bg = decanter.reduce(_RAW_OBJ, calib, subtract_background=True)
    assert (1.05, 163) in r_bg.obj and r_bg.obj[(1.05, 163)].flux.size > 0


def test_reduce_rejects_unimplemented_mode() -> None:
    """Only mode='warp' exists today; other recipes raise before any I/O."""
    import pytest

    import decanter
    with pytest.raises(ValueError, match="not implemented"):
        decanter.reduce("obj.fits", None, mode="default")
