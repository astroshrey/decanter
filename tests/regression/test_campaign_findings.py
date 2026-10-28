"""Real-data regressions for the all-transit parity campaign findings
(qa/decanter_parity_all): one frame, the affected order, locked path (WARP's
own per-frame aperture databases), against the archived WARP VAC products.

- s02 factor map (Qatar6 NO9 m172, HD207496 NO35 m170).
- inverted aperture swapped by apall (TOI270c NO38 m163).
- APNUM1 text from the per-frame databases (every frame of the reductions
  whose headers decanter used to get wrong).
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import decanter
from decanter._localpaths import WINERED_ROOT
from decanter.config import Config

RED = WINERED_ROOT / "reductions"
CAL = WINERED_ROOT / "calibration_data"
RAW = WINERED_ROOT / "science_data"

CASES = [  # testname, frame, frame-dir prefix, order, night, calibration set
    ("Qatar6", 9, "Qatar6", 172, "2024_04_26", "calibration_LCO24a_setting7_HIRES-Y100"),
    ("HD207496", 35, "HD207496", 170, "2024_09_17", "calibration_LCO24b_setting6_HIRES-Y100"),
    ("TOI270c", 38, "TOI-270", 163, "2024_09_14", "calibration_LCO24b_setting5_HIRES-Y100"),
]


def _have(tn, night, cset):
    return (RED / tn).is_dir() and (RAW / night).is_dir() and (CAL / cset).is_dir()


@pytest.mark.regression
@pytest.mark.parametrize("mode", ["warp-clone"])
@pytest.mark.parametrize("tn,no,prefix,m,night,cset", CASES)
def test_frame_order_matches_archived_warp(tn, no, prefix, m, night, cset, mode):
    if not _have(tn, night, cset):
        pytest.skip(f"{tn} data not available")
    W = RED / tn
    line = [ln.split() for ln in (W / f"{tn}.txt").read_text().splitlines() if ln.strip()][no - 1]
    shift = float(np.load(W / "reduction_log/waveshift_log.npz")["waveShiftAdopted"][no - 1])
    cal = decanter.Calibration.from_dir(CAL / cset)
    db = W / "calibration_data/database" / f"ap{prefix}_NO{no}_sscfm_m{m}trans"
    cal = dataclasses.replace(cal, trans_apdbs={m: db})
    cfg = dataclasses.replace(
        Config.from_parameter_file(WINERED_ROOT / "parameter_files/nominal_params.txt"),
        reduce_full_data=False, selected_orders=(m,))
    r = decanter.reduce(RAW / night / f"{line[0]}.fits", cal, sky=RAW / night / f"{line[1]}.fits",
                        config=cfg, mode=mode, shift_wave=shift)
    fdir = W / f"{prefix}_NO{no}"
    for (cut, mm), sp in r.obj.items():
        want = fits.getdata(fdir / f"onedspec/VAC_flux/fsr{cut:.2f}/{prefix}_NO{no}_m{mm}_fsr{cut:.2f}_VAC.fits")
        assert np.count_nonzero(want) > 0
        assert np.array_equal(sp.flux, want), (cut, mm)
    for (cut, mm), sp in (r.sky or {}).items():
        want = fits.getdata(fdir / f"sky_emission/VAC/fsr{cut:.2f}/"
                            f"{prefix}_skyNO{no}_fm_m{mm}trans1dcutw_fsr{cut:.2f}_VAC.fits")
        assert np.array_equal(sp.flux, want), ("sky", cut, mm)


APNUM_REDUCTIONS = ["AUMic", "GJ3090_nominal", "GJ3090_nominal_nows", "HD136352", "HD15337",
                    "HD86226", "L9859_d", "LHS1140", "TOI1231", "TOI270_oct", "TOI270c",
                    "TOI712", "piMen"]


@pytest.mark.regression
@pytest.mark.parametrize("tn", APNUM_REDUCTIONS)
def test_apnum_from_frame_database_matches_warp(tn):
    """APNUM1 computed the way warptree writes it (from the per-frame database
    tokens) equals WARP's header on every frame and order (fsr1.05 products)."""
    from decanter.warptree import _apnum, _read_ref_aperture
    W = RED / tn
    if not W.is_dir():
        pytest.skip(f"{tn} not available")
    n = 0
    for f in sorted(W.glob("*_NO*/onedspec/VAC_flux/fsr1.05/*_VAC.fits")):
        mo = re.match(r"(.+)_NO(\d+)_m(\d+)_fsr", f.name)
        prefix, no, m = mo.group(1), int(mo.group(2)), int(mo.group(3))
        db = W / "calibration_data/database" / f"ap{prefix}_NO{no}_sscfm_m{m}trans"
        if not db.is_file():
            continue
        ref = _read_ref_aperture(db, m)        # same reader; low/high = this frame's DB
        assert _apnum(ref, ref.ap_low, ref.ap_high) == fits.getheader(f)["APNUM1"], f.name
        n += 1
    assert n > 0
