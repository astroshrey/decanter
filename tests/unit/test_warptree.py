"""WARP-layout writer: IRAF header text, aperture databases, logs, listfiles.

Expected strings are copied from WARP products on disk (TOI2109, 2025-08-07,
calibration_LCO25b_setting2_HIRES-Y100) and from the compiled IRAF ``dtoc3`` /
``ctod`` (``scripts/iraf_oracle``), so these tests pin decanter to WARP's
bytes without needing the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.config import Config
from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iraf_fmt import dtoc3, imputd_text
from decanter.wavelength.fsr import fsr_bounds
from decanter.wavelength.iraf_wcs import dispcor_wcs, pyraf_real_text, scopy_wcs
from decanter.waveshift.measure import (
    warp_reference_frame, warp_waveshift_clip, warp_waveshift_log)
from decanter.warptree import (
    _apnum, _base_header, _frame_db_stem, _frame_trans_dbs, _HeaderContext, _Pool,
    _read_ref_aperture, _snr_place, _wcs_cards, _write, obj_suffixes, read_warp_listfile,
    render_frame_apdb, short_name, warp_objname, write_aperture_log)

# --- IRAF number <-> text --------------------------------------------------


@pytest.mark.parametrize("value, text", [
    (0.03446124719101115, "0.0344612471910112"),   # IRAF agrees with %.15g here
    (9792.74565200001, "9792.74565200001"),
    (1.0, "1."),
    (-817.859217704501, "-817.859217704501"),
    (0.5, "0.5"),
])
def test_imputd_text(value, text):
    assert imputd_text(value) == text


def test_dtoc3_14_digits_and_fixed():
    assert dtoc3(0.03446124719101115, 14) == "0.034461247191011"
    assert dtoc3(9792.74565200001, 14) == "9792.745652"
    assert dtoc3(168.7099914550781, 2, "f") == "168.71"


def test_ctod_seventeen_digits_is_fused():
    # value from the compiled IRAF ctod (ctod.o fuses value*10 + digit)
    assert iraf_ctod("0.46703113640550675") == 0.46703113640550753
    assert iraf_ctod("1.000000") == 1.0000000000000007   # ctod is not exact


def test_pyraf_real_text_is_12_digits():
    assert pyraf_real_text(10783.029999999999) == "10783.03"
    assert pyraf_real_text(1.0) == "1.0"


# --- header WCS: dispcor (_csw) and scopy FSR cut (_fsr*_VAC) ----------------

def test_dispcor_wcs_goes_through_the_14_digit_multispec_string():
    dc = dispcor_wcs(10754.808714000006, 0.03446124719101115, 4095)
    assert dc.crval1 == "10754.808714"
    assert dc.cdelt1 == "0.0344612471910111"      # WARP's _csw CDELT1


def test_scopy_wcs_matches_warp_toi2109_m163():
    lo, hi = fsr_bounds(10792.90, 10858.70, 1.3)
    sc = scopy_wcs("10754.808714", "0.0344612471910111", 4095, lo, hi)
    assert sc.n == 2483
    assert sc.crval1 == "10754.8089798823"
    assert sc.crpix1 == "-817.859217704501"
    assert sc.cdelt1 == "0.0344641581177711"
    assert sc.ltv1 == "-818.859301701174"
    assert sc.ltm1_1 == "1.00008399667268"
    assert sc.np2 is None


# --- headers on disk ----------------------------------------------------------

def _raw_header():
    h = fits.Header()
    h["OBJECT"] = "TOI 2109/b"
    h["DATE-OBS"] = "2025-08-07"
    h["UT-STR"] = "23:29:51.3"
    h["UT"] = "23:32:21.25"
    h["EXPTIME"] = 300.0
    h["NODPOS"] = "A1"
    h["NODOFS1"] = 0.0
    h["RA"] = "16:52:44.71"
    h["DEC"] = "+16:34:52.4"
    return h


def test_written_header_keeps_raw_keywords_and_iraf_wcs_text(tmp_path):
    ref = _RefApertureFixture.make(tmp_path)
    ctx = _HeaderContext(flat="flat_X_mscmn.fits", apsc="flat_X", comp_stem="comp_X_fm_ecall",
                         fixpix_time="Jul  3 12:31")
    cards = {"CRVAL1": "10754.8089798823", "CRPIX1": "-817.859217704501",
             "CDELT1": "0.0344641581177711", "CD1_1": "0.0344641581177711",
             "LTV1": "-818.859301701174", "LTM1_1": "1.00008399667268"}
    h = _base_header(_raw_header())
    _wcs_cards(h, 163, cards, _apnum(ref, -27.23, -10.36), ctx, ncombine=True)
    out = tmp_path / "x.fits"
    _write(out, np.arange(2483, dtype=np.float32), h)
    with fits.open(out) as hd:
        g = hd[0].header
        img = {c.keyword: c.image[10:30].strip() for c in g.cards}
        for k, v in cards.items():
            assert img[k] == v
        for k in ("DATE-OBS", "UT-STR", "UT", "EXPTIME", "NODPOS", "NODOFS1", "RA", "DEC"):
            assert g[k] == _raw_header()[k]
        assert g["PIPELINE"] == "3.8.14"
        assert g["APNUM1"] == "163 163 168.71 185.58"
        assert g["EXTEND"] is False
        # the wavelength grid downstream code builds is WARP's, to the bit
        wav = g["CDELT1"] * (np.arange(2483) - g["CRPIX1"]) + g["CRVAL1"]
        ref_wav = 0.0344641581177711 * (np.arange(2483) - -817.859217704501) + 10754.8089798823
        assert np.array_equal(wav, ref_wav)


# --- per-frame aperture databases ---------------------------------------------

_REF_DB = """# Fri 11:36:03 08-Aug-2025
begin\taperture\tmultihole_HIRES-Y100_20250806_163trans 163 195.94 2814.00
\timage\tmultihole_HIRES-Y100_20250806_163trans
\taperture\t163
\tbeam\t163
\tcenter\t195.94 2814.000000
\tlow\t-28.00 -2813.000000
\thigh\t28.00 2815.000000
\tbackground
\t\txmin -28.00
\t\txmax 28.00
\t\tfunction chebyshev
\t\torder 1
\t\tsample -10:-5,5:10
\t\tnaverage -3
\t\tniterate 0
\t\tlow_reject 3.
\t\thigh_reject 3.
\t\tgrow 0.
\taxis 1
\tcurve 11
\t\t1.000000
\t\t7.000000
\t\t993.000000
\t\t5593.000000
\t\t10.836971
\t\t97.680911
\t\t-10.698080
\t\t-0.062737
\t\t-0.381021
\t\t-0.041655
\t\t-0.084493

"""

# WARP's reductions/TOI2109/calibration_data/database/apTOI2109_NO10_sscfm_m163trans
_WARP_FRAME_DB = _REF_DB.replace("# Fri 11:36:03 08-Aug-2025", "# Fri 22:10:09 03-Jul-2026") \
    .replace("multihole_HIRES-Y100_20250806_163trans", "TOI2109_NO10_sscfm_m163trans") \
    .replace("\tlow\t-28.00", "\tlow\t6.45").replace("\thigh\t28.00", "\thigh\t21.85") \
    .replace("xmin -28.00", "xmin 6.45").replace("xmax 28.00", "xmax 21.85") \
    .replace("sample -10:-5,5:10", "sample INDEF")


class _RefApertureFixture:
    @staticmethod
    def make(tmp_path: Path):
        p = tmp_path / "apmultihole_163trans"
        p.write_text(_REF_DB)
        return _read_ref_aperture(p, 163)


def test_frame_apdb_matches_warp_bytes(tmp_path):
    ref = _RefApertureFixture.make(tmp_path)
    got = render_frame_apdb(ref, "TOI2109_NO10_sscfm_m163trans", 6.45, 21.85,
                            timestamp="# Fri 22:10:09 03-Jul-2026")
    assert got == _WARP_FRAME_DB


def test_frame_apdb_name_uses_warp_shortfile_for_long_objects(tmp_path):
    # WARP's shortFile: OBJECT names longer than 13 characters become "OBJECT"
    # in work names, and write_apdatabase gets the work name (Warp_sci.py:553).
    assert _frame_db_stem("TOI2109", 3) == "TOI2109_NO3_sscfm"
    assert _frame_db_stem("A" * 13, 3) == "A" * 13 + "_NO3_sscfm"      # 13: kept
    assert _frame_db_stem("HD_189733_b_tr", 3) == "OBJECT_NO3_sscfm"   # 14: short
    ref = _RefApertureFixture.make(tmp_path)
    img = f"{_frame_db_stem('HD_189733_b_tr', 3)}_m163trans"
    txt = render_frame_apdb(ref, img, 6.45, 21.85, timestamp="# x")
    assert "begin\taperture\tOBJECT_NO3_sscfm_m163trans 163 " in txt
    assert "\timage\tOBJECT_NO3_sscfm_m163trans\n" in txt
    db = tmp_path / "root" / "calibration_data" / "database"
    db.mkdir(parents=True)
    (db / "apOBJECT_NO3_sscfm_m163trans").write_text(txt)
    (db / "apOBJECT_NO13_sscfm_m163trans").write_text(txt)
    assert _frame_trans_dbs(tmp_path / "root", "HD_189733_b_tr", 3) == {
        163: db / "apOBJECT_NO3_sscfm_m163trans"}
    assert _frame_trans_dbs(tmp_path / "root", "TOI2109", 3) == {}


def test_object_names_without_apscatter_are_sf_sfm(tmp_path):
    # Warp_sci.py:284-297: obj_ssc_list = obj_s_list when Apscatter is off,
    # so the flat/fixpix frames, strips and per-frame DBs are _sf/_sfm.
    assert obj_suffixes(True) == ("sscf", "sscfm")
    assert obj_suffixes(False) == ("sf", "sfm")
    assert _frame_db_stem("TOI2109", 1, False) == "TOI2109_NO1_sfm"
    assert _frame_db_stem("HD_189733_b_tr", 2, False) == "OBJECT_NO2_sfm"
    db = tmp_path / "root" / "calibration_data" / "database"
    db.mkdir(parents=True)
    (db / "apTOI2109_NO1_sfm_m163trans").write_text("x")
    (db / "apTOI2109_NO1_sfm_m164trans").write_text("x")
    want = {163: db / "apTOI2109_NO1_sfm_m163trans", 164: db / "apTOI2109_NO1_sfm_m164trans"}
    assert _frame_trans_dbs(tmp_path / "root", "TOI2109", 1, apscatter=False) == want
    # a reduction with the other setting is still found (locking to it)
    assert _frame_trans_dbs(tmp_path / "root", "TOI2109", 1) == want
    (db / "apTOI2109_NO1_sscfm_m163trans").write_text("x")
    assert _frame_trans_dbs(tmp_path / "root", "TOI2109", 1) == {
        163: db / "apTOI2109_NO1_sscfm_m163trans"}
    assert _frame_trans_dbs(tmp_path / "root", "TOI2109", 1, apscatter=False) == want


def test_snr_products_placement_follows_frame_number_limit(tmp_path):
    # Warp_sci.py:736-741, 884-890: SNR files for 1 < n <= 28 frames, moved to
    # SNR_dat/fsr<cut>/ only for n < 28 (left in the tree root at n = 28).
    assert _snr_place(tmp_path, "1.05", 1) is None
    assert _snr_place(tmp_path, "1.05", 2) == tmp_path / "SNR_dat" / "fsr1.05"
    assert _snr_place(tmp_path, "1.30", 27) == tmp_path / "SNR_dat" / "fsr1.30"
    assert _snr_place(tmp_path, "1.05", 28) == tmp_path
    assert _snr_place(tmp_path, "1.05", 29) is None


def _boom():
    raise RuntimeError("x")


def test_inline_pool_runs_in_order_and_reraises():
    pool = _Pool(1)
    assert list(pool.map(abs, [-1, 2, -3])) == [1, 2, 3]
    assert pool.submit(pow, 2, 5).result() == 32
    fut = pool.submit(_boom)
    with pytest.raises(RuntimeError):
        fut.result()
    pool.close()


# --- listfiles, names, numbering ----------------------------------------------

def test_listfile_every_line_is_a_frame(tmp_path):
    lf = tmp_path / "x.txt"
    lf.write_text("WINA00053571 WINA00053572\nWINA00053572 WINA00053571 ws=0.12\n\n")
    lines = read_warp_listfile(lf)
    assert [(ln.obj, ln.sky) for ln in lines] == [("WINA00053571", "WINA00053572"),
                                                   ("WINA00053572", "WINA00053571")]
    assert lines[1].ws == pytest.approx(0.12)


def test_objname_sanitized_like_warp():
    h = fits.Header()
    h["OBJECT"] = "TOI 2109/b's #1"
    assert warp_objname(h) == "TOI_2109_b_s__1"
    assert warp_objname(fits.Header()) == "N_A"
    assert short_name("A_VERY_LONG_NAME_X_NO3", "A_VERY_LONG_NAME_X") == "OBJECT_NO3"
    assert short_name("TOI2109_NO3", "TOI2109") == "TOI2109_NO3"


# --- logs ------------------------------------------------------------------------

def test_aperture_log_format(tmp_path):
    low = np.array([[-27.234, 1.0]])
    high = np.array([[-10.355, 2.0]])
    write_aperture_log(tmp_path / "a.txt", tmp_path / "a.npz", low, high, (163, 164))
    txt = (tmp_path / "a.txt").read_text()
    assert "### Log of Apertures ####" in txt
    assert "m=163\t-27.23:-10.36\t\t\n" in txt
    z = np.load(tmp_path / "a.npz")
    assert set(z.files) == {"apertureLow", "apertureUpp", "echelleOrder", "frameNum"}


def test_waveshift_clip_and_log_keys(tmp_path):
    rng = np.random.default_rng(0)
    nframe, norder = 5, 6
    mat = [[0.0] + list(rng.normal(0.3, 0.02, nframe - 1)) for _ in range(norder)]
    mat[2][3] = 3.0                                   # an outlier order
    ave, num, std, flag = warp_waveshift_clip(mat, nframe, norder)
    assert ave[0] == 0.0 and std[0] == 0.0 and num[0] == norder and flag[0] == 0
    assert num[3] == norder - 1
    # single-frame / no-measure path: zeros, WARP's npz keys
    spec = [[np.ones(4095, np.float32)] for _ in range(norder)]
    log = warp_waveshift_log(spec, tuple(range(159, 159 + norder)), 0.5)
    log.save_npz(tmp_path / "w.npz")
    z = np.load(tmp_path / "w.npz")
    assert set(z.files) == {"waveShift", "waveShiftAve", "waveShiftStd", "waveShiftNum",
                            "waveShiftAdopted", "echelleOrder", "frameNum"}
    assert z["waveShift"].shape == (norder, 1)


def test_reference_frame_is_float32_argmax():
    a = np.full(10, 1.0, np.float32)
    rows = [[a, a * 3, a * 2], [a, a * 3, a * 2]]
    assert warp_reference_frame(rows) == 1
    assert warp_reference_frame([[a, a * 3], [a, a * 3]]) == 0   # <= 2 frames -> 0


# --- parameter files ------------------------------------------------------------

def test_parameter_file_like_warp(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("Apscatter: yes\nCosmic Ray Correction: no\nSet cut range: 1.05, 1.3\n"
                 "Sky Emission: no \nCorrect Shift: no\nSelected Orders: 163\n")
    c = Config.from_parameter_file(p)
    assert c.flag_bpmask is False and c.flag_skyemission is False
    assert c.flag_wscorrect is False and c.cutrange_list == (1.05, 1.3)
    assert c.selected_orders == (163,)     # 'Selected Orders' (capital O) is ignored by WARP


# --- provenance --------------------------------------------------------------

def test_provenance_records_implementation_and_backend(tmp_path):
    """decanter_provenance.json names the implementation and backend (numpy / mlx <ver>)."""
    import json

    import decanter
    from decanter._localpaths import WINERED_ROOT, raw_night_dir
    from decanter.fast._backend import backend
    raw = raw_night_dir("20250807", "2025_08_07")
    cal = WINERED_ROOT / "calibration_data" / "calibration_LCO25b_setting2_HIRES-Y100"
    if not (raw / "WINA00053571.fits").is_file() or not cal.is_dir():
        pytest.skip("TOI2109 raw frames / calibration set not available")
    cfg = Config(selected_orders=(163,), reduce_full_data=False)
    with backend("numpy"):
        decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                     tmp_path / "t", config=cfg,            # default mode
                                     intermediates=False, log=lambda s: None)
    prov = json.loads((tmp_path / "t" / "reduction_log" / "decanter_provenance.json").read_text())
    assert prov["mode"] == "warp" and prov["implementation"] == "vectorised"
    assert prov["fast_backend"] == "numpy"
    decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                 tmp_path / "r", mode="warp-reference", config=cfg,
                                 intermediates=False, log=lambda s: None)
    prov = json.loads((tmp_path / "r" / "reduction_log" / "decanter_provenance.json").read_text())
    assert prov["mode"] == "warp-reference" and prov["implementation"] == "reference"
    assert prov["fast_backend"] is None
    from astropy.io import fits
    for t, m in (("t", "warp"), ("r", "warp-reference")):
        f = next((tmp_path / t).glob("*_NO1/onedspec/VAC_flux/fsr1.05/*_m163_fsr1.05_VAC.fits"))
        assert fits.getheader(f)["DECANTER"].endswith(f"mode={m}")


# --- IRAF's run-time stamps in a tree (both implementations) --------------------

def _card(path, key: str) -> str | None:
    b = Path(path).read_bytes()
    for i in range(0, len(b) - 79, 80):
        c = b[i:i + 80]
        if c[:8] == key.ljust(8).encode():
            return c.decode().rstrip()
        if c[:8] == b"END     ":
            return None
    return None


def test_tree_stamps_are_irafs_at_a_single_digit_hour(tmp_path, monkeypatch):
    """FIXPIX's run time and the databases IRAF writes carry IRAF's cnvdate / cnvtime of
    IRAF's clock, in the vectorised and the reference implementation.  At 2026-10-22
    01:03:47 PDT IRAF's clock reads 00:03:47 (standard time): FIXPIX 'Oct 22  0:03'
    (decanter f49c5d4 wrote 'Oct 22 01:03'), databases '# Thu 00:03:47 22-Oct-2026';
    WARP's own aperture.py stamp stays UTC."""
    import os
    import re
    import time as _time

    import decanter
    from decanter._localpaths import WINERED_ROOT, raw_night_dir
    from decanter.fast._backend import backend
    from decanter.utils import iraf_time
    raw = raw_night_dir("20250807", "2025_08_07")
    cal = WINERED_ROOT / "calibration_data" / "calibration_LCO25b_setting2_HIRES-Y100"
    if not (raw / "WINA00053571.fits").is_file() or not cal.is_dir():
        pytest.skip("TOI2109 raw frames / calibration set not available")
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    _time.tzset()
    gmt = 1792656227                                   # 2026-10-22T08:03:47Z = 01:03:47 PDT
    monkeypatch.setattr(iraf_time, "_now", lambda: gmt + 0.5)
    cfg = Config(selected_orders=(163,), reduce_full_data=False)
    try:
        with backend("numpy"):
            decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                         tmp_path / "t", config=cfg, intermediates=True,
                                         rawdata=False, plots=False, log=lambda s: None)
        decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                     tmp_path / "r", mode="warp-reference", config=cfg,
                                     intermediates=True, rawdata=False, plots=False,
                                     log=lambda s: None)
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        _time.tzset()
    fixpix = "FIXPIX  = 'Oct 22  0:03 Bad pixel file is maskflat_TOI2109_NO1_s'"
    for t in ("t", "r"):
        root = tmp_path / t
        files = [root / "TOI2109_NO1/intermediate_files/OBJ/4-OBJ_mask/TOI2109_NO1_sscfm.fits",
                 *sorted(root.glob("TOI2109_NO1/onedspec/*/fsr*/*_m163_*.fits")),
                 *sorted(root.glob("TOI2109_NO1/sky_emission/*/fsr*/*.fits")),
                 *sorted(root.glob("TOI2109_NO1/intermediate_files/OBJ/8A-OBJ-1DSPEC/*/*.fits")),
                 *sorted(root.glob("TOI2109_sum/*/fsr*/*.fits"))]
        assert len(files) > 20
        for f in files:
            assert _card(f, "FIXPIX") == fixpix, f
        db = root / "calibration_data/database"
        for name in ("apTOI2109_NO1_s", "apTOI2109_skyNO1_fm_m163trans", "aplast"):
            assert (db / name).read_text().splitlines()[0] == "# Thu 00:03:47 22-Oct-2026", name
        # WARP's own (Python, UTC) stamp of the per-frame trans database is not IRAF's
        assert re.fullmatch(r"# \w{3} \d\d:\d\d:\d\d \d\d-\w{3}-\d{4}",
                            (db / "apTOI2109_NO1_sscfm_m163trans").read_text().splitlines()[0])


# --- locked trees: WARP's logged aperture window ------------------------------

def test_locked_aperture_log_takes_warps_logged_window(tmp_path):
    """With apertures locked to a WARP tree, aperture_log.npz carries WARP's
    own logged (unrounded) window where its %.2f text is the database token;
    otherwise the database value, and the note says so."""
    from decanter.warptree import warp_logged_window
    rl = tmp_path / "reduction_log"
    rl.mkdir()
    wl = np.array([[-17.234123, -16.995001], [10.4449, 10.5551]])
    wh = np.array([[-0.355999, -0.4012], [18.9049, 19.1]])
    np.savez(rl / "aperture_log.npz", apertureLow=wl, apertureUpp=wh,
             echelleOrder=np.array([163, 164]), frameNum=2)
    low = np.array([[-17.23, -17.00], [10.44, 10.56]])        # %.2f tokens
    high = np.array([[-0.36, -0.40], [18.90, 19.10]])
    out_l, out_h, note = warp_logged_window(tmp_path, low, high, (163, 164))
    assert np.array_equal(out_l, wl) and np.array_equal(out_h, wh)
    assert "4 of 4" in note
    low2 = low.copy()
    low2[1, 0] = 10.47                                       # a token WARP did not write
    out_l, out_h, note = warp_logged_window(tmp_path, low2, high, (163, 164))
    assert out_l[1, 0] == 10.47 and out_l[0, 0] == wl[0, 0] and "3 of 4" in note
    out_l, _, note = warp_logged_window(tmp_path, low, high, (163, 165))
    assert np.array_equal(out_l, low) and "orders/frames" in note
    out_l, _, note = warp_logged_window(tmp_path / "nowhere", low, high, (163, 164))
    assert np.array_equal(out_l, low) and "not found" in note


@pytest.mark.skipif(not (Path.home() / "Disk/winered/reductions/TOI3486_decanterref"
                         / "reduction_log/aperture_log.npz").is_file(),
                    reason="TOI3486_decanterref not available")
def test_locked_aperture_log_reproduces_warps_npz():
    """The per-frame database tokens of a WARP tree select WARP's whole log."""
    import re
    from decanter.warptree import warp_logged_window
    root = Path.home() / "Disk/winered/reductions/TOI3486_decanterref"
    z = np.load(root / "reduction_log/aperture_log.npz")
    orders = [int(o) for o in z["echelleOrder"]]
    low = np.zeros(z["apertureLow"].shape)
    high = np.zeros(z["apertureUpp"].shape)
    for i in range(low.shape[0]):
        for j, m in enumerate(orders):
            t = (root / "calibration_data/database" /
                 f"apTOI-3486_NO{i + 1}_sscfm_m{m}trans").read_text()
            low[i, j] = float(re.search(r"\tlow\t(\S+)", t).group(1))
            high[i, j] = float(re.search(r"\thigh\t(\S+)", t).group(1))
    out_l, out_h, note = warp_logged_window(root, low, high, orders)
    assert np.array_equal(out_l, z["apertureLow"]) and np.array_equal(out_h, z["apertureUpp"])
    assert f"{low.size} of {low.size}" in note
    assert not np.array_equal(low, z["apertureLow"])          # the tokens are rounded


# --- calibration-set entries WARP leaves in the tree root ----------------------

def _fake_calib_set(root: Path) -> Path:
    import os
    cs = root / "calibration_X_setting5_HIRES-Y100"
    (cs / "database").mkdir(parents=True)
    (cs / "database" / "apflat_X").write_text("db\n")
    (cs / "input_files.txt").write_text(
        "WINERED_pipeline\nflat_X_mscmn.fits\t#flat file\nmask_flat_X.fits\t#mask file\n"
        "comp_X_fm_ecall.fits\t#comp file\nmultihole_X\t#ap file\nmultihole_X\t#aptrans file\n"
        "flat_X\t#ap file for apscatter\n")
    for f in ("flat_X_mscmn.fits", "mask_flat_X.fits", "comp_X_fm_ecall.fits",
              "calibration_parameters.txt", "readme.txt"):
        (cs / f).write_bytes(b"x")
    (cs / "sys").write_bytes(b"")                    # calibration_LCO25a_setting5's stray
    os.chmod(cs / "sys", 0o770)
    (cs / "notes.txt").write_text("hello\n")
    (cs / "old").mkdir()
    (cs / "old" / "flat_old.fits").write_bytes(b"old")
    (cs / ".swp").write_bytes(b"vim")                # a dot file: glob("*") skips it
    (cs / "dangling").symlink_to(cs / "nowhere")      # neither a file nor a directory
    (cs / "TOI.txt").write_text("listfile of the same name\n")
    (cs / "WINA1.fits").write_bytes(b"raw of the same name")
    (cs / "status.txt").write_text("status\n")
    return cs


def test_calib_set_strays_are_the_entries_warp_leaves_in_the_root(tmp_path):
    """Warp_sci.py:202-209 copies every non-dot entry not yet in the work directory;
    Warp_sci.py:1099-1108 moves only input_files.txt's flat/mask/comp, database,
    input_files.txt, calibration_parameters.txt and readme.txt to calibration_data/."""
    import os
    import stat

    from decanter.warptree import _copy_calibration_strays, calib_set_strays
    cs = _fake_calib_set(tmp_path)
    strays = calib_set_strays(cs, skip=["TOI.txt", "status.txt", "WINA1.fits", "WINA2.fits"])
    assert [p.name for p in strays] == ["notes.txt", "old", "sys"]
    dest = tmp_path / "tree"
    dest.mkdir()
    assert _copy_calibration_strays(strays, dest) == ["notes.txt", "old", "sys"]
    assert (dest / "sys").read_bytes() == b""
    assert stat.S_IMODE(os.stat(dest / "sys").st_mode) == 0o770       # shutil.copy keeps the mode
    assert (dest / "notes.txt").read_text() == "hello\n"
    assert (dest / "old" / "flat_old.fits").read_bytes() == b"old"
    assert calib_set_strays(None) == []


def test_reduce_to_warp_tree_copies_the_strays_like_warp(tmp_path):
    """A calibration set's other entries end up in the tree root (WARP's ``sys``);
    an entry named like a WARP-tree product is refused."""
    import json
    import os

    import decanter
    from decanter._localpaths import WINERED_ROOT, raw_night_dir
    from decanter.fast._backend import backend
    raw = raw_night_dir("20250807", "2025_08_07")
    src = WINERED_ROOT / "calibration_data" / "calibration_LCO25b_setting2_HIRES-Y100"
    if not (raw / "WINA00053571.fits").is_file() or not src.is_dir():
        pytest.skip("TOI2109 raw frames / calibration set not available")
    cal = tmp_path / src.name
    cal.mkdir()
    for p in src.iterdir():
        if p.is_dir():
            os.symlink(p, cal / p.name)
        else:
            os.symlink(p, cal / p.name)
    (cal / "input_files.txt").unlink()
    (cal / "input_files.txt").write_text((src / "input_files.txt").read_text())
    (cal / "sys").write_bytes(b"")
    (cal / "extra").mkdir()
    (cal / "extra" / "note.txt").write_text("n\n")
    (cal / ".hidden").write_text("h\n")
    cfg = Config(selected_orders=(163,), reduce_full_data=False)
    with backend("numpy"):
        decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                     tmp_path / "t", config=cfg, intermediates=False,
                                     rawdata=False, plots=False, log=lambda s: None)
    root = {p.name for p in (tmp_path / "t").iterdir()}
    assert {"sys", "extra"} <= root and ".hidden" not in root
    assert (tmp_path / "t" / "sys").read_bytes() == b""
    assert (tmp_path / "t" / "extra" / "note.txt").read_text() == "n\n"
    prov = json.loads((tmp_path / "t" / "reduction_log" / "decanter_provenance.json").read_text())
    assert prov["calibration_strays"] == ["extra", "sys"]
    (cal / "reduction_log").mkdir()
    with pytest.raises(ValueError, match="reduction_log"):
        decanter.reduce_to_warp_tree([("WINA00053571", "WINA00053572")], raw, cal,
                                     tmp_path / "u", config=cfg, intermediates=False,
                                     rawdata=False, plots=False, log=lambda s: None)
