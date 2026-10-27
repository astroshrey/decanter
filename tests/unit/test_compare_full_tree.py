"""decanter.compare_full_tree: every class of difference is caught and kept
apart (FITS data bit for bit in every HDU, header card text, raw bytes;
npz bit for bit), and only the documented exceptions pass."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from decanter.utils.iraf_header import IrafHeader, num_image, str_image
from decanter.warptree_io import _cards_bytes, iraf_fits_bytes

SCRIPT = Path(__file__).resolve().parents[2] / "decanter" / "compare_full_tree.py"
spec = importlib.util.spec_from_file_location("compare_full_tree", SCRIPT)
cft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cft)

R = np.float32
VAC = "X_NO1/onedspec/VAC_flux/fsr1.05/X_NO1_m163_fsr1.05_VAC.fits"


def _ua(stamp="2026-10-20T10:00:00"):
    return IrafHeader.from_images([
        str_image("ORIGIN", "NOAO-IRAF FITS Image Kernel July 2003"),
        str_image("IRAF-TLM", stamp), str_image("OBJECT", "X"),
        str_image("DATE", stamp), num_image("CRVAL1", "10800.5"), str_image("APNUM1", "1 1")])


def _iraf(data, *, prov=None, stamp="2026-10-20T10:00:00", ua=None):
    """WARP-like (prov None) or decanter-like IRAF bytes with fixed time stamps."""
    import decanter.warptree_io as wio
    old = wio.iraf_now
    wio.iraf_now = lambda: stamp
    try:
        return iraf_fits_bytes(np.asarray(data, R), ua or _ua(stamp), provenance=prov)
    finally:
        wio.iraf_now = old


def _run(tmp_path, files_w: dict, files_d: dict, *extra):
    W, D = tmp_path / "w", tmp_path / "d"
    for root, files in ((W, files_w), (D, files_d)):
        for rel, blob in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
    js = tmp_path / "out.json"
    rc = cft.main([str(W), str(D), "--json", str(js), *extra])
    return rc, json.loads(js.read_text())


def test_documented_exceptions_only(tmp_path):
    """Time stamps, DECANTER in the first blank card and the s13 tail value
    pass every class, bytes included."""
    x = np.linspace(1, 2, 300).astype(R)
    y = x.copy()
    y[-1] = R(7.0)                                       # WARP's heap garbage
    w = {"a/s.fits": _iraf(x), VAC: _iraf(y)}
    d = {"a/s.fits": _iraf(x, prov="decanter 0.0.1 mode=warp", stamp="2026-10-21T11:22:33"),
         VAC: _iraf(x, prov="decanter 0.0.1 mode=warp", stamp="2026-10-21T11:22:33")}
    rc, r = _run(tmp_path, w, d, "--bytes")
    assert rc == 0
    t = r["class_totals"]
    assert t["fits-data"] == {"identical": 1, "tail": 1}
    assert t["fits-header"] == {"identical": 2} and t["fits-bytes"] == {"identical": 2}
    assert r["tail_pixel_files"] == [VAC] and r["decanter_card_files"] == 2


@pytest.mark.parametrize("bits, note", [
    ((0x00000000, 0x80000000), "signed zeros"),          # +0.0 vs -0.0
    ((0x7FC00000, 0x7FC00001), "NaN payloads"),
    ((0x3F800000, 0x3F800001), "values")])               # 1 ULP
def test_data_compared_bit_for_bit(tmp_path, bits, note):
    a = np.ones(50, R)
    b = a.copy()
    a.view(np.uint32)[10], b.view(np.uint32)[10] = bits
    rc, r = _run(tmp_path, {"a/s.fits": _iraf(a)}, {"a/s.fits": _iraf(b)}, "--bytes")
    assert rc == 1
    t = r["class_totals"]
    assert t["fits-data"] == {"differ": 1}
    assert t["fits-header"] == {"identical": 1}
    assert t["fits-bytes"] == {"differ": 1}
    ex = next(v for k, v in r["examples"].items() if k.endswith("[fits-data]"))[0]
    assert "1/50 values" in ex[1]
    if note != "values":
        assert "equal as numbers" in ex[1]


def test_layout_is_its_own_class(tmp_path):
    """The same cards and data with END right after the last card and space
    padding (decanter's layout before 2026-10-21): data and card text
    identical, bytes different."""
    x = np.arange(10, dtype=R)
    good = _iraf(x)
    head_end = good.index(b"END     ")
    cards = [good[i:i + 80].decode() for i in range(0, head_end, 80)]
    cards = [c for c in cards if c.strip()]
    old = "".join(cards) + "END".ljust(80)
    old = (old + " " * ((-len(old)) % 2880)).encode()
    blob = old + good[len(good) - 2880:]
    rc, r = _run(tmp_path, {"a/s.fits": good}, {"a/s.fits": blob}, "--bytes")
    t = r["class_totals"]
    assert t["fits-data"] == {"identical": 1} and t["fits-header"] == {"identical": 1}
    assert t["fits-bytes"] == {"differ": 1} and rc == 1
    # without --bytes the layout is not checked
    rc, r = _run(tmp_path / "nb", {"a/s.fits": good}, {"a/s.fits": blob})
    assert rc == 0 and "fits-bytes" not in r["class_totals"]


def test_decanter_card_only_in_a_blank_slot(tmp_path):
    """DECANTER where WARP has a real card is a header and a byte difference."""
    x = np.arange(10, dtype=R)
    ua = _ua()
    ua_d = IrafHeader.from_images([c for c in ua.images() if not c.startswith("APNUM1")]
                                  + [str_image("DECANTER", "decanter 0.0.1")])
    rc, r = _run(tmp_path, {"a/s.fits": _iraf(x, ua=ua)}, {"a/s.fits": _iraf(x, ua=ua_d)},
                 "--bytes")
    t = r["class_totals"]
    assert t["fits-header"] == {"differ": 1} and t["fits-bytes"] == {"differ": 1}


def test_every_hdu(tmp_path):
    """Extension HDUs are compared too (data bits and cards)."""
    def two(v, key="A"):
        h = fits.HDUList([fits.PrimaryHDU(np.zeros(4, R)),
                          fits.ImageHDU(np.full(4, v, R), name=key)])
        import io
        buf = io.BytesIO()
        h.writeto(buf)
        return buf.getvalue()
    rc, r = _run(tmp_path, {"e.fits": two(1.0)}, {"e.fits": two(-1.0)})
    assert r["class_totals"]["fits-data"] == {"differ": 1}
    ex = next(v for k, v in r["examples"].items() if k.endswith("[fits-data]"))[0]
    assert ex[1].startswith("HDU 1:")
    rc, r = _run(tmp_path / "k", {"e.fits": two(1.0)}, {"e.fits": two(1.0, "B")})
    assert r["class_totals"]["fits-header"] == {"differ": 1}
    assert r["class_totals"]["fits-data"] == {"identical": 1}


def test_tail_only_for_the_last_value_of_a_vac_spectrum(tmp_path):
    x = np.linspace(1, 2, 100).astype(R)
    y = x.copy()
    y[-1] = R(5)
    z = x.copy()
    z[-2] = R(5)
    rc, r = _run(tmp_path, {VAC: _iraf(y), "a/other.fits": _iraf(y),
                            VAC.replace("NO1_m163", "NO1_m164"): _iraf(z)},
                 {VAC: _iraf(x), "a/other.fits": _iraf(x),
                  VAC.replace("NO1_m163", "NO1_m164"): _iraf(x)})
    assert r["tail_pixel_files"] == [VAC]
    assert r["class_totals"]["fits-data"] == {"tail": 1, "differ": 2}
    assert rc == 1


def test_npz_bitwise_and_missing_files(tmp_path):
    import io

    def npz(**arrs):
        buf = io.BytesIO()
        np.savez(buf, **arrs)
        return buf.getvalue()
    rc, r = _run(tmp_path, {"l.npz": npz(a=np.array([0.0, 1.0])), "only_warp.txt": b"x\n"},
                 {"l.npz": npz(a=np.array([-0.0, 1.0]))})
    assert r["class_totals"]["npz"] == {"differ": 1}
    assert r["totals"]["missing"] == 1 and rc == 1


def test_parser_reads_iraf_layout():
    """The comparator's FITS parser splits decanter's IRAF layout as written:
    END is the block's last card, the data follow it."""
    b = _iraf(np.arange(700, dtype=R), prov="decanter 0.0.1")
    (h,) = cft.parse_fits(b)
    assert h.hdr_len == 2880 and len(h.cards) == 36 and h.cards[-1].startswith(b"END")
    assert h.data_len == 2800 and h.pad_len == 80 and h.bitpix == -32
    assert _cards_bytes([num_image("NAXIS", "0")], "d")[-80:].startswith(b"END")


def _csw_ua(crval1: str, cdelt1: str):
    return IrafHeader.from_images([
        str_image("ORIGIN", "NOAO-IRAF FITS Image Kernel July 2003"),
        num_image("CRVAL1", crval1), num_image("CDELT1", cdelt1)])


def test_tail_needs_an_n_r_lt_n_cut_where_warp_kept_the_s12_spectrum(tmp_path):
    """A last-value-only difference is the s13 tail pixel only on an n_r < n cut:
    where WARP's tree kept the frame's s12 _csw, its WCS decides (TOI2109 m175
    fsr1.30: n 2339, n_r 2338; WIDE m42 fsr1.30: n = n_r = 1864, the s11 edge pixel
    of a +0.99 px shift before 2026-10-21); without it the tail is unverified."""
    s12 = "intermediate_files/OBJ/8A-OBJ-1DSPEC/4-OBJ-1DSPEC_dispcor"
    y_cut = "Y_NO1/onedspec/VAC_flux/fsr1.30/Y_NO1_m175_fsr1.30_VAC.fits"
    w_cut = "W_NO1/onedspec/VAC_flux/fsr1.30/W_NO1_m42_fsr1.30_VAC.fits"
    u_cut = "U_NO1/onedspec/VAC_flux/fsr1.30/U_NO1_m175_fsr1.30_VAC.fits"
    x = np.linspace(1, 2, 100).astype(R)
    y = x.copy()
    y[-1] = R(5)
    csw = np.zeros(4095, R)
    files_w = {y_cut: _iraf(y), w_cut: _iraf(y), u_cut: _iraf(y),
               f"Y_NO1/{s12}/Y_NO1_m175csw.fits": _iraf(csw, ua=_csw_ua("10016.836955",
                                                                       "0.0322009286761121")),
               f"W_NO1/{s12}/W_NO1_m42csw.fits": _iraf(csw[:2048], ua=_csw_ua("13103.909962",
                                                                            "0.22229416414265"))}
    files_d = {k: (v if "csw" in k else _iraf(x)) for k, v in files_w.items()}
    rc, r = _run(tmp_path, files_w, files_d)
    assert set(r["tail_pixel_files"]) == {y_cut, u_cut}
    assert r["tail_pixel_unverified"] == [u_cut]
    assert r["last_value_not_tail"] == [w_cut]
    assert r["class_totals"]["fits-data"] == {"tail": 2, "differ": 1, "identical": 2}
    assert rc == 1


# ---------------------------------------------------------------------------
# --bytes is strict: text / databases / npz byte for byte, only verified
# time-stamp fields masked in place
# ---------------------------------------------------------------------------

STATUS = "reduction_log/status.txt"
RUN_DB = "calibration_data/database/apX_NO1_sscfm_m163trans"
COPY_DB = "calibration_data/database/apflat_HIRES-Y100_20250806"
SNR = "SNR_dat/fsr1.05/SNratio_m163_fsr1.05.dat"


def _status(start="Thu Jul  2 14:41:21 2026", end="Thu Jul  2 14:43:12 2026",
            elapsed="1m51.2s", tail=""):
    return ("WARP ver.3.8.14\n"
            f"Starting time: {start}\n\nInput data:\n    WINA00053571 WINA00053572\n\n"
            f"Termination time: {end}\nElapsed time: {elapsed}\n\n"
            "Pipeline status: Successfully finished." + tail).encode()


def _db(stamp="Mon 22:11:18 19-Oct-2026"):
    return (f"# {stamp}\nbegin\taperture X 1 1024. 1024.\n\timage\tX\n"
            "\tlow\t-12.3 -1023.\n\thigh\t11.9 1024.\n").encode()


def _classes(r, *names):
    return {n: r["class_totals"].get(n) for n in names}


def test_text_time_fields_masked_in_place(tmp_path):
    """status.txt's ctime values and %dm%.1fs (any width), the run databases'
    # <cnvtime> lines and an SNR file's first-line Time: are masked in place."""
    w = {STATUS: _status(), RUN_DB: _db(), "calibration_data/database/aplast": _db(),
         SNR: b"Time: 2026-10-19 23:11:44.708627\n\nInput: \t\n"}
    d = {STATUS: _status("Wed Oct 21 19:57:57 2026", "Wed Oct 21 19:58:48 2026", "0m51.6s"),
         RUN_DB: _db("Thu 02:58:40 22-Oct-2026"),
         "calibration_data/database/aplast": _db("Wed 19:58:46 21-Oct-2026"),
         SNR: b"Time: 2026-10-21 20:00:14\n\nInput: \t\n"}       # str(datetime), microsecond 0
    rc, r = _run(tmp_path, w, d, "--bytes")
    assert rc == 0
    assert _classes(r, "text", "text-bytes", "apdb", "apdb-bytes") == {
        "text": {"identical": 2}, "text-bytes": {"identical": 2},
        "apdb": {"identical": 2}, "apdb-bytes": {"identical": 2}}
    assert r["masked_differences"] == {
        "SNratio .dat Time": 1, "database # <cnvtime>": 2, "status.txt Elapsed time": 1,
        "status.txt Starting time": 1, "status.txt Termination time": 1}


@pytest.mark.parametrize("d_status, what", [
    (_status().replace(b"Elapsed time: 1m51.2s\n", b""), "line dropped"),
    (_status(start="soon"), "not a time stamp"),
    (_status(start="Thu Jul  2 14:41:21 2026 (local)"), "text after the stamp"),
    (_status().replace(b"Termination time:", b"Termination  time:"), "label"),
    (_status(start="Thu Jul 02 14:41:21 2026"), "ctime pads the day with a space")])
def test_text_only_verified_fields_masked(tmp_path, d_status, what):
    """Lines are never dropped: a missing time line, a value that is not the
    time stamp WARP writes, or any other change on the line is a difference
    (the old tool dropped every Starting/Termination/Elapsed/Time: line)."""
    rc, r = _run(tmp_path, {STATUS: _status()}, {STATUS: d_status}, "--bytes")
    assert rc == 1, what
    assert _classes(r, "text", "text-bytes") == {"text": {"differ": 1},
                                                 "text-bytes": {"differ": 1}}, what


def test_text_terminators_and_final_newline_count(tmp_path):
    """Line terminators and the final newline are bytes: the line class still
    matches, text-bytes does not (WARP's status.txt ends without a newline)."""
    w = {STATUS: _status(), "reduction_log/aperture_log.txt": b"a\nb\n"}
    d = {STATUS: _status(tail="\n"), "reduction_log/aperture_log.txt": b"a\r\nb\n"}
    rc, r = _run(tmp_path, w, d, "--bytes")
    assert rc == 1
    assert _classes(r, "text", "text-bytes") == {"text": {"identical": 2},
                                                 "text-bytes": {"differ": 2}}
    notes = {e[0]: e[1] for k, v in r["examples"].items() if k.endswith("[text-bytes]") for e in v}
    assert notes[STATUS].startswith("final newline")
    assert "terminator" in notes["reduction_log/aperture_log.txt"]
    rc, r = _run(tmp_path / "nb", w, d)                   # without --bytes: lines only
    assert rc == 0 and "text-bytes" not in r["class_totals"]


def test_snr_time_masked_on_line_one_only(tmp_path):
    w = {SNR: b"Time: 2026-10-19 23:11:44.708627\n\nTime: 2026-10-19 23:11:44.708627\n"}
    d = {SNR: b"Time: 2026-10-21 20:00:14.613624\n\nTime: 2026-10-21 20:00:14.613624\n"}
    rc, r = _run(tmp_path, w, d, "--bytes")
    assert rc == 1 and r["class_totals"]["text-bytes"] == {"differ": 1}
    ex = next(v for k, v in r["examples"].items() if k.endswith("[text-bytes]"))[0]
    assert ex[1].startswith("line 3:")


def test_apdb_stamps_only_in_the_runs_databases(tmp_path):
    """A calibration-set database is a copy: its time stamps must be WARP's."""
    w = {RUN_DB: _db(), COPY_DB: _db()}
    d = {RUN_DB: _db("Thu 02:58:40 22-Oct-2026"), COPY_DB: _db("Thu 02:58:40 22-Oct-2026")}
    rc, r = _run(tmp_path, w, d, "--bytes")
    assert rc == 1
    assert _classes(r, "apdb", "apdb-bytes") == {"apdb": {"identical": 1, "differ": 1},
                                                 "apdb-bytes": {"identical": 1, "differ": 1}}
    bad = {e[0] for k, v in r["examples"].items() if k.endswith("[apdb-bytes]") for e in v}
    assert bad == {COPY_DB}
    # a stamp line must be exactly "# <cnvtime>"
    rc, r = _run(tmp_path / "x", {RUN_DB: _db()},
                 {RUN_DB: _db("Thu 02:58:40 22-Oct-2026 UTC")}, "--bytes")
    assert rc == 1 and r["class_totals"]["apdb-bytes"] == {"differ": 1}


def test_mask_text_keeps_everything_but_the_field():
    m, f = cft.mask_text(STATUS, _status(elapsed="12m3.4s", tail="\n"))
    assert m == _status(start="<time>", end="<time>", elapsed="<time>", tail="\n")
    assert [x[1] for x in f] == ["status.txt Starting time", "status.txt Termination time",
                                 "status.txt Elapsed time"]
    assert cft.mask_text("reduction_log/other.txt", _status()) == (_status(), [])
    # a literal mask is not a masked field
    rc_m, _ = cft.mask_text(STATUS, _status(start="<time>"))
    assert rc_m == _status(start="<time>", end="<time>", elapsed="<time>")


def test_literal_mask_text_never_passes(tmp_path):
    """A file that contains the mask text itself is not a masked time stamp."""
    rc, r = _run(tmp_path, {STATUS: _status()}, {STATUS: _status(start="<time>")}, "--bytes")
    assert rc == 1
    assert _classes(r, "text", "text-bytes") == {"text": {"differ": 1},
                                                 "text-bytes": {"differ": 1}}


# --- npz: zip containers member by member -----------------------------------------------

def _npy(arr, version=None):
    import io
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(arr), version=version)
    return buf.getvalue()


def _npz(members, *, date=(1980, 1, 1, 0, 0, 0), compress=False):
    """A zip like np.savez's: (name, .npy bytes) in order."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, blob in members:
            zi = zipfile.ZipInfo(name, date_time=date)
            zi.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
            with z.open(zi, "w", force_zip64=True) as f:
                f.write(blob)
    return buf.getvalue()


A = np.array([0.5, -1.0, 2.25])
B = np.arange(4, dtype=np.int64)


@pytest.mark.parametrize("d_npz, arrays_equal, why", [
    (_npz([("a.npy", _npy(A)), ("b.npy", _npy(B))], date=(2026, 10, 21, 19, 58, 40)),
     True, None),                                                      # zip time stamps only
    (_npz([("b.npy", _npy(B)), ("a.npy", _npy(A))]), True, "members"),  # order
    (_npz([("a.npy", _npy(A)), ("b.npy", _npy(B))], compress=True), True, "metadata"),
    (_npz([("a.npy", _npy(A, version=(2, 0))), ("b.npy", _npy(B))]), True, "header"),
    (_npz([("a.npy", _npy(A)), ("c.npy", _npy(B))]), False, "members"),
    (_npz([("a.npy", _npy(np.array([0.5, -1.0, 2.5]))), ("b.npy", _npy(B))]), False, "data")])
def test_npz_container_member_by_member(tmp_path, d_npz, arrays_equal, why):
    w = {"reduction_log/x_log.npz": _npz([("a.npy", _npy(A)), ("b.npy", _npy(B))])}
    rc, r = _run(tmp_path, w, {"reduction_log/x_log.npz": d_npz}, "--bytes")
    t = r["class_totals"]
    assert t["npz"] == ({"identical": 1} if arrays_equal else {"differ": 1})
    if why is None:
        assert rc == 0 and t["npz-bytes"] == {"identical": 1}
        assert r["masked_differences"] == {"zip time stamps": 1}
        return
    assert rc == 1 and t["npz-bytes"] == {"differ": 1}
    ex = next(v for k, v in r["examples"].items() if k.endswith("[npz-bytes]"))[0]
    assert why in ex[1]


def test_npz_of_numpy_savez_parsed():
    """np.savez's own files: every time field found (2 per member), member names
    in order; masking them leaves every other byte."""
    import io
    buf = io.BytesIO()
    np.savez(buf, psfCenter=A, psfWidth=A * 2, echelleOrder=B)
    b = buf.getvalue()
    spans, names, local = cft.zip_time_fields(b)
    assert names == local == [b"psfCenter.npy", b"psfWidth.npy", b"echelleOrder.npy"]
    assert len(spans) == 6 and all(e - s == 4 for s, e in spans)
    m = cft._masked(b, spans)
    assert sum(x != y for x, y in zip(b, m, strict=True)) <= 6 * 4


# --- FITS: only cards that ARE time stamps ----------------------------------------------

def _ua2(*extra, stamp="2026-10-20T10:00:00", date=None):
    return IrafHeader.from_images([
        str_image("ORIGIN", "NOAO-IRAF FITS Image Kernel July 2003"),
        str_image("IRAF-TLM", stamp), str_image("OBJECT", "X"),
        str_image("DATE", date or stamp), *extra])


@pytest.mark.parametrize("w_extra, d_extra, same", [
    ([str_image("FIXPIX", "Oct 20 16:49 Bad pixel file is maskflat_X_NO1_s")],
     [str_image("FIXPIX", "Oct 21  9:05 Bad pixel file is maskflat_X_NO1_s")], True),
    ([str_image("FIXPIX", "Oct 20 16:49 Bad pixel file is maskflat_X_NO1_s")],
     [str_image("FIXPIX", "Oct 20 16:49 Bad pixel file is maskflat_X_NO2_s")], False),
    ([str_image("FIXPIX", "Oct 20 16:49 Bad pixel file is maskflat_X_NO1_s")],
     [str_image("FIXPIX", "2026-10-21  Bad pixel file is maskflat_X_NO1_s")], False),
    ([str_image("FIXPIX", "Jul  9  2025 Bad pixel file is m")],
     [str_image("FIXPIX", "Oct 21 19:58 Bad pixel file is m")], True)])  # cnvdate's year form
def test_fixpix_run_time_masked_only_as_cnvdate(tmp_path, w_extra, d_extra, same):
    x = np.arange(5, dtype=R)
    rc, r = _run(tmp_path, {"a/s.fits": _iraf(x, ua=_ua2(*w_extra))},
                 {"a/s.fits": _iraf(x, ua=_ua2(*d_extra, stamp="2026-10-21T19:58:41"))},
                 "--bytes")
    t = r["class_totals"]
    want = {"identical": 1} if same else {"differ": 1}
    assert t["fits-header"] == want and t["fits-bytes"] == want and rc == (0 if same else 1)


def _set_card(blob: bytes, key: str, image: str) -> bytes:
    """``blob`` with the first header card of ``key`` replaced by ``image``
    (the writer itself puts fresh IRAF-TLM / DATE values in)."""
    k = key.ljust(8).encode()
    for i in range(0, 2880 * 4, 80):
        if blob[i:i + 8] == k:
            return blob[:i] + image.ljust(80).encode() + blob[i + 80:]
    raise KeyError(key)


def test_date_cards_that_are_not_time_stamps_compared(tmp_path):
    """A raw frame's DATE (a date, no time) is not a write time: compared in full;
    so is an IRAF-TLM card whose value is not the kernel's ISO time."""
    x = np.arange(5, dtype=R)
    w, d = _iraf(x), _iraf(x, stamp="2026-10-21T19:58:41")
    raw = "DATE    = '2025-08-07'         /  Creation UTC (CCCC-MM-DD) date of FITS header"
    rc, r = _run(tmp_path, {"a/r.fits": _set_card(w, "DATE", raw),
                            "a/t.fits": w, "a/ok.fits": w},
                 {"a/r.fits": _set_card(d, "DATE", raw.replace("08-07", "08-08")),
                  "a/t.fits": _set_card(d, "IRAF-TLM", "IRAF-TLM= 'unknown time       '"),
                  "a/ok.fits": d}, "--bytes")
    t = r["class_totals"]
    assert t["fits-header"] == {"differ": 2, "identical": 1}
    assert t["fits-bytes"] == {"differ": 2, "identical": 1} and rc == 1
    assert r["masked_differences"] == {"fits DATE": 1, "fits IRAF-TLM": 1}


def test_literal_mask_in_a_fits_card_never_passes(tmp_path):
    x = np.arange(5, dtype=R)
    w = _iraf(x)
    d = _set_card(w, "IRAF-TLM", "IRAF-TLM= '<time>' / Time of last modification")
    rc, r = _run(tmp_path, {"a/s.fits": w}, {"a/s.fits": d}, "--bytes")
    t = r["class_totals"]
    assert t["fits-header"] == {"differ": 1} and t["fits-bytes"] == {"differ": 1} and rc == 1


def test_decanter_card_must_be_decanters(tmp_path):
    """Only a DECANTER= 'decanter ...' card passes in WARP's first blank card."""
    x = np.arange(10, dtype=R)
    rc, r = _run(tmp_path, {"a/s.fits": _iraf(x)}, {"a/s.fits": _iraf(x, prov="hello")},
                 "--bytes")
    t = r["class_totals"]
    assert t["fits-header"] == {"differ": 1} and t["fits-bytes"] == {"differ": 1} and rc == 1


def test_pdf_creation_date_masked_in_place(tmp_path):
    pdf = b"%PDF-1.4\n1 0 obj << /CreationDate (D:20260702144309-07'00') >> endobj\n%%EOF\n"
    later = pdf.replace(b"20260702144309", b"20261021195840")
    other = later.replace(b"1 0 obj", b"2 0 obj")
    shorter = pdf.replace(b"-07'00'", b"Z")
    rc, r = _run(tmp_path, {"a.pdf": pdf, "b.pdf": pdf, "c.pdf": pdf},
                 {"a.pdf": later, "b.pdf": other, "c.pdf": shorter}, "--bytes")
    assert r["class_totals"]["pdf"] == {"identical": 1, "differ": 2} and rc == 1
    assert r["masked_differences"] == {"pdf /CreationDate": 1}
