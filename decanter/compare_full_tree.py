"""Compare EVERY file of a WARP reduction tree with a decanter tree.

Walks the WARP tree and, for each file, looks for the same relative path in
the decanter tree.  Each kind of comparison is a class of its own, counted
and reported separately:

* ``fits-data``: every HDU's data unit compared BIT for bit (the raw
  big-endian words, so ``-0.0`` vs ``+0.0`` and NaN payloads count as
  differences), and the number of HDUs.  A per-frame FSR-cut ``VAC``
  spectrum whose only difference is its last value (header identical) is
  the documented s13 tail pixel (WARP's uninitialised memory when
  ``n_r < n``, docs/BITEXACT.md), counted as ``tail`` and not as a
  difference -- unless the WARP tree kept that frame's s12 spectrum
  (``4-OBJ-1DSPEC_dispcor/*_m<m>csw``, ``7-SKY_dispcor/*_m<m>trans1dcutw``)
  and its header WCS says the cut is not ``n_r < n`` (``scopy_wcs`` with
  the FSR table's bounds): then it is a ``fits-data`` difference ("last
  value only, not an n_r < n cut"; the JSON lists it under
  ``last_value_not_tail``).  Tails the WARP tree cannot confirm are listed
  under ``tail_pixel_unverified``.
* ``fits-header``: every HDU's non-blank card images, text for text, in
  order.  Excluded: decanter's provenance card ``DECANTER`` and the value of
  a time stamp -- only where the card IS one (``TIME STAMPS`` below; any
  other ``DATE`` / ``IRAF-TLM`` / ``FIXPIX`` card, e.g. a raw frame's
  ``DATE = '2025-08-07'``, is compared in full).
* ``fits-bytes`` (``--bytes``): the whole file byte for byte -- header
  layout (blank cards, ``END`` position, block count), data, padding --
  with exactly these exceptions: the time-stamp field of a verified time
  stamp (same card, same slot, both files), decanter's ``DECANTER`` card
  (``DECANTER= 'decanter ...'``) where WARP has its first blank card after
  the last card (decanter's only intended difference,
  ``decanter.warptree_io``), and the accepted tail pixel's four bytes.
* ``apdb``: aperture databases (``calibration_data/database/*``) as text,
  line for line; ``apdb-bytes`` (``--bytes``): byte for byte (line
  terminators and the final newline count).  Both with the time-stamp field
  of the ``# <cnvtime>`` lines masked IN PLACE (the ``# `` and the line stay)
  -- only in the databases the run writes (per-frame ``*_NO<i>_*`` /
  ``*_skyNO<i>_*`` and ``aplast``); the calibration set's copies must be
  identical as they are.
* ``npz``: every array, element for element (floats bit for bit), by name;
  ``npz-bytes`` (``--bytes``): the zip container member by member -- the
  member names in order, each member's ``.npy`` bytes, and the whole file
  byte for byte (local headers, central directory, compression, CRCs,
  attributes) with only the zip time stamps (DOS time/date of each local
  header and central-directory entry, extended-timestamp/NTFS extra fields)
  masked.
* ``text``: other text (logs, ``.dat``, listfile, ``status.txt``) line for
  line; ``text-bytes`` (``--bytes``): byte for byte.  Masked in place, and
  only there: ``reduction_log/status.txt`` ``Starting time: <ctime>``,
  ``Termination time: <ctime>``, ``Elapsed time: <%dm%.1fs>``; the first
  line ``Time: <datetime>`` of an ``SNratio_m*_fsr*.dat``.
* ``png``: byte for byte; ``pdf``: byte for byte with the ``/CreationDate``
  value masked in place (same length).

TIME STAMPS (each field is masked only when it has exactly the format WARP or
IRAF writes there, in both files, on the same card / line; everything else on
the card / line must be identical): FITS ``IRAF-TLM= 'YYYY-MM-DDThh:mm:ss'``
and ``DATE    = 'YYYY-MM-DDThh:mm:ss'`` (the FITS kernel's write time,
fxfupdhdr.x), the ``cnvdate`` text (``%3.3s %2d %2d:%02d`` or ``%3.3s %2d
%5d``, sys/etc/cnvdate.x) at the start of ``FIXPIX  = '... Bad pixel file is
...'`` (t_fixpix.x:124-125); database lines ``# <cnvtime>`` (``%3.3s
%02d:%02d:%02d %02d-%3.3s-%04d``, xtools/dttext.x:dtptime; WARP's
``time.strftime("# %a %H:%M:%S %d-%b-%Y")``, aperture.py:154);
``status.txt``'s ``time.ctime()`` values and ``%dm%.1fs`` (config.py:417-428,
Warp_sci.py:109-110, 1137-1139); ``"Time: %s" % datetime.datetime.today()``
(SNratio_estimate.py:169).  These are every time-like field of the text files
of 123 WARP trees (59 reductions, 64 fresh WARP 3.8.14 runs; census
qa/decanter_parity_all/probes/text_timestamp_census.py) that the run itself
writes; the IRAF task logs ``reduction_log/{logfile,null}`` hold more but are
not written by decanter.  In the 2,096,086 FITS files of those trees every
``IRAF-TLM`` and ``FIXPIX`` card, and every ``DATE`` card but the raw frames'
date-only ones in ``rawdata_image/``, has the masked format
(probes/fits_timestamp_census.py).  With ``--bytes`` the JSON counts every masked field
whose two values differ (``masked_differences``).

Prints a per-class summary, a per-pattern table, every WARP file missing from
the decanter tree (grouped by pattern), and decanter files WARP does not
have.  Exit status 0 = no difference in any class and nothing missing.

    python -m decanter.compare_full_tree WARP_ROOT DECANTER_ROOT [--bytes]
        [--frames N] [--json out.json] [--max-examples 5]
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BLANK = b" " * 80
CLASSES = ("fits-data", "fits-header", "fits-bytes", "apdb", "apdb-bytes", "npz", "npz-bytes",
           "text", "text-bytes", "png", "pdf")
TEXT_CLASSES = ("apdb", "apdb-bytes", "npz", "npz-bytes", "text", "text-bytes", "png", "pdf")

# ---------------------------------------------------------------------------
# Verified time stamps
# ---------------------------------------------------------------------------

_DAY = rb"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_MON = rb"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
#: the FITS kernel's ISO write time (fxfupdhdr.x, fxf_encode_date)
ISO_STAMP = rb"\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-6]\d"
#: IRAF cnvdate: "%3.3s %2d " then "%2d:%02d" (within a year) or "%5d" (sys/etc/cnvdate.x)
CNVDATE = _MON + rb" [ 123]\d (?:[ 12]\d:[0-5]\d| \d{4})"
#: IRAF cnvtime "%3.3s %02d:%02d:%02d %02d-%3.3s-%04d" (sys/etc/cnvtime.x; dttext.x:dtptime)
#: = WARP's time.strftime("%a %H:%M:%S %d-%b-%Y") (aperture.py:154)
CNVTIME = _DAY + rb" [0-2]\d:[0-5]\d:[0-6]\d [0-3]\d-" + _MON + rb"-\d{4}"
#: Python time.ctime(): "%a %b %e %H:%M:%S %Y"
CTIME = _DAY + rb" " + _MON + rb" [ 123]\d [0-2]\d:[0-5]\d:[0-6]\d \d{4}"
#: str(datetime.datetime.today()) (no fraction when the microsecond is 0)
PYDATETIME = rb"\d{4}-[01]\d-[0-3]\d [0-2]\d:[0-5]\d:[0-6]\d(?:\.\d{6})?"
#: "%dm%.1fs" % (int(t / 60), t % 60)
ELAPSED = rb"\d+m\d+\.\ds"

#: FITS cards whose value IS a time stamp: (8-byte key, regex on the card; ``ts`` = the field)
FITS_STAMPS = {
    b"IRAF-TLM": re.compile(rb"IRAF-TLM= '(?P<ts>" + ISO_STAMP + rb")'"),
    b"DATE    ": re.compile(rb"DATE    = '(?P<ts>" + ISO_STAMP + rb")'"),
    b"FIXPIX  ": re.compile(rb"FIXPIX  = '(?P<ts>" + CNVDATE + rb") Bad pixel file is "),
}
DECANTER_CARD = re.compile(rb"DECANTER= 'decanter [^']*' *")


class TextStamp:
    """A time-stamp field of a text file: files matching ``path``, only on line
    ``line`` (1-based) when given, the whole line (terminator excluded) matching
    ``rx``; the ``ts`` group is the field."""

    __slots__ = ("label", "line", "path", "rx")

    def __init__(self, label: str, path: str, rx: bytes, line: int | None = None):
        self.label, self.path, self.line = label, re.compile(path), line
        self.rx = re.compile(rx)


RUN_DB = r"^calibration_data/database/(?:ap[^/]*_(?:sky)?NO\d+_[^/]*|aplast)$"
TEXT_STAMPS = (
    TextStamp("status.txt Starting time", r"^reduction_log/status\.txt$",
              rb"Starting time: (?P<ts>" + CTIME + rb")"),
    TextStamp("status.txt Termination time", r"^reduction_log/status\.txt$",
              rb"Termination time: (?P<ts>" + CTIME + rb")"),
    TextStamp("status.txt Elapsed time", r"^reduction_log/status\.txt$",
              rb"Elapsed time: (?P<ts>" + ELAPSED + rb")"),
    TextStamp("SNratio .dat Time", r"^(?:SNR_dat/fsr[\d.]+/)?SNratio_m\d+_fsr[\d.]+\.dat$",
              rb"Time: (?P<ts>" + PYDATETIME + rb")", line=1),
    TextStamp("database # <cnvtime>", RUN_DB, rb"# (?P<ts>" + CNVTIME + rb")"),
)
MASK = b"<time>"
PDF_DATE = re.compile(rb"/CreationDate \((?P<ts>D:\d{14}(?:Z|[+-]\d\d'\d\d'?)?)\)")


def _pattern(rel: str) -> str:
    s = re.sub(r"_NO\d+", "_NO#", rel)
    s = re.sub(r"skyNO\d+", "skyNO#", s)
    s = re.sub(r"_m\d+", "_m#", s)
    s = re.sub(r"\.\d{4}\.fits", ".####.fits", s)
    s = re.sub(r"WINA\d+", "WINA#", s)
    s = re.sub(r"fsr\d\.\d\d", "fsr#", s)
    s = re.sub(r"(_|/)\d+trans", r"\1#trans", s)
    s = re.sub(r"(ecall)\.\d{4}$", r"\1.####", s)
    return s


# ---------------------------------------------------------------------------
# FITS, parsed from the bytes (no astropy: no scaling, no card re-formatting)
# ---------------------------------------------------------------------------

class Hdu:
    """One header-data unit: its card images (80 bytes each, up to and
    including ``END``), header length (whole blocks), data bytes, padding."""

    __slots__ = ("cards", "hdr_start", "hdr_len", "data_start", "data_len", "pad_len", "bitpix")

    def __init__(self, cards, hdr_start, hdr_len, data_start, data_len, pad_len, bitpix):
        self.cards = cards
        self.hdr_start = hdr_start
        self.hdr_len = hdr_len
        self.data_start = data_start
        self.data_len = data_len
        self.pad_len = pad_len
        self.bitpix = bitpix


def _card_int(card: bytes) -> int:
    return int(card[10:80].split(b"/")[0].strip())


def parse_fits(b: bytes) -> list[Hdu]:
    """Split a FITS file into its HDUs (primary, extensions); raises
    ValueError for a file that is not FITS."""
    hdus = []
    pos = 0
    while pos < len(b):
        start = pos
        cards = []
        while True:
            if pos + 2880 > len(b):
                raise ValueError(f"truncated header at byte {start}")
            blk = b[pos:pos + 2880]
            pos += 2880
            done = False
            for i in range(0, 2880, 80):
                c = blk[i:i + 80]
                cards.append(c)
                if c[:8] == b"END     ":
                    done = True
                    break
            if done:
                break
        if not hdus and cards[0][:8] != b"SIMPLE  ":
            raise ValueError("not a FITS file")
        kv = {}
        for c in cards:
            k = c[:8].decode("ascii", "replace").strip()
            if c[8:10] == b"= " and k not in kv:
                kv[k] = c
        bitpix = _card_int(kv["BITPIX"])
        naxis = _card_int(kv["NAXIS"])
        n = 1 if naxis else 0
        for i in range(1, naxis + 1):
            n *= _card_int(kv[f"NAXIS{i}"])
        pcount = _card_int(kv["PCOUNT"]) if "PCOUNT" in kv else 0
        gcount = _card_int(kv["GCOUNT"]) if "GCOUNT" in kv else 1
        dlen = abs(bitpix) // 8 * gcount * (pcount + n)
        plen = (-dlen) % 2880
        hdus.append(Hdu(cards, start, pos - start, pos, dlen, plen, bitpix))
        pos += dlen + plen
        if pos > len(b):
            raise ValueError("truncated data unit")
    return hdus


def _words(b: bytes, bitpix: int) -> np.ndarray:
    dt = {8: ">u1", 16: ">u2", 32: ">u4", 64: ">u8", -32: ">u4", -64: ">u8"}[bitpix]
    return np.frombuffer(b, dtype=dt)


def _values(b: bytes, bitpix: int) -> np.ndarray:
    dt = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[bitpix]
    return np.frombuffer(b, dtype=dt)


def stamp_span(card: bytes) -> tuple[int, int] | None:
    """Byte span of the time-stamp field of a card that IS a time stamp
    (``FITS_STAMPS``), else None."""
    rx = FITS_STAMPS.get(card[:8])
    if rx is None:
        return None
    m = rx.match(card)
    return m.span("ts") if m else None


def _mask_time(card: bytes) -> tuple[bytes, bool]:
    """The card with a verified time-stamp field masked, and whether it was
    one (so a card that merely contains the mask text never passes)."""
    sp = stamp_span(card)
    return (card, False) if sp is None else (card[:sp[0]] + MASK + card[sp[1]:], True)


def _card_text(h: Hdu) -> list[str]:
    """Non-blank card images in order (without END and decanter's
    ``DECANTER`` card), a verified time-stamp field masked and flagged."""
    out = []
    for c in h.cards[:-1]:                        # without END
        if c == BLANK or DECANTER_CARD.fullmatch(c):
            continue
        img, is_stamp = _mask_time(c)
        out.append(img.decode("ascii", "replace") + (" [time stamp]" if is_stamp else ""))
    return out


def compare_fits(fw: Path, fd: Path, *, check_bytes: bool, tail_ok: bool,
                 masked: Counter | None = None):
    """Per-class results for one FITS pair: ``{class: None | note}``, the
    tail flag, the header diff (examples) and whether decanter's file has a
    ``DECANTER`` card.  ``masked`` (``--bytes``) counts the time-stamp fields
    whose values differ."""
    bw, bd = fw.read_bytes(), fd.read_bytes()
    try:
        hw, hd = parse_fits(bw), parse_fits(bd)
    except (ValueError, KeyError) as exc:
        note = f"unparsable: {exc}"
        return {"fits-data": note, "fits-header": note,
                **({"fits-bytes": note} if check_bytes else {})}, False, None, False
    has_card = any(c[:8] == b"DECANTER" for h in hd for c in h.cards)
    res: dict[str, str | None] = {}
    # --- data, bit for bit, every HDU ------------------------------------------
    tail = False
    tail_bytes: tuple[int, int] | None = None
    notes = []
    if len(hw) != len(hd):
        notes.append(f"{len(hw)} vs {len(hd)} HDUs")
    for k, (a, b) in enumerate(zip(hw, hd)):
        da = bw[a.data_start:a.data_start + a.data_len]
        db = bd[b.data_start:b.data_start + b.data_len]
        if a.bitpix != b.bitpix or len(da) != len(db):
            notes.append(f"HDU {k}: BITPIX/size {a.bitpix}/{len(da)} vs {b.bitpix}/{len(db)}")
            continue
        if da == db:
            continue
        wa, wb = _words(da, a.bitpix), _words(db, b.bitpix)
        diff = np.nonzero(wa != wb)[0]
        va, vb = _values(da, a.bitpix)[diff], _values(db, b.bitpix)[diff]
        with np.errstate(invalid="ignore"):
            same_val = (va == vb) | (np.isnan(va) & np.isnan(vb)) if a.bitpix < 0 else va == vb
        note = f"HDU {k}: {diff.size}/{wa.size} values"
        if same_val.all():
            note += " (all equal as numbers: signed zeros / NaN payloads)"
        elif same_val.any():
            note += f" ({int(same_val.sum())} of them equal as numbers)"
        if k == 0 and len(hw) == 1 and diff.size == 1 and diff[0] == wa.size - 1:
            tail = True
            nb = abs(a.bitpix) // 8
            tail_bytes = (a.data_start + (wa.size - 1) * nb, nb)
            note = "last value only"
        notes.append(note)
    # --- header cards ---------------------------------------------------------------
    hdiff = None
    hnotes = []
    for k, (a, b) in enumerate(zip(hw, hd)):
        ta, tb = _card_text(a), _card_text(b)
        if ta != tb:
            hnotes.append(f"HDU {k}")
            if hdiff is None:
                hdiff = [ln for ln in difflib.unified_diff(ta, tb, lineterm="", n=0)
                         if not ln.startswith(("---", "+++", "@@"))][:8]
    res["fits-header"] = ("cards differ in " + ", ".join(hnotes)) if hnotes else None
    if tail and not (tail_ok and res["fits-header"] is None and len(notes) == 1):
        tail = False                                  # only a lone last value counts
    data_notes = [n for n in notes if not (tail and n == "last value only")]
    res["fits-data"] = "; ".join(data_notes) if data_notes else None
    # --- raw bytes ----------------------------------------------------------------------
    if check_bytes:
        res["fits-bytes"] = _compare_bytes(bw, bd, hw, hd, tail_bytes if tail else None,
                                           masked)
    return res, tail, hdiff, has_card


def _compare_bytes(bw: bytes, bd: bytes, hw: list[Hdu], hd: list[Hdu],
                   tail: tuple[int, int] | None, masked: Counter | None = None) -> str | None:
    """None when the files are byte-identical up to the documented
    exceptions, else where they first differ."""
    if len(bw) != len(bd):
        return f"size {len(bw)} vs {len(bd)} bytes"
    if len(hw) != len(hd):
        return "HDU count"
    excused: Counter = Counter()
    for k, (a, b) in enumerate(zip(hw, hd)):
        if a.hdr_len != b.hdr_len or len(a.cards) != len(b.cards):
            return (f"HDU {k}: header {a.hdr_len // 2880} vs {b.hdr_len // 2880} blocks, "
                    f"END at card {len(a.cards)} vs {len(b.cards)}")
        last = max((i for i, c in enumerate(a.cards[:-1]) if c != BLANK), default=-1)
        for i, (ca, cb) in enumerate(zip(a.cards, b.cards)):
            if ca == cb:
                continue
            sa, sb = stamp_span(ca), stamp_span(cb)
            if sa is not None and sa == sb and ca[:sa[0]] == cb[:sb[0]] and ca[sa[1]:] == cb[sb[1]:]:
                excused["fits " + ca[:8].decode("ascii").strip()] += 1
                continue
            if ca == BLANK and i == last + 1 and DECANTER_CARD.fullmatch(cb):
                continue
            return f"HDU {k}: card {i + 1} {ca[:30]!r} vs {cb[:30]!r}"
        rest_w = bw[a.hdr_start + len(a.cards) * 80:a.hdr_start + a.hdr_len]
        rest_d = bd[b.hdr_start + len(b.cards) * 80:b.hdr_start + b.hdr_len]
        if rest_w != rest_d:
            return f"HDU {k}: bytes after END differ"
        lo, hi = a.data_start, a.data_start + a.data_len + a.pad_len
        if bw[lo:hi] != bd[lo:hi]:
            if tail is not None and lo <= tail[0] < hi:
                s, n = tail
                if bw[lo:s] == bd[lo:s] and bw[s + n:hi] == bd[s + n:hi]:
                    continue
            j = next(i for i in range(lo, hi) if bw[i] != bd[i])
            where = "data" if j < a.data_start + a.data_len else "padding"
            return f"HDU {k}: {where} byte {j - lo}"
    if masked is not None:
        masked.update(excused)
    return None


# ---------------------------------------------------------------------------
# Text and databases
# ---------------------------------------------------------------------------

def mask_text(rel: str, data: bytes) -> tuple[bytes, list[tuple[int, str, bytes]]]:
    """``data`` with every verified time-stamp field (``TEXT_STAMPS`` that
    apply to ``rel``) replaced in place by ``<time>``; everything else --
    the labels, the rest of the line, ``\\n`` / ``\\r\\n`` terminators, a
    missing final newline -- kept as it is.  Also returns the masked fields
    ``(line number, label, original bytes)``: two files compare equal only
    when their masked fields are on the same lines with the same labels."""
    rules = [r for r in TEXT_STAMPS if r.path.search(rel)]
    if not rules:
        return data, []
    lines = data.split(b"\n")
    fields = []
    for n, ln in enumerate(lines, 1):
        for r in rules:
            if r.line is not None and r.line != n:
                continue
            m = r.rx.fullmatch(ln)
            if m:
                s, e = m.span("ts")
                fields.append((n, r.label, ln[s:e]))
                lines[n - 1] = ln[:s] + MASK + ln[e:]
                break
    return b"\n".join(lines), fields


def _field_places(fields) -> list[tuple[int, str]]:
    return [(n, lab) for n, lab, _ in fields]


def compare_text(fw: Path, fd: Path, rel: str):
    """Line for line (``splitlines``: terminators not compared), time-stamp
    fields masked in place."""
    a, fa = mask_text(rel, fw.read_bytes())
    b, fb = mask_text(rel, fd.read_bytes())
    la = a.decode("utf-8", "replace").splitlines()
    lb = b.decode("utf-8", "replace").splitlines()
    if la == lb:
        if _field_places(fa) == _field_places(fb):
            return None
        return [f"time-stamp fields at {_field_places(fa)} vs {_field_places(fb)}"]
    return [ln for ln in difflib.unified_diff(la, lb, lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))][:6]


def compare_text_bytes(fw: Path, fd: Path, rel: str, masked: Counter | None = None):
    """Byte for byte with the verified time-stamp fields masked in place
    (terminators and the final newline count); None or where it differs."""
    a, fa = mask_text(rel, fw.read_bytes())
    b, fb = mask_text(rel, fd.read_bytes())
    if a == b:
        if _field_places(fa) != _field_places(fb):
            return f"time-stamp fields at {_field_places(fa)} vs {_field_places(fb)}"
        if masked is not None:
            for (_, lab, x), (_, _, y) in zip(fa, fb, strict=True):
                masked[lab] += x != y
        return None
    nl = b"\n"
    la, lb = a.split(nl), b.split(nl)
    for i, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x != y:
            if x.rstrip(b"\r") == y.rstrip(b"\r"):
                return f"line {i + 1}: line terminator ({x[-1:]!r} vs {y[-1:]!r})"
            return f"line {i + 1}: {x[:70]!r} vs {y[:70]!r}"
    if a.rstrip(nl) == b.rstrip(nl):
        ta, tb = len(a) - len(a.rstrip(nl)), len(b) - len(b.rstrip(nl))
        return f"final newline: {ta} vs {tb} trailing newline(s)"
    return f"{len(la)} vs {len(lb)} lines"


# ---------------------------------------------------------------------------
# npz
# ---------------------------------------------------------------------------

def _bits_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise identity of two same-dtype arrays: every fixed-size
    element by its bytes (so floats bit for bit), objects by ``==``."""
    if x.dtype.kind == "O":
        return np.array([bool(np.all(a == b)) for a, b in zip(x.ravel(), y.ravel())],
                        bool).reshape(x.shape)
    n = x.dtype.itemsize
    xb = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (n,))
    yb = np.ascontiguousarray(y).view(np.uint8).reshape(y.shape + (n,))
    return (xb == yb).all(axis=-1)


def compare_npz(fw: Path, fd: Path):
    A, B = np.load(fw, allow_pickle=True), np.load(fd, allow_pickle=True)
    bad = []
    for k in A.files:
        if k not in B.files:
            bad.append(f"{k}: missing")
            continue
        x, y = np.asarray(A[k]), np.asarray(B[k])
        if x.shape != y.shape:
            bad.append(f"{k}: shape {x.shape} vs {y.shape}")
            continue
        if x.dtype != y.dtype:
            bad.append(f"{k}: dtype {x.dtype} vs {y.dtype}")
            continue
        eq = _bits_equal(x, y)
        if not np.all(eq):
            bad.append(f"{k}: {int((~eq).sum())}/{x.size} differ")
    for k in B.files:
        if k not in A.files:
            bad.append(f"{k}: extra")
    return bad or None


#: extra-field IDs that hold time stamps: extended timestamp (UT), NTFS, Info-ZIP Unix (old)
_ZIP_TIME_EXTRA = {0x5455, 0x000A, 0x5855}


def zip_time_fields(b: bytes) -> tuple[list[tuple[int, int]], list[bytes], list[bytes]]:
    """The byte spans of every zip time stamp of a zip file -- DOS time+date of
    each central-directory entry and of its local header, and the data of
    time-stamp extra fields -- plus the member names in central-directory
    order and in local-header (file) order.  Raises ValueError for a file
    that is not a zip."""
    e = b.rfind(b"PK\x05\x06", max(0, len(b) - 65557))
    if e < 0:
        raise ValueError("no end-of-central-directory record")
    n_ent, _cd_size, cd_off = struct.unpack("<HII", b[e + 10:e + 20])
    if n_ent == 0xFFFF or cd_off == 0xFFFFFFFF:
        z = b.rfind(b"PK\x06\x06", 0, e)
        if z < 0:
            raise ValueError("zip64 end record missing")
        n_ent, _cd_size, cd_off = struct.unpack("<QQQ", b[z + 32:z + 56])
    spans: list[tuple[int, int]] = []

    def extra_spans(start: int, length: int):
        p, end = start, start + length
        while p + 4 <= end:
            hid, hlen = struct.unpack("<HH", b[p:p + 4])
            if hid in _ZIP_TIME_EXTRA:
                spans.append((p + 4, min(p + 4 + hlen, end)))
            p += 4 + hlen

    names, local = [], []
    p = cd_off
    for _ in range(n_ent):
        if b[p:p + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central-directory entry at {p}")
        (nlen, xlen, clen) = struct.unpack("<HHH", b[p + 28:p + 34])
        (lho,) = struct.unpack("<I", b[p + 42:p + 46])
        name = b[p + 46:p + 46 + nlen]
        if lho == 0xFFFFFFFF:
            # the zip64 extra field (0x0001) holds, in order, only the 0xFFFFFFFF fields:
            # uncompressed size, compressed size, local header offset
            csz, usz = struct.unpack("<II", b[p + 20:p + 28])
            k = (usz == 0xFFFFFFFF) + (csz == 0xFFFFFFFF)
            q, end = p + 46 + nlen, p + 46 + nlen + xlen
            while q + 4 <= end:
                hid, hlen = struct.unpack("<HH", b[q:q + 4])
                if hid == 0x0001:
                    (lho,) = struct.unpack("<Q", b[q + 4 + 8 * k:q + 12 + 8 * k])
                    break
                q += 4 + hlen
        spans.append((p + 12, p + 16))                # central directory: time, date
        extra_spans(p + 46 + nlen, xlen)
        if b[lho:lho + 4] != b"PK\x03\x04":
            raise ValueError(f"bad local header at {lho}")
        spans.append((lho + 10, lho + 14))            # local header: time, date
        lnlen, lxlen = struct.unpack("<HH", b[lho + 26:lho + 30])
        extra_spans(lho + 30 + lnlen, lxlen)
        names.append(name)
        local.append((lho, b[lho + 30:lho + 30 + lnlen]))
        p += 46 + nlen + xlen + clen
    return spans, names, [n for _, n in sorted(local)]


def _npy_header_len(x: bytes) -> int:
    """Length of a ``.npy`` file's magic + header (numpy format 1.0 / 2.0 / 3.0)."""
    if x[:6] != b"\x93NUMPY" or len(x) < 12:
        return min(len(x), 10)
    if x[6] == 1:
        return 10 + struct.unpack("<H", x[8:10])[0]
    return 12 + struct.unpack("<I", x[8:12])[0]


def _masked(b: bytes, spans: list[tuple[int, int]]) -> bytes:
    m = bytearray(b)
    for s, e in spans:
        m[s:e] = b"\0" * (e - s)
    return bytes(m)


def compare_npz_bytes(fw: Path, fd: Path, masked: Counter | None = None):
    """The zip containers member by member: names and their order (central
    directory and file order), each member's ``.npy`` bytes, then the whole
    file with only the zip time stamps masked."""
    a, b = fw.read_bytes(), fd.read_bytes()
    if a == b:
        return None
    try:
        sa, na, la = zip_time_fields(a)
        sb, nb, lb = zip_time_fields(b)
    except (ValueError, struct.error) as exc:
        return f"not a zip container: {exc}"
    if na != nb:
        return f"members {[n.decode() for n in na]} vs {[n.decode() for n in nb]}"
    if la != lb:
        return f"member file order {[n.decode() for n in la]} vs {[n.decode() for n in lb]}"
    za, zb = zipfile.ZipFile(io.BytesIO(a)), zipfile.ZipFile(io.BytesIO(b))
    for n in na:
        x, y = za.read(n.decode()), zb.read(n.decode())
        if x != y:
            part = "header" if x[:_npy_header_len(x)] != y[:_npy_header_len(y)] else "data"
            return f"{n.decode()}: .npy bytes differ ({part}; {len(x)} vs {len(y)} bytes)"
    for ia, ib in zip(za.infolist(), zb.infolist(), strict=True):
        for f in ("compress_type", "flag_bits", "create_system", "create_version",
                  "extract_version", "external_attr", "internal_attr", "comment", "CRC",
                  "compress_size", "file_size"):
            if getattr(ia, f) != getattr(ib, f):
                return (f"{ia.filename}: zip metadata {f} {getattr(ia, f)!r} vs "
                        f"{getattr(ib, f)!r}")
    if za.comment != zb.comment:
        return f"zip metadata comment {za.comment!r} vs {zb.comment!r}"
    if len(a) != len(b):
        return f"container size {len(a)} vs {len(b)} bytes"
    ma, mb = _masked(a, sa), _masked(b, sb)
    if sorted(sa) != sorted(sb) or ma != mb:
        j = next((i for i in range(len(a)) if ma[i] != mb[i]), None)
        return f"zip container byte {j} differs (metadata)"
    if masked is not None:
        masked["zip time stamps"] += 1
    return None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def mask_pdf(b: bytes) -> tuple[bytes, list[tuple[int, int]]]:
    """``/CreationDate (D:...)`` value masked in place (same length), and
    where."""
    spans = [m.span("ts") for m in PDF_DATE.finditer(b)]
    m = bytearray(b)
    for s, e in spans:
        m[s:e] = b"X" * (e - s)
    return bytes(m), spans


def compare_pdf(fw: Path, fd: Path, masked: Counter | None = None):
    a, b = fw.read_bytes(), fd.read_bytes()
    if a == b:
        return None
    (ma, sa), (mb, sb) = mask_pdf(a), mask_pdf(b)
    if ma == mb and sa == sb:
        if masked is not None:
            masked["pdf /CreationDate"] += 1
        return None
    return "bytes differ"


# ---------------------------------------------------------------------------
# s13 tail prediction
# ---------------------------------------------------------------------------

def _tail_spectrum(rel: str) -> bool:
    """A per-frame FSR-cut VAC spectrum (where WARP's s13 tail pixel lives)."""
    return bool(re.search(r"_NO\d+/(onedspec/VAC_flux|sky_emission/VAC)/fsr", rel))


_VAC_OBJ = re.compile(r"^(?P<fd>[^/]+_NO\d+)/onedspec/VAC_flux/fsr(?P<c>[\d.]+)/[^/]*_m(?P<m>\d+)_fsr[\d.]+_VAC\.fits$")
_VAC_SKY = re.compile(r"^(?P<fd>[^/]+_NO\d+)/sky_emission/VAC/fsr(?P<c>[\d.]+)/[^/]*_m(?P<m>\d+)trans1dcutw_fsr[\d.]+_VAC\.fits$")
_FSR: dict = {}


def _card_value_text(card: bytes) -> str:
    v = card[10:].decode("ascii", "replace")
    if "/" in v and not v.strip().startswith("'"):
        v = v.split("/")[0]
    return v.strip()


def tail_predicted(warp_root: Path, rel: str) -> bool | None:
    """Whether the s13 cut ``rel`` is an ``n_r < n`` cut, from the WARP tree's
    own s12 spectrum of that frame and order (its CRVAL1/CDELT1 text and
    length, the FSR table's bounds, ``scopy_wcs``); None when the WARP tree
    did not keep it."""
    m = _VAC_OBJ.match(rel)
    sub, pat = "intermediate_files/OBJ/8A-OBJ-1DSPEC/4-OBJ-1DSPEC_dispcor", "csw.fits"
    if not m:
        m = _VAC_SKY.match(rel)
        sub, pat = "intermediate_files/SKY/7-SKY_dispcor", "trans1dcutw.fits"
    if not m:
        return None
    src = sorted((warp_root / m["fd"] / sub).glob(f"*_m{m['m']}{pat}"))
    if len(src) != 1:
        return None
    from decanter.calib.discover import _BUNDLED_FSR_TABLE
    from decanter.io.fsr import load as load_fsr
    from decanter.wavelength.fsr import fsr_bounds
    from decanter.wavelength.iraf_wcs import scopy_wcs
    if not _FSR:
        _FSR.update(load_fsr(_BUNDLED_FSR_TABLE))
    order = int(m["m"])
    if order not in _FSR:
        return None
    h = parse_fits(src[0].read_bytes())[0]
    kv: dict[str, bytes] = {}
    for c in h.cards:
        if c[8:10] == b"= ":
            kv.setdefault(c[:8].decode("ascii", "replace").strip(), c)
    if not {"CRVAL1", "CDELT1", "NAXIS1"} <= set(kv):
        return None
    lo, hi = fsr_bounds(_FSR[order].lambda_min, _FSR[order].lambda_max, float(m["c"]))
    sc = scopy_wcs(_card_value_text(kv["CRVAL1"]), _card_value_text(kv["CDELT1"]),
                   int(_card_value_text(kv["NAXIS1"])), lo, hi)
    return bool(sc.n_r < sc.n)


def frame_index(rel: str) -> int | None:
    m = re.match(r"[^/]*_NO(\d+)/", rel)
    return int(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("warp")
    ap.add_argument("decanter")
    ap.add_argument("--bytes", action="store_true",
                    help="also compare every file byte for byte (FITS layout and padding, text "
                         "terminators, npz containers), only verified time stamps masked")
    ap.add_argument("--frames", type=int, default=0, help="only frame dirs NO1..NOn "
                    "(and the per-frame DB files of those frames)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-examples", type=int, default=4)
    a = ap.parse_args(argv)
    W, D = Path(a.warp), Path(a.decanter)

    def keep(rel: str) -> bool:
        if not a.frames:
            return True
        i = frame_index(rel)
        if i is not None:
            return i <= a.frames
        m = re.search(r"_(?:sky)?NO(\d+)_", Path(rel).name)
        if rel.startswith("calibration_data/database/") and m:
            return int(m.group(1)) <= a.frames
        return True

    wfiles = sorted(p.relative_to(W).as_posix() for p in W.rglob("*") if p.is_file())
    wfiles = [r for r in wfiles if keep(r)]
    dfiles = {p.relative_to(D).as_posix() for p in D.rglob("*") if p.is_file()}
    stats: dict[str, Counter] = defaultdict(Counter)         # per pattern
    cls_tot: dict[str, Counter] = {c: Counter() for c in CLASSES}
    examples: dict[str, list] = defaultdict(list)
    missing: Counter = Counter()
    masked: Counter | None = Counter() if a.bytes else None
    tails: list[str] = []
    not_tail: list[str] = []          # last value only, but WARP's s12 WCS: no n_r < n cut
    unverified: list[str] = []        # tails the WARP tree cannot confirm (no s12 kept)
    decanter_cards = 0

    def note(cls_key, rel, *info):
        if len(examples[cls_key]) < a.max_examples:
            examples[cls_key].append([rel, *info])

    for rel in wfiles:
        fw, fd = W / rel, D / rel
        pat = _pattern(rel)
        if rel not in dfiles:
            missing[pat] += 1
            continue
        suf = fw.suffix.lower()
        results: dict[str, object] = {}
        try:
            if suf == ".fits":
                res, tail, hdiff, has_card = compare_fits(fw, fd, check_bytes=a.bytes,
                                                          tail_ok=_tail_spectrum(rel),
                                                          masked=masked)
                if tail:
                    pred = tail_predicted(W, rel)
                    if pred is False:              # WARP's own s12 WCS: no n_r < n cut
                        not_tail.append(rel)
                        res, tail, hdiff, has_card = compare_fits(fw, fd, check_bytes=a.bytes,
                                                                  tail_ok=False)
                        res["fits-data"] = (res["fits-data"] or "") + \
                            " (last value only, not an n_r < n cut)"
                    elif pred is None:
                        unverified.append(rel)
                decanter_cards += has_card
                if tail:
                    tails.append(rel)
                    cls_tot["fits-data"]["tail"] += 1
                    stats[pat]["tail"] += 1
                for c, r in res.items():
                    results[c] = r
                    if r is not None:
                        note(f"{pat} [{c}]", rel, r, hdiff if c == "fits-header" else None)
            elif rel.startswith("calibration_data/database/"):
                results["apdb"] = compare_text(fw, fd, rel)
                if a.bytes:
                    results["apdb-bytes"] = compare_text_bytes(fw, fd, rel, masked)
            elif suf == ".npz":
                results["npz"] = compare_npz(fw, fd)
                if a.bytes:
                    results["npz-bytes"] = compare_npz_bytes(fw, fd, masked)
            elif suf in (".png", ".eps"):
                results["png"] = None if fw.read_bytes() == fd.read_bytes() else "bytes differ"
            elif suf == ".pdf":
                results["pdf"] = compare_pdf(fw, fd, masked)
            else:
                results["text"] = compare_text(fw, fd, rel)
                if a.bytes:
                    results["text-bytes"] = compare_text_bytes(fw, fd, rel, masked)
        except Exception as exc:  # pragma: no cover
            results["error"] = f"{type(exc).__name__}: {exc}"
        for c, r in results.items():
            if c == "error":
                note(f"{pat} [error]", rel, r)
                continue
            if c != "fits-data" or rel not in tails or r is not None:
                cls_tot[c]["identical" if r is None else "differ"] += 1
            if r is not None and c in TEXT_CLASSES:
                note(f"{pat} [{c}]", rel, r)
        bad = [c for c, r in results.items() if r is not None]
        if bad:
            stats[pat]["differ"] += 1
            for c in bad:
                stats[pat][f"differ:{c}"] += 1
        elif rel not in tails:
            stats[pat]["identical"] += 1
    extra: Counter = Counter()
    wset = set(wfiles)
    for rel in dfiles:
        if rel not in wset and keep(rel):
            extra[_pattern(rel)] += 1
    n_id = sum(v["identical"] for v in stats.values())
    n_df = sum(v["differ"] for v in stats.values())
    n_tail = len(tails)
    report = {
        "warp": str(W), "decanter": str(D), "frames": a.frames or "all",
        "bytes_checked": a.bytes,
        "tail_pixel_files": tails,
        "tail_pixel_unverified": unverified,
        "last_value_not_tail": not_tail,
        "classes": {k: dict(v) for k, v in sorted(stats.items())},
        "class_totals": {c: dict(v) for c, v in cls_tot.items() if v},
        "decanter_card_files": decanter_cards,
        "masked_differences": dict(sorted(masked.items())) if masked is not None else None,
        "missing_in_decanter": dict(sorted(missing.items())),
        "extra_in_decanter": dict(sorted(extra.items())),
        "examples": dict(examples),
        "totals": {"warp_files": len(wfiles), "identical": n_id, "differ": n_df,
                   "tail_pixel": n_tail, "missing": sum(missing.values()),
                   "extra": sum(extra.values())},
    }
    print(f"WARP {W}\ndecanter {D}")
    print(f"files: {len(wfiles)} WARP, identical {n_id}, differ {n_df}, "
          f"accepted s13 tail pixel {n_tail}, "
          f"missing in decanter {sum(missing.values())}, extra in decanter {sum(extra.values())}")
    print("by class (a file counts once per class it is compared in):")
    for c in CLASSES:
        v = cls_tot[c]
        if not v:
            if c.endswith("-bytes") and not a.bytes:
                print(f"  {c:12s} not checked (--bytes)")
            continue
        tail = f", {v['tail']} accepted s13 tail" if v.get("tail") else ""
        print(f"  {c:12s} {v['identical']:7d} identical {v['differ']:6d} differ{tail}")
    print(f"  ({decanter_cards} decanter FITS files carry the DECANTER provenance card)")
    if masked is not None:
        print("  masked time-stamp fields whose values differ: "
              + (", ".join(f"{k} {v}" for k, v in sorted(masked.items()) if v) or "none"))
    for k, v in sorted(stats.items()):
        flag = "" if not v["differ"] else "   <-- DIFFER " + ",".join(
            sorted(c.split(":", 1)[1] for c in v if c.startswith("differ:")))
        tail = f" {v['tail']:4d} tail" if v["tail"] else ""
        print(f"  {k:90s} {v['identical']:6d} identical {v['differ']:5d} differ{tail}{flag}")
    if tails:
        print("accepted s13 tail-pixel files (only the last value differs, header identical; "
              f"{len(tails) - len(unverified)} confirmed n_r < n by WARP's own s12 spectrum, "
              f"{len(unverified)} not checkable):")
        for t in tails[:50]:
            print("  ", t)
    if not_tail:
        print("last value only, but not an n_r < n cut (WARP's s12 WCS) -- counted as differing:")
        for t in not_tail[:50]:
            print("  ", t)
    if missing:
        print("missing in decanter:")
        for k, v in sorted(missing.items()):
            print(f"  {k:90s} {v:6d}")
    if extra:
        print("extra in decanter:")
        for k, v in sorted(extra.items()):
            print(f"  {k:90s} {v:6d}")
    for k, v in examples.items():
        for e in v:
            print("  example", k, json.dumps(e, default=str)[:700])
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))
    return 0 if (n_df == 0 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
