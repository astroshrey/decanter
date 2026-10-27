"""The IRAF FITS kernel's file layout (``decanter.warptree_io``).

``fxf_write_header`` (sys/imio/iki/fxf/fxfupdhdr.x) puts blank cards before
``END`` so that ``END`` is the last card of the last 2880-byte block; decanter
writes the same bytes, with its ``DECANTER`` card in the first of those blank
cards (or none when there is no blank card).  With a WARP reference tree on
disk, WARP's own IRAF files are rebuilt from their data and header text and
must come out byte for byte, time stamps and the ``DECANTER`` card aside.
"""
from __future__ import annotations

import numpy as np
import pytest

from decanter._localpaths import WINERED_ROOT
from decanter.utils.iraf_header import IrafHeader, str_image
from decanter.warptree_io import _cards_bytes, iraf_fits_bytes, read_ua

REFS = [WINERED_ROOT / "reductions" / n for n in
        ("TOI3486_decanterref", "TOI2109_decanterref", "WASP69_decanterref")]
REF = next((r for r in REFS if r.is_dir()), None)
needs_ref = pytest.mark.skipif(REF is None, reason="no WARP -s reference tree")

BLANK = " " * 80
TIME_COLS = {"IRAF-TLM": (11, 30), "DATE": (11, 30)}


def _cards(n: int) -> list[str]:
    return [f"KEY{i:05d}= {i:20d}".ljust(80) for i in range(n)]


def _split(b: bytes) -> list[str]:
    assert len(b) % 2880 == 0
    return [b[i:i + 80].decode("ascii") for i in range(0, len(b), 80)]


@pytest.mark.parametrize("n", [1, 7, 34, 35, 36, 37, 71, 72, 209, 215])
def test_end_is_the_last_card_of_the_last_block(n):
    cards = _split(_cards_bytes(_cards(n)))
    assert cards[-1] == "END".ljust(80)
    assert len(cards) == -(-(n + 1) // 36) * 36          # the minimal block count
    assert all(c == BLANK for c in cards[n:-1])          # blank cards, no trailing spaces
    assert cards[:n] == _cards(n)


@pytest.mark.parametrize("n", [1, 34, 36, 70, 209])
def test_decanter_card_takes_the_first_blank_card(n):
    plain = _split(_cards_bytes(_cards(n)))
    prov = _split(_cards_bytes(_cards(n), "decanter 0.0.1 mode=warp"))
    assert len(prov) == len(plain)                       # same size as IRAF's file
    diff = [i for i, (a, b) in enumerate(zip(plain, prov)) if a != b]
    assert diff == [n]
    assert plain[n] == BLANK
    assert prov[n] == str_image("DECANTER", "decanter 0.0.1 mode=warp")


@pytest.mark.parametrize("n", [35, 71, 215])
def test_no_blank_card_no_decanter_card(n):
    # the cards fill the last block up to END: IRAF has no blank card, and
    # decanter does not grow the header for its card
    assert _cards_bytes(_cards(n), "decanter 0.0.1") == _cards_bytes(_cards(n))
    assert len(_cards_bytes(_cards(n))) == (n + 1) * 80


def test_long_string_cards_count_their_continue_cards():
    long_card = ("LONGSTR = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx&'"
                 .ljust(80) + "CONTINUE  'yyyy'".ljust(80))
    cards = _split(_cards_bytes(_cards(33) + [long_card], "d"))   # 35 slots: no blank
    assert len(cards) == 36 and cards[-1].startswith("END")
    assert not any(c.startswith("DECANTER") for c in cards)
    cards = _split(_cards_bytes(_cards(32) + [long_card], "d"))   # 34 slots + DECANTER
    assert len(cards) == 36 and cards[34].startswith("DECANTER")
    assert cards[33].startswith("CONTINUE")


def test_written_file_layout_and_zero_padding():
    ua = IrafHeader.from_images([str_image("OBJECT", "x"), str_image("IRAF-TLM", "t")])
    data = np.arange(7, dtype=np.float32)
    b = iraf_fits_bytes(data, ua, provenance="decanter test")
    cards = _split(b[:2880])
    assert cards[-1].startswith("END") and cards[7].startswith("DECANTER")
    body = b[2880:]
    assert len(body) == 2880 and body[28:] == b"\0" * (2880 - 28)
    assert np.array_equal(np.frombuffer(body[:28], ">f4"), data)


def _iraf_files(root):
    """WARP's IRAF-written FITS files of a tree (not the astropy cosmic-ray
    files, the copied calibration set or the raw copies, whose BZERO/BSCALE
    read_ua does not keep)."""
    skip = ("/mask_", "/mf1_", "/mf2_", "/noise_", "_resample.", "calibration_data/",
            "rawdata_image/")
    out = []
    for p in sorted(root.rglob("*.fits")):
        rel = "/" + p.relative_to(root).as_posix()
        if not any(s in rel for s in skip):
            out.append(p)
    return out


@needs_ref
def test_warp_iraf_files_are_rebuilt_byte_for_byte():
    """Every kind of IRAF file WARP writes (2-D stages, strips, 1-D chain,
    VAC/AIR/norm/cont, sums): its data and UA through iraf_fits_bytes give
    WARP's bytes, except the time-stamp values and DECANTER in the first
    blank card."""
    files = _iraf_files(REF)
    by_kind = {}
    for p in files:
        kind = (p.parent.parent.name + "/" + p.parent.name + "/"
                + ("sky" if "sky" in p.name else "obj"))
        by_kind.setdefault(kind, p)
    assert len(by_kind) >= 30
    for p in by_kind.values():
        warp = p.read_bytes()
        data, ua = read_ua(p)
        mine = iraf_fits_bytes(data, ua, provenance="decanter test")
        assert len(mine) == len(warp), p
        end = next(i for i in range(0, len(warp), 80) if warp[i:i + 8] == b"END     ")
        n_hdr = end + 80
        assert n_hdr % 2880 == 0, p                     # WARP's END ends its block
        wc = _split(warp[:n_hdr])
        dc = _split(mine[:n_hdr])
        last = max(i for i, c in enumerate(wc[:-1]) if c != BLANK)
        for i, (a, b) in enumerate(zip(wc, dc)):
            if a == b:
                continue
            key = a[:8].strip()
            if key in TIME_COLS:
                x, z = TIME_COLS[key]
                assert a[:x] == b[:x] and a[z:] == b[z:], (p, i)
                continue
            assert a == BLANK and b.startswith("DECANTER") and i == last + 1, (p, i, a, b)
        assert mine[n_hdr:] == warp[n_hdr:], p                     # data + padding
