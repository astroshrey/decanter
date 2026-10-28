"""FITS headers and files the way WARP's IRAF tasks leave them.

Every FITS file WARP keeps was last written by the IRAF FITS kernel
(``sys/imio/iki/fxf/fxfupdhdr.x``), except the cosmic-ray products that
``warp/badpixmask.py`` saves with astropy (:func:`write_astropy_image`).
The kernel writes ``SIMPLE``/``BITPIX``/``NAXIS``/``NAXISn``/``EXTEND`` with
its own comments, then the user area (UA): ``ORIGIN``, ``IRAF-TLM`` and
``OBJECT`` (the image title) lead it, a ``DATE`` card already near the top
stays in place with a fresh value, and every other card follows in the order
the tasks added it (a new keyword is appended; an existing one is updated in
place; :mod:`decanter.wavelength.iraf_onedspec` has the card primitives).

Here a header is carried as its UA (an astropy ``Header`` whose numeric WCS
cards hold IRAF's text) and :func:`write_iraf_fits` adds the kernel cards
and writes the bytes directly, so every card image is exactly the one built,
laid out as the kernel lays it out (``fxf_write_header``,
fxfupdhdr.x:927-971): the cards, blank cards up to the end of the last
2880-byte block, and ``END`` as that block's last card; the data unit
big-endian, zero-padded to a block (fxfupdhdr.x:231-242).  A file decanter
writes this way is byte for byte the file WARP's IRAF task writes, except:

* the time stamps: the values of ``IRAF-TLM`` and ``DATE`` (the write time;
  ``FIXPIX``'s run time is in the UA text the tree writer builds);
* the provenance card ``DECANTER`` (``'decanter <version>[ mode=<mode>]'``),
  which decanter writes in the first blank card after the UA -- where IRAF
  has a blank card -- so the file keeps IRAF's size and every other byte.
  A header whose cards fill its last block up to ``END`` (no blank card)
  gets no ``DECANTER`` card (:func:`_cards_bytes`).

Files WARP writes with astropy (:func:`write_astropy_image`,
:func:`write_astropy_replaced`) are written with astropy here too.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from astropy.io import fits as _afits

from decanter.utils.iraf_header import Card80, IrafHeader, str_image

FITS_ORIGIN = "NOAO-IRAF FITS Image Kernel July 2003"
_CARDS_PER_BLOCK = 36                  # 2880 / 80
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BZERO", "BSCALE", "PCOUNT",
               "GCOUNT", "XTENSION", "END"}


def _is_structural(key: str) -> bool:
    return key in _STRUCTURAL or (key.startswith("NAXIS") and key[5:].isdigit())


def iraf_now() -> str:
    """``fxf_encode_date`` text (ISO, no fraction)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def raw_ua(raw, *, pipeline: str | None = None, keep_scale: bool = False) -> IrafHeader:
    """The UA IRAF makes of a raw frame (``fxf_rfits`` + ``fxf_updhdr``).

    ``OBJECT`` becomes the title and is written right after ``IRAF-TLM``;
    the raw ``DATE`` card keeps its place.  ``pipeline`` adds WARP's
    ``hedit PIPELINE`` (Warp_sci.py:167).  ``raw`` is an astropy header
    (as read from the file) or an :class:`IrafHeader`.  ``keep_scale``
    keeps ``BZERO``/``BSCALE`` in place (an image updated in place, like
    WARP's ``hedit`` of its raw copies; a NEW_COPY drops them).
    """
    src = IrafHeader.from_astropy(raw)
    ua = IrafHeader()
    ua.append(_afits.Card("ORIGIN", FITS_ORIGIN, "FITS file originator").image)
    ua.append(_afits.Card("IRAF-TLM", iraf_now(), "Time of last modification").image)
    title = src.get("OBJECT")
    if title is not None and str(title) != "":
        ua.append(_afits.Card("OBJECT", str(title), "Name of the object observed").image)
    for c in src.cards:
        if keep_scale and c.keyword in ("BZERO", "BSCALE"):
            ua.append(Card80(c.keyword, c.image))
            continue
        if _is_structural(c.keyword) or c.keyword in ("ORIGIN", "IRAF-TLM", "OBJECT"):
            continue
        if not c.image.strip():
            continue
        ua.append(Card80(c.keyword, c.image))
    if pipeline is not None:
        ua.put_str("PIPELINE", pipeline)
    return ua


def _cards_bytes(cards: list[str], provenance: str | None = None) -> bytes:
    """A header unit laid out as the IRAF FITS kernel writes it.

    ``fxf_write_header`` (sys/imio/iki/fxf/fxfupdhdr.x:927-971) writes the
    cards record by record, fills the last 2880-byte record with blanks and
    puts ``END`` in its last 80 bytes (a header whose cards end a record
    exactly gets a record of 35 blank cards and ``END``).  So ``END`` is
    always the last card of the last block, preceded by blank cards; no
    space padding follows it.  (WARP's 103,785 IRAF and astropy FITS files
    in 61 fresh runs, and the three ``-s`` references, all have this layout
    with the minimal number of blocks: the kernel allocates
    ``ceil((cards + 1) / 36)`` blocks, and no WARP task shrinks a header
    after its pixels are written.)

    ``provenance`` is decanter's ``DECANTER`` card value: it is written in
    the first blank card, i.e. right after ``cards``, and only when the
    layout has a blank card there (``len(cards) % 36 != 35``); otherwise
    it is left out, so the file keeps IRAF's size and every other byte.
    """
    out = []
    for img in cards:
        if len(img) % 80:
            img = img.ljust((len(img) // 80 + 1) * 80)
        out.append(img)
    s = "".join(out)
    n = len(s) // 80                                # card slots before END
    if provenance is not None and n % _CARDS_PER_BLOCK != _CARDS_PER_BLOCK - 1:
        s += str_image("DECANTER", provenance)
        n += 1
    nblank = (-(n + 1)) % _CARDS_PER_BLOCK
    s += " " * (80 * nblank) + "END".ljust(80)
    return s.encode("ascii")


def iraf_header_cards(shape: tuple[int, ...], bitpix: int, ua: IrafHeader, *,
                      stamp: bool = True, keep_date: bool = False) -> list[str]:
    """Card images of a FITS-kernel header for an image of ``shape`` (numpy
    order): the kernel's cards, then the UA (any ``DECANTER`` card in it
    dropped; :func:`_cards_bytes` places decanter's own)."""
    cards = [_afits.Card("SIMPLE", True, "Fits standard").image,
             _afits.Card("BITPIX", bitpix, "Bits per pixel").image,
             _afits.Card("NAXIS", len(shape), "Number of axes").image]
    for i, n in enumerate(reversed(shape), start=1):
        cards.append(_afits.Card(f"NAXIS{i}", int(n), "Axis length").image)
    cards.append(_afits.Card("EXTEND", False, "File may contain extensions").image)
    h = IrafHeader.from_astropy(ua)          # a copy (IrafHeader or astropy Header)
    if stamp:
        now = iraf_now()
        for key in ("IRAF-TLM",) + (() if keep_date else ("DATE",)):
            if key in h:
                h.set_string_keep_comment(key, now)
    while "DECANTER" in h:
        h.delete("DECANTER")
    return cards + h.images()


def iraf_fits_bytes(data: np.ndarray, ua: IrafHeader, *, provenance: str | None = None,
                    dtype=np.float32, keep_date: bool = False) -> bytes:
    """The bytes :func:`write_iraf_fits` writes: the header unit
    (:func:`_cards_bytes`) and the big-endian data, zero-padded to 2880
    bytes as ``fxf_updhdr`` pads it (``amovki (0, ...)``)."""
    arr = np.ascontiguousarray(np.asarray(data, dtype=dtype))
    bitpix = {np.dtype(np.float32): -32, np.dtype(np.float64): -64,
              np.dtype(np.int16): 16, np.dtype(np.int32): 32}[arr.dtype]
    head = _cards_bytes(iraf_header_cards(arr.shape, bitpix, ua, keep_date=keep_date),
                        provenance)
    body = arr.astype(arr.dtype.newbyteorder(">"), copy=False).tobytes()
    pad = (-len(body)) % 2880
    return head + body + b"\0" * pad


def write_iraf_fits(path: Path | str, data: np.ndarray, ua: IrafHeader, *,
                    provenance: str | None = None, dtype=np.float32,
                    keep_date: bool = False) -> None:
    """Write ``data`` with the FITS-kernel header built from ``ua``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = iraf_fits_bytes(data, ua, provenance=provenance, dtype=dtype, keep_date=keep_date)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(blob)
    tmp.replace(path)


def write_astropy_replaced(path: Path | str, template: bytes, data: np.ndarray) -> None:
    """``Spec1Dtools.resample2Dspec``'s output: the template FITS (an IRAF
    file) opened with astropy, its data replaced, ``writeto`` (astropy
    updates the structural cards and keeps every other card image)."""
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _afits.open(io.BytesIO(template)) as f:
        f[0].data = np.asarray(data, dtype=np.float32)
        tmp = path.with_name(path.name + ".tmp")
        f.writeto(tmp, output_verify="ignore", overwrite=True)
    tmp.replace(path)


def write_astropy_image(path: Path | str, data: np.ndarray) -> None:
    """``warp/Spec2Dtools.py:savefitsimage``: a bare astropy primary HDU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _afits.HDUList([_afits.PrimaryHDU(data)]).writeto(path, overwrite=True)


def read_ua(path: Path | str) -> tuple[np.ndarray, IrafHeader]:
    """Data and UA (header minus the kernel's structural cards and decanter's
    ``DECANTER`` card) of a FITS file."""
    with _afits.open(path) as h:
        data = np.array(h[0].data)
        hdr = h[0].header
        ua = IrafHeader()
        for c in hdr.cards:
            if _is_structural(c.keyword) or not c.image.strip() or c.keyword == "DECANTER":
                continue
            ua.append(Card80(c.keyword, c.image))
    return data, ua


__all__ = ["raw_ua", "write_iraf_fits", "write_astropy_image", "read_ua", "iraf_now",
           "iraf_header_cards", "FITS_ORIGIN", "iraf_fits_bytes", "write_astropy_replaced"]
