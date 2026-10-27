"""What an IRAF task reads from a REAL (``BITPIX = -32``) FITS image.

IRAF's FITS kernel turns on IEEE input mapping whenever it opens an image's
pixels for reading (``fxf_opix``, sys/imio/iki/fxf/fxfopix.x:61-68: ``case
READ_ONLY, READ_WRITE, WRITE_ONLY: ieesnanr (0.0); ieemapr (YES, NO)``), and
every buffer of REAL pixels is unpacked by ``ieevupkr`` (fxfupk.x:61,
zfiofxf.x:190; sys/osb/ieeer.x, the portable version, compiled into
``libvops.a`` on this Mac).  With mapping on, ``ieevupkr`` replaces

* every value whose exponent field is 0 -- ``-0.0`` and the float32
  subnormals, positive and negative -- by ``+0.0`` (``native[i] = 0``), and
* every value whose exponent field is all ones -- ``+-Inf`` and every NaN --
  by the NaN replacement value, which ``fxf_opix`` set to ``0.0``;

``+0.0`` and the normal numbers pass unchanged.  Output is never mapped
(``ieemapr (YES, NO)``), so a task *writes* subnormals and ``-0.0`` that the
next task reads as ``+0.0``.

In WARP the only files that hold such values are the rectified strips that
``transform`` writes (``6-OBJ_transform``, ``4-SKY_transform``: subnormals and
``-0.0`` off the illuminated order); ``apall`` reads them mapped, while
WARP's own Python (the PSF center search, ``make_slit_profile``,
``resample2Dspec``: astropy) reads them as written.  decanter applies
:func:`imio_read_real` wherever it runs an IRAF task on an image it holds in
memory, and nowhere else.

Checked against the compiled routine (``scripts/iraf_oracle/ieee_oracle.py``:
``ieesmapr`` + ``ieesnanr`` + ``ieevupkr`` of this Mac's ``libvops.a`` on
every special class and random bit patterns).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EXP = np.uint32(0x7F800000)        # ieeer.x NaNmask (exponent field)
_MANT = np.uint32(0x007FFFFF)


def imio_mapped(a: NDArray) -> NDArray[np.bool_]:
    """Where IRAF's input mapping changes a REAL value: ``-0.0``, the
    subnormals, ``+-Inf`` and NaN (``+0.0`` is unchanged)."""
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    e = bits & _EXP
    return ((e == 0) & (bits != 0)) | (e == _EXP)


def imio_read_real(a, *, keep_nan: bool = False) -> NDArray[np.float32]:
    """The pixels an IRAF task reads from a REAL image holding ``a``.

    Returns ``a`` as float32 with every value of exponent field 0 (``-0.0``,
    subnormals) and every ``+-Inf``/NaN replaced by ``+0.0``
    (``ieevupkr`` with ``mapin = YES`` and the NaN value ``0.0``); the input
    itself is never modified, and when nothing is mapped the float32 array
    is returned as is (no copy).

    ``keep_nan`` leaves NaN alone (``+-Inf`` still become 0): for the
    vectorised extraction's band-evaluated strips, whose NaN pixels are not
    data but mark the pixels the band rectification did not evaluate
    (``decanter.fast.rectify``); reading one must still make the extraction
    non-finite so the order is rectified in full.  A strip IRAF's
    ``transform`` writes never holds a NaN (its input was read mapped).

    IRAF equivalent: ``ieevupkr`` (sys/osb/ieeer.x:95-150) as enabled by
    ``fxf_opix`` (sys/imio/iki/fxf/fxfopix.x:61-68).
    """
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    bits = a.view(np.uint32)
    e = bits & _EXP
    bad = (e == 0) & (bits != 0)
    if keep_nan:
        bad |= (e == _EXP) & ((bits & _MANT) == 0)
    else:
        bad |= e == _EXP
    if not bad.any():
        return a
    out = a.copy()
    out[bad] = np.float32(0.0)
    return out
