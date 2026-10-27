"""Stage 5 — bad-pixel interpolation (bit-identical to WARP).

WARP equivalent: ``warp/badpixmask.py:pyfixpix`` (line 434):
``iraf.imcopy(input, output)`` then
``iraf.fixpix(output, mask, linterp="INDEF", cinterp="INDEF")``. The
interpolation kernel is :func:`decanter.utils.fixpix.fixpix`, a bit-exact
port of IRAF ``pkg/proto/t_fixpix.x`` + ``pkg/xtools/fixpix/xtfixpix.x``
(``xt_fpinit``) + ``xtfp.gx`` (``xt_fpsr``): each bad pixel is linearly
interpolated along the narrower of its line / column bad runs (ties: along
the line), in REAL arithmetic with one fused multiply-add.

Mask. With ``flag_bpmask`` WARP builds
``maskflat_<obj>_NO<i>_s = imarith(mask_file, "+", mask_<obj>_NO<i>_s)``
(``Warp_sci.py:352``): the static mask (float, values 0/1/2) plus the
per-frame cosmic-ray mask (short 0/1), a REAL image with values 0..3.
Without ``flag_bpmask`` it passes the static ``mask_file`` alone
(``Warp_sci.py:361``). IRAF reads a mask image as INTEGER (truncating) and
treats values > 0 as bad (``ytpmmap.x:yt_pmimmap``); the values do not
otherwise matter with INDEF interpolation codes. The sky frame
(``_skyNO<i>_f`` -> ``_fm``) uses the *same* mask as the object
(``Warp_sci.py:370/372``).

Output suffix: ``_sscfm`` (object) and ``_skyNO<i>_fm`` (sky), saved by WARP
under ``<OBJ>_NO<i>/intermediate_files/{OBJ/4-OBJ_mask,SKY/2-SKY_mask}/``.
Parity: 0 differing float32 pixels vs WARP on TOI2109_decanterref NO1
(object and sky) and on all AUMic / GJ3090 object frames; the AUMic /
GJ3090 sky frames (no WARP ``_fm`` saved) match ``iraf.fixpix`` run
through pyraf. See ``docs/BITEXACT.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.fixpix import fixpix, iraf_bad_pixels


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def warp_maskflat(static_mask: np.ndarray, cr_mask: np.ndarray | None) -> np.ndarray:
    """The mask WARP hands to fixpix: ``imarith(mask_file, "+", cr_mask)``.

    WARP equivalent: ``Warp_sci.py:352`` (``maskflat_*``; REAL sum of the
    static and cosmic-ray masks), or the static ``mask_file`` alone when
    there is no CR mask (``Warp_sci.py:361``).
    """
    static = np.asarray(static_mask, dtype=np.float32)
    if cr_mask is None:
        return static
    return static + np.asarray(cr_mask, dtype=np.float32)


def fix_bad_pixels(data: np.ndarray, mask: np.ndarray, *,
                   mask_nonfinite: bool = True) -> np.ndarray:
    """Pure task function: interpolate over bad pixels in one 2D frame.

    Bit-exact ``iraf.fixpix(image, mask, linterp=INDEF, cinterp=INDEF)``
    (see :func:`decanter.utils.fixpix.fixpix`). ``mask`` is boolean or mask
    image values (IRAF semantics: truncated to integer, > 0 is bad).

    ``mask_nonfinite`` (default True) additionally treats non-finite input
    pixels as bad so the output is finite. IRAF does not do this, but WARP
    never feeds it non-finite pixels (IRAF ``imarith`` writes 0 on
    divide-by-zero, as does :func:`flatfield_divide`), so for finite input
    it is a no-op and the result is WARP's bit for bit. Pass False for
    strict IRAF semantics on non-finite input.

    Returns the interpolated frame at the input dtype (computed as REAL).
    """
    bad = iraf_bad_pixels(mask)
    work = np.asarray(data, dtype=np.float32)
    if mask_nonfinite:
        nonfinite = ~np.isfinite(work)
        if nonfinite.any():
            bad |= nonfinite
            work = np.where(nonfinite, np.float32(0.0), work)
    fixed = fixpix(work, bad)
    return fixed.astype(data.dtype, copy=False)


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    static_bp_mask_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Interpolate over bad pixels in every ``_sscf.fits`` frame.

    Args:
        config: pipeline configuration. ``flag_bpmask`` controls whether
            the per-frame CR mask is added to the static mask (True,
            default; WARP's ``maskflat_*``) or only the static mask is used
            (False) — ``Warp_sci.py:352-372``.
        workdir: reads ``{obj}_NO{i}_sscf.fits`` (s04 output);
            reads ``mask_{obj}_NO{i}_s.fits`` (s02 output) when
            ``flag_bpmask``;
            writes ``{obj}_NO{i}_sscfm.fits``.
        listfile: WARP-style input list.
        static_bp_mask_path: path to the static bad-pixel mask FITS.
            Required.

    Notes:
        Non-finite pixels in the flat-fielded frame (from divide-by-zero
        in s04) are added to the mask before interpolation — without
        this, the interpolated values themselves can be NaN.
    """
    if static_bp_mask_path is None:
        raise ValueError(
            "s05_badpix_interp requires static_bp_mask_path; the calibration "
            "loader (s00) will provide this automatically once wired in."
        )

    static_bp, _ = _fits.read_image(static_bp_mask_path)
    pairs = parse_listfile(listfile)

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        # Build the mask once (WARP's ``maskflat_*`` = static + per-frame
        # CR, Warp_sci.py:352). The SKY path uses the *same* mask — no CR
        # detection runs on the sky frame, so WARP reuses the object's
        # ``obj_s_maskflat_list[i]`` (Warp_sci.py:370).
        cr_data = None
        if config.flag_bpmask:
            cr_path = workdir / f"mask_{objname}_NO{i}_s.fits"
            cr_data, _ = _fits.read_image(cr_path)
        mask_base = warp_maskflat(static_bp, cr_data)

        # --- OBJ path ----------------------------------------------------------
        data, header = _fits.read_image(workdir / f"{objname}_NO{i}_sscf.fits")
        out = fix_bad_pixels(data, mask_base)

        out_header = header.copy()
        out_header["BP_MASK"] = (
            str(Path(static_bp_mask_path).name),
            "Static bad-pixel mask applied (s05)",
        )
        _fits.write_image(workdir / f"{objname}_NO{i}_sscfm.fits", out, out_header,
                          overwrite=True)

        # --- SKY path (mirrors WARP ``pyfixpix(sky_f, sky_fm, ...)``) ----------
        sky_in_path = workdir / f"{objname}_skyNO{i}_f.fits"
        if config.flag_skyemission and sky_in_path.exists():
            sky_data, sky_header = _fits.read_image(sky_in_path)
            sky_out = fix_bad_pixels(sky_data, mask_base)
            sky_out_header = sky_header.copy()
            sky_out_header["BP_MASK"] = (
                str(Path(static_bp_mask_path).name),
                "Static bad-pixel mask applied (s05, sky)",
            )
            _fits.write_image(workdir / f"{objname}_skyNO{i}_fm.fits", sky_out,
                              sky_out_header, overwrite=True)
