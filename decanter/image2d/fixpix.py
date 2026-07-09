"""Stage 5 — bad-pixel interpolation.

WARP equivalent: ``warp/badpixmask.py:pyfixpix`` (line 434) wrapping
``iraf.fixpix(image, mask, linterp="INDEF", cinterp="INDEF")``. The
``INDEF`` defaults mean: for each bad-pixel run, interpolate along
whichever axis (row or column) is narrower.

WARP combines the static bad-pixel mask and the per-frame cosmic-ray
mask into ``obj_s_maskflat_list[i]`` via
``iraf.imarith(mask_file, "+", obj_s_mask_list[i], obj_s_maskflat_list[i])``
(``Warp_sci.py:352``). We do the equivalent logical-or here when
``flag_bpmask`` is set; otherwise just the static mask.

Output suffix: ``_sscfm``. The output file is a hard parity target —
WARP saves it under ``TOI2109_NO{i}/intermediate_files/OBJ/4-OBJ_mask/``
even on default reductions (without ``-s``), so it's the first stage
where bit-for-bit comparison against WARP is possible without re-running
the WARP pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits

from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.fixpix import fixpix


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def fix_bad_pixels(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pure task function: interpolate over bad pixels in one 2D frame.

    Mirrors ``iraf.fixpix(image, mask, linterp=INDEF, cinterp=INDEF)``.
    Non-finite pixels in ``data`` are added to the mask before
    interpolation so the output is fully finite.

    Returns the interpolated frame at the input dtype.
    """
    full_mask = mask.astype(bool, copy=True)
    full_mask |= ~np.isfinite(data)
    clean = np.where(np.isfinite(data), data, 0.0).astype(np.float32, copy=False)
    fixed = fixpix(clean, full_mask)
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
            the per-frame CR mask is combined with the static mask
            (True, default) or only the static mask is used (False —
            matches WARP's behavior at ``Warp_sci.py:358-361``).
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
    static_mask = static_bp.astype(bool)
    pairs = parse_listfile(listfile)

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        # Build the combined mask once (static + per-frame CR). The SKY
        # path uses the *same* combined mask — IRAF apscatter doesn't
        # run a CR detection on the sky frame, so WARP reuses the
        # object's ``obj_s_maskflat_list[i]`` (see ``Warp_sci.py:370``).
        mask_base = static_mask.copy()
        if config.flag_bpmask:
            cr_path = workdir / f"mask_{objname}_NO{i}_s.fits"
            cr_data, _ = _fits.read_image(cr_path)
            mask_base |= cr_data.astype(bool)

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
