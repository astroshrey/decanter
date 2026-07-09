"""Stage 2 — cosmic-ray detection (the main Phase-1 speedup target).

WARP equivalent: ``warp/badpixmask.py:cosmicRayMask`` (lines 115-296).
The detection algorithm is ported in :mod:`decanter.utils.cosmic_ray`;
this module owns the FITS I/O wiring and the ABBA-from-NODPOS check.

Output: a 2048×2048 int16 mask FITS per frame (``mask_*.fits``).

Parity target: ≥99% pixel agreement with WARP **and** max-connected-
cluster of disagreement ≤ 50 px. See PLAN_FULL.md §Validation
tolerance table. The largest-cluster guard catches localized misses
that the percentage gate would pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.cosmic_ray import detect_cosmic_rays, ndr_from_header

# Slit-coordinate window: matches WARP's hardcoded apmaskArray call
# (warp/badpixmask.py:174 — `apset.apmaskArray(lowlim=-30, upplim=30)`).
_SLIT_WINDOW: tuple[float, float] = (-30.0, 30.0)


def _sanitize_objname(raw: str) -> str:
    """Match s01's filename-sanitization rule."""
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def _is_abba(nodpos: str) -> bool:
    """Translate a NODPOS header value into the ABBA / stare-mode flag.

    WARP equivalent: ``Warp_sci.py:339`` — ``"O" not in NODPOS`` means ABBA.
    Examples: ``"A1"``/``"B1"`` → ABBA (True); ``"O"``/``"O1"`` → stare (False).
    """
    return "O" not in nodpos


def cr_mask(
    diff: np.ndarray,
    obj_raw: np.ndarray,
    sky_raw: np.ndarray,
    apset: ApertureSet,
    static_bp: np.ndarray,
    *,
    nodpos: str,
    ndr_obj: int,
    ndr_sky: int,
    config: Config,
) -> np.ndarray:
    """Pure task function: cosmic-ray mask for one (obj, sky) frame pair.

    Wraps :func:`decanter.utils.cosmic_ray.detect_cosmic_rays` with the
    aperture-mask + slit-coord arrays from ``apset`` and the WARP-fixed
    slit window (``-30..30`` slit coordinate units).

    Returns:
        2D int16 mask, 1 where CR detected, 0 elsewhere.
    """
    apmask = apset.apmask_array(low_lim=_SLIT_WINDOW[0], upp_lim=_SLIT_WINDOW[1])
    slitcoord = apset.slitcoord_array(low_lim=_SLIT_WINDOW[0], upp_lim=_SLIT_WINDOW[1])
    static_bp_int = static_bp.astype(np.int16) if static_bp.dtype != np.int16 else static_bp
    result = detect_cosmic_rays(
        diff=diff, raw1=obj_raw, raw2=sky_raw,
        apmask=apmask, slitcoord=slitcoord, static_bp=static_bp_int,
        ndr1=ndr_obj, ndr2=ndr_sky, abba=_is_abba(nodpos),
        echelle_orders=apset.echelle_orders, array_length=apset.array_length,
        xlim1=_SLIT_WINDOW[0], xlim2=_SLIT_WINDOW[1],
        threshold=config.CR_threshold, max_sigma=config.CR_max_sigma,
        varatio=config.CR_var_ratio, slitposratio=config.CR_slitpos_ratio,
        fixsigma=config.CR_fix_sigma,
    )
    return result.mask


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    static_bp_mask_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Detect cosmic rays per frame, write a binary mask FITS.

    Args:
        config: pipeline configuration (gated on ``flag_bpmask``; uses
            ``CR_threshold``, ``CR_var_ratio``, ``CR_slitpos_ratio``,
            ``CR_max_sigma``, ``CR_fix_sigma``).
        workdir: working directory; reads ``{obj}_NO{i}_s.fits`` from
            s01's output plus the raw frames named in ``listfile``;
            writes ``mask_{obj}_NO{i}_s.fits``.
        listfile: WARP-style input list.
        apdb_path: path to the IRAF aperture database file (e.g.
            ``database/apflat_HIRESY_20170727_m``). Required.
        static_bp_mask_path: path to the static bad-pixel mask FITS.
            Required.

    Notes:
        Phase 1 takes ``apdb_path`` and ``static_bp_mask_path`` as
        explicit kwargs because s00 (calibration loader) isn't yet
        wired in. Once s00 lands, the orchestrator will pre-load these
        and stash them on a ``CalibBundle`` so this signature can
        shrink.

        Stage is a no-op when ``config.flag_bpmask`` is False (mirrors
        WARP's ``if conf.flag_bpmask`` gate at ``Warp_sci.py:337``).
    """
    if not config.flag_bpmask:
        return
    if apdb_path is None or static_bp_mask_path is None:
        raise ValueError(
            "s02_cosmic_ray requires apdb_path and static_bp_mask_path until "
            "s00 (calib loader) is implemented; see PLAN_FULL.md."
        )

    apset = ApertureSet.load(apdb_path)
    static_bp, _ = _fits.read_image(static_bp_mask_path)

    pairs = parse_listfile(listfile)

    for i, pair in enumerate(pairs, start=1):
        obj_data, obj_header = _fits.read_image(workdir / pair.object_name)
        sky_data, sky_header = _fits.read_image(workdir / pair.sky_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))
        diff_data, _ = _fits.read_image(workdir / f"{objname}_NO{i}_s.fits")

        mask = cr_mask(
            diff_data, obj_data, sky_data, apset, static_bp,
            nodpos=str(headers.get(obj_header, "NODPOS", default="A1")),
            ndr_obj=ndr_from_header(obj_header),
            ndr_sky=ndr_from_header(sky_header),
            config=config,
        )

        out_path = workdir / f"mask_{objname}_NO{i}_s.fits"
        _fits.write_image(out_path, mask, obj_header, overwrite=True)
