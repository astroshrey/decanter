"""Top-level reduce() / combine() API.

:func:`decanter.reduce` runs the full single-frame chain (image2d →
rectify → extract → wavelength). It composes the per-task functions
from :mod:`decanter.image2d`, :mod:`decanter.rectify`, :mod:`decanter.extract`,
and :mod:`decanter.wavelength` so each step is a pure function on
in-memory arrays. No waveshift correction is applied (waveshift is
relative across frames; meaningless for a single frame).

:func:`decanter.combine` is reserved for multi-frame stacks (eventual
S/N-weighted combination after cross-frame waveshift alignment). It
raises :class:`NotImplementedError` for now — transit-style per-frame
work uses :func:`reduce` in a loop and keeps each frame independent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter._reduction import Intermediates, OrderSpectrum, Reduction
from decanter.calib import Calibration
from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.extract.box_extract_1d import box_extract
from decanter.extract.psf_center import _infer_trace_x
from decanter.image2d import (
    cr_mask, fix_bad_pixels, flatfield_divide, sky_subtract, subtract_apscatter,
)
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.fsr import load as load_fsr
from decanter.rectify import rectify_orders
from decanter.utils.cosmic_ray import ndr_from_header
from decanter.waveshift.apply import apply_waveshift_one_order
from decanter.wavelength import dispcor_one_order, truncate_spectrum

_OBJNAME_BAD_CHARS: tuple[str, ...] = (" ", "'", "\"", "#", "/")


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in _OBJNAME_BAD_CHARS:
        name = name.replace(ch, "_")
    return name


def _is_abba(nodpos: str) -> bool:
    return "O" not in nodpos


def _read_input(
    obj: Path | NDArray,
    sky: Path | NDArray | None,
) -> tuple[NDArray, _astrofits.Header, NDArray | None, _astrofits.Header | None,
           Path | None, Path | None]:
    """Resolve obj / sky to (data, header) tuples. Accept Path or NDArray."""
    obj_path = obj if isinstance(obj, Path) else None
    sky_path = sky if isinstance(sky, Path) else None
    if isinstance(obj, Path):
        obj_data, obj_header = _fits.read_image(obj)
    else:
        obj_data, obj_header = np.asarray(obj), _astrofits.Header()
    if sky is None:
        sky_data, sky_header = None, None
    elif isinstance(sky, Path):
        sky_data, sky_header = _fits.read_image(sky)
    else:
        sky_data, sky_header = np.asarray(sky), _astrofits.Header()
    return obj_data, obj_header, sky_data, sky_header, obj_path, sky_path


def reduce(
    obj: Path | NDArray,
    calib: Calibration,
    *,
    sky: Path | NDArray | None = None,
    workdir: Path | None = None,
    save_intermediates: bool = False,
    config: Config | None = None,
    shift_wave: float = 0.0,
    check_calib: bool = True,
    mode: str = "warp",
) -> Reduction:
    """Single-frame reduction of one WINERED object frame.

    With ``sky`` given, the sky frame is subtracted first (the usual ABBA
    nod-subtracted reduction). With ``sky=None`` (the default) the object
    frame is reduced **on its own** — useful for reducing each nod position
    independently (A-only, then B-only, ...).

    .. warning::
        In the ``sky=None`` (no-subtraction) mode the additive components that
        nod subtraction normally removes are **retained** in the reduced
        spectrum: sky background emission (including the OH airglow lines),
        dark current, bias, and stray/scattered light. Cosmic-ray masking is
        also skipped (it needs the object/sky pair). Treat these products as
        raw A-position spectra, not background-subtracted science.

    No waveshift is applied by default (only meaningful across frames); pass
    ``shift_wave`` to inject an externally-measured cross-frame shift (e.g.
    WARP's ``waveShiftAdopted[i]``, in Å) for apples-to-apples parity against
    a multi-frame WARP reduction.

    Pipeline (step 1 is skipped when ``sky`` is None):
        1. image2d.sky_subtract  (obj - sky)
        2. image2d.cr_mask       (cosmic-ray detection)
        3. image2d.subtract_apscatter (scattered-light removal)
        4. image2d.flatfield_divide (master flat)
        5. image2d.fix_bad_pixels    (interpolate over CRs + static BPs)
        6. rectify.rectify_orders    (curved orders -> rectified strips)
        7. extract.box_extract       (per-order 1D box sum)
        8. wavelength.dispcor_one_order (apply id-file dispersion solution)
        9. wavelength.truncate_spectrum (per FSR cut, produces _VAC.fits)

    Sky path runs in parallel from step 4 onward when ``config.flag_skyemission``
    is set. Each task is a pure in-memory function — disk I/O is owned by
    this orchestrator.

    Args:
        obj: object frame — Path to a FITS file or a 2D ``ndarray``.
        calib: :class:`Calibration` bundle (required). Use
            :meth:`Calibration.from_dir` to auto-discover it from a
            calibration-set directory or a WARP reduction root.
        sky: sky frame to subtract, or None (default) for a no-subtraction
            A-only reduction — see the warning above.
        workdir: optional output directory. None means in-memory only
            (the returned :class:`Reduction` is the entire output).
        save_intermediates: when True, the returned :class:`Reduction`'s
            ``intermediates`` field is populated with the per-stage 2D /
            per-order arrays, and (if ``workdir`` is also set) those
            intermediates are written to disk alongside the final
            spectra following WARP's suffix conventions.
        config: pipeline configuration. Defaults to :class:`Config`
            (full-spectrum + sky-emission, all 26 orders).
        shift_wave: wavelength shift in Å applied at the s11 truncate
            step (WARP: ``scopy`` with the adopted cross-frame shift).
            Obj only — WARP never waveshifts the sky path. Default 0.0
            (single-frame semantics).
        check_calib: verify the calibration set matches the object frame's
            mode/slit/setting before reducing (default True). Raises
            :class:`~decanter.calib.CalibrationMismatch` on a mismatch; set
            False to bypass (e.g. for a deliberately re-purposed set).
        mode: reduction recipe. Only ``"warp"`` (bit-for-bit WARP clone) is
            implemented today; the argument exists so future recipes (e.g.
            a ``"default"`` with decanter's own improved steps) are a
            drop-in opt-in. Any other value raises ``ValueError``.

    Returns:
        :class:`Reduction` carrying per-(fsr_cut, order) calibrated 1D
        spectra. Per-frame products keyed as ``(fsr_cut, order)`` so a
        caller can grab a specific order with ``r.obj[(1.05, 163)]``.
    """
    if mode != "warp":
        raise ValueError(
            f"mode={mode!r} is not implemented; only 'warp' (the WARP-clone "
            f"recipe) is available today")
    cfg = config or Config()
    obj_data, obj_header, sky_data, sky_header, obj_path, sky_path = _read_input(
        obj, sky,
    )
    if check_calib:
        calib.assert_matches(obj_header)
    raw_objname = str(headers.get(obj_header, "OBJECT", default=obj_path.stem
                                  if obj_path else "frame"))
    objname = _sanitize_objname(raw_objname)
    apset_multi = ApertureSet.load(calib.apdb_multihole)
    apset_apsc = ApertureSet.load(calib.apdb_apsc)
    static_bp, _ = _fits.read_image(calib.static_bp_mask)
    flat, _ = _fits.read_image(calib.flat)

    inter = Intermediates() if save_intermediates else Intermediates()
    if save_intermediates:
        inter.obj_raw = obj_data

    # --- s01 sky subtract -------------------------------------------------
    diff = sky_subtract(obj_data, sky_data) if sky_data is not None else obj_data.copy()
    if save_intermediates:
        inter.obj_s = diff

    # --- s02 cosmic-ray mask ----------------------------------------------
    if cfg.flag_bpmask and sky_data is not None:
        nodpos = str(headers.get(obj_header, "NODPOS", default="A1"))
        mask = cr_mask(
            diff, obj_data, sky_data, apset_multi, static_bp,
            nodpos=nodpos,
            ndr_obj=ndr_from_header(obj_header),
            ndr_sky=ndr_from_header(sky_header) if sky_header is not None else 1,
            config=cfg,
        )
    else:
        mask = np.zeros_like(static_bp, dtype=np.int16)
    if save_intermediates:
        inter.cr_mask = mask
    combined_mask = mask.astype(bool) | static_bp.astype(bool)

    # --- s03 apscatter ----------------------------------------------------
    if cfg.flag_apscatter:
        obj_ssc, scatter = subtract_apscatter(diff, apset_apsc)
    else:
        obj_ssc, scatter = diff, np.zeros_like(diff)
    if save_intermediates:
        inter.obj_ssc = obj_ssc
        inter.scatter_model = scatter

    # --- s04 flatfield divide --------------------------------------------
    obj_sscf = flatfield_divide(obj_ssc, flat)
    if save_intermediates:
        inter.obj_sscf = obj_sscf

    # --- s05 fixpix -------------------------------------------------------
    obj_sscfm = fix_bad_pixels(obj_sscf, combined_mask)
    if save_intermediates:
        inter.obj_sscfm = obj_sscfm

    # Sky 2D path (when requested)
    sky_fm = None
    if cfg.flag_skyemission and sky_data is not None:
        sky_f = flatfield_divide(sky_data, flat)
        sky_fm = fix_bad_pixels(sky_f, combined_mask)
        if save_intermediates:
            inter.sky_f = sky_f
            inter.sky_fm = sky_fm

    # --- s06 rectify per order -------------------------------------------
    orders = (
        apset_multi.echelle_orders
        if (cfg.reduce_full_data or not cfg.selected_orders)
        else tuple(m for m in apset_multi.echelle_orders if m in cfg.selected_orders)
    )
    # Comp file's CDELT1 is dy.
    comp_header = _fits.read_image(calib.comp)[1]
    dy = float(comp_header.get("CDELT1", 0.5))

    strips_obj = rectify_orders(
        obj_sscfm, apset_multi,
        fc_dir=calib.fc_dir, fc_refname=calib.fc_refname, dy=dy, orders=orders,
    )
    if save_intermediates:
        inter.strips_obj = {m: s.data for m, s in strips_obj.items()}

    strips_sky_arrays: dict[int, NDArray] = {}
    strips_sky_full = None
    if sky_fm is not None:
        strips_sky_full = rectify_orders(
            sky_fm, apset_multi,
            fc_dir=calib.fc_dir, fc_refname=calib.fc_refname, dy=dy, orders=orders,
        )
        strips_sky_arrays = {m: s.data for m, s in strips_sky_full.items()}
        if save_intermediates:
            inter.strips_sky = strips_sky_arrays

    # --- s07/s08 trace + 1D extract --------------------------------------
    trans_apdbs = calib.trans_apdbs or {}
    obj_1d: dict[int, NDArray] = {}
    sky_1d: dict[int, NDArray] = {}
    for m, strip in strips_obj.items():
        strip_arr = strip.data
        # Read trace + ap-window from the WARP-supplied trans aperture DB
        # when available; otherwise infer the trace via argmax+legfit and
        # use the multihole aperture window.
        if m in trans_apdbs:
            trans_set = ApertureSet.load(
                trans_apdbs[m], array_length=strip_arr.shape[0]
            )
            if m in trans_set.apertures:
                trans_ap = trans_set.apertures[m]
                ap_low, ap_high = float(trans_ap.entry.low), float(trans_ap.entry.high)
                trace_x = trans_ap.trace_x
            else:
                ap = apset_multi.apertures[m]
                ap_low, ap_high = float(ap.entry.low), float(ap.entry.high)
                trace_x = _infer_trace_x(strip_arr)
        else:
            ap = apset_multi.apertures[m]
            ap_low, ap_high = float(ap.entry.low), float(ap.entry.high)
            trace_x = _infer_trace_x(strip_arr)
        obj_1d[m] = box_extract(strip_arr, trace_x, ap_low=ap_low, ap_high=ap_high)
        if m in strips_sky_arrays:
            sky_1d[m] = box_extract(
                strips_sky_arrays[m], trace_x, ap_low=ap_low, ap_high=ap_high,
            )
    if save_intermediates:
        inter.spectra_1d = dict(obj_1d)
        inter.sky_1d = dict(sky_1d)

    # --- s11 truncate (shift=shift_wave) -> 4095 px, matches WARP -----------
    # Even with no cross-frame shift, WARP runs scopy(rebin=YES) to truncate
    # from the 5628-px rectified strip to the [1, 2048] Å range (4095 px,
    # matching the comp file's pixel count). That sets the input length for
    # dispcor; without this step dispcor's output is 5628 px and downstream
    # FSR truncate shapes diverge from WARP.
    obj_truncated: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    sky_truncated: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    for m, spec in obj_1d.items():
        strip = strips_obj[m]
        h = _astrofits.Header()
        h["CRVAL1"] = strip.lambda_min
        h["CDELT1"] = strip.dy
        h["CRPIX1"] = 1.0
        obj_truncated[m] = apply_waveshift_one_order(spec, h, shift_wave=shift_wave)
        if m in sky_1d:
            # WARP never waveshifts the sky path (truncate only) — see
            # bench_sky_multi_order.py; sky always gets shift 0.
            sky_truncated[m] = apply_waveshift_one_order(
                sky_1d[m], h, shift_wave=0.0,
            )

    # --- s12 dispcor per order -------------------------------------------
    obj_dispcor: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    sky_dispcor_d: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    for m, (trunc_data, trunc_header) in obj_truncated.items():
        id_path = Path(calib.id_dir) / f"id{calib.id_refname}.{m:04d}"
        if not id_path.exists():
            continue
        obj_dispcor[m] = dispcor_one_order(trunc_data, trunc_header, id_path)
        if m in sky_truncated:
            sky_data_t, sky_header_t = sky_truncated[m]
            sky_dispcor_d[m] = dispcor_one_order(sky_data_t, sky_header_t, id_path)
    if save_intermediates:
        inter.spectra_dispcor = {m: a for m, (a, _) in obj_dispcor.items()}
        inter.sky_dispcor = {m: a for m, (a, _) in sky_dispcor_d.items()}

    # --- s13 FSR truncate -> final per-cut per-order spectra -------------
    fsr_table = load_fsr(calib.fsr_table)
    obj_out: dict[tuple[float, int], OrderSpectrum] = {}
    sky_out: dict[tuple[float, int], OrderSpectrum] = {}

    def _fsr_cut(data: NDArray, hdr: _astrofits.Header, m: int, cut: float
                 ) -> OrderSpectrum | None:
        if m not in fsr_table:
            return None
        fsr_lo = fsr_table[m].lambda_min
        fsr_hi = fsr_table[m].lambda_max
        center = (fsr_lo + fsr_hi) / 2.0
        lo = center - (center - fsr_lo) * float(cut)
        hi = center + (fsr_hi - center) * float(cut)
        sliced, wcs = truncate_spectrum(
            data, float(hdr["CRVAL1"]), float(hdr["CDELT1"]), lo, hi,
        )
        if sliced.size == 0:
            return None
        return OrderSpectrum(
            order=m, fsr_cut=cut, flux=sliced.astype(np.float32),
            crval1=float(wcs["crval1"]),
            cdelt1=float(wcs["cdelt1"]),
            crpix1=float(wcs["crpix1"]),
        )

    for cut in cfg.cutrange_list:
        for m, (data, hdr) in obj_dispcor.items():
            spec = _fsr_cut(data, hdr, m, cut)
            if spec is not None:
                obj_out[(cut, m)] = spec
        for m, (data, hdr) in sky_dispcor_d.items():
            spec = _fsr_cut(data, hdr, m, cut)
            if spec is not None:
                sky_out[(cut, m)] = spec

    r = Reduction(
        obj_name=objname,
        obj_path=obj_path,
        sky_path=sky_path,
        obj=obj_out,
        sky=sky_out if cfg.flag_skyemission else None,
        intermediates=inter,
    )

    if workdir is not None:
        r.write_to(workdir, save_intermediates=save_intermediates)

    return r


def combine(*args: Any, **kwargs: Any) -> None:
    """Stub for multi-frame SNR-weighted combination after cross-frame waveshift.

    Raises :class:`NotImplementedError` for now. For transit-style
    per-frame analysis, loop :func:`reduce` over your frame list and
    keep each frame's output separate.
    """
    raise NotImplementedError(
        "decanter.combine() is not yet implemented. For per-frame analysis "
        "(e.g. transit time series), loop decanter.reduce() over your frame "
        "pairs — each call produces an independent Reduction."
    )
