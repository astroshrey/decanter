"""Top-level reduce() / combine() API.

:func:`decanter.reduce` runs the full single-frame chain (image2d →
rectify → extract → wavelength). It composes the per-task functions
from :mod:`decanter.image2d`, :mod:`decanter.rectify`, :mod:`decanter.extract`,
and :mod:`decanter.wavelength` so each step is a pure function on
in-memory arrays. No waveshift correction is applied (waveshift is
relative across frames; meaningless for a single frame).

:func:`decanter.reduce_many` preserves WARP's relative cross-frame waveshift
and can optionally layer the physical telluric/OH wavecal on its output.
:func:`decanter.combine` S/N-weights an aligned set into a master spectrum.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter._reduction import Intermediates, OrderSpectrum, Reduction
from decanter.calib import Calibration
from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.extract.aperture_solve import solve_frame_aperture
from decanter.extract.box_extract_1d import box_extract
from decanter.extract.optimal_extract import optimal_extract
from decanter.image2d import (
    cr_mask, fix_bad_pixels, flatfield_divide, sky_subtract, subtract_apscatter,
)
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.fsr import load as load_fsr
from decanter.rectify import rectify_orders
from decanter.utils.cosmic_ray import ndr_from_header
from decanter.waveshift.apply import apply_waveshift_one_order
from decanter.waveshift.measure import waveshift_clip, waveshift_one_order
from decanter.wavelength import dispcor_one_order, truncate_spectrum

if TYPE_CHECKING:
    from decanter.wavecal.config import WavecalConfig
    from decanter.wavecal.solution import WavecalSolution

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
    subtract_background: bool = False,
    extract: str = "box",
    optimal_profile: dict[int, tuple] | None = None,
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
        subtract_background: subtract a local background during extraction
            (default False). For each order, a background level is estimated
            per dispersion row from the slit flanking the aperture and
            subtracted from the object box sum (IRAF apall's ``background``
            option). This suppresses the additive slit background — notably
            the OH airglow lines — so it is the way to clean up an A-only
            (``sky=None``) spectrum, or to knock down OH residuals left by an
            imperfect A-B subtraction. Applied to the object path only.
        mode: reduction recipe. Only ``"warp"`` (bit-for-bit WARP clone) is
            implemented today; the argument exists so future recipes (e.g.
            a ``"default"`` with decanter's own improved steps) are a
            drop-in opt-in. Any other value raises ``ValueError``.
        extract: 1D extraction method. ``"box"`` (default) is the WARP-parity
            box sum; ``"optimal"`` is Horne (1986) profile-weighted extraction
            (higher SNR, an algorithmic upgrade over WARP — see
            :mod:`decanter.extract.optimal_extract`).

    Returns:
        :class:`Reduction` carrying per-(fsr_cut, order) calibrated 1D
        spectra. Per-frame products keyed as ``(fsr_cut, order)`` so a
        caller can grab a specific order with ``r.obj[(1.05, 163)]``.
    """
    if mode != "warp":
        raise ValueError(
            f"mode={mode!r} is not implemented; only 'warp' (the WARP-clone "
            f"recipe) is available today")
    if extract not in ("box", "optimal"):
        raise ValueError(f"extract={extract!r} must be 'box' or 'optimal'")
    cfg = config or Config()
    obj_data, obj_header, sky_data, sky_header, obj_path, sky_path = _read_input(
        obj, sky,
    )
    if check_calib:
        calib.assert_matches(obj_header)
    raw_objname = str(headers.get(obj_header, "OBJECT", default=obj_path.stem
                                  if obj_path else "frame"))
    objname = _sanitize_objname(raw_objname)
    # Nod pattern drives both the CR mask and the center-search O-position
    # rejection; resolve it once here so both the sky and A-only (sky=None)
    # paths have it defined.
    nodpos = str(headers.get(obj_header, "NODPOS", default="A1"))
    apset_multi = ApertureSet.load(calib.apdb_multihole)
    apset_apsc = ApertureSet.load(calib.apdb_apsc)
    static_bp, _ = _fits.read_image(calib.static_bp_mask)
    flat, _ = _fits.read_image(calib.flat)

    inter = Intermediates()
    if save_intermediates:
        inter.obj_raw = obj_data

    # --- s01 sky subtract -------------------------------------------------
    diff = sky_subtract(obj_data, sky_data) if sky_data is not None else obj_data.copy()
    if save_intermediates:
        inter.obj_s = diff

    # --- s02 cosmic-ray mask ----------------------------------------------
    if cfg.flag_bpmask and sky_data is not None:
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

    # When per-frame WARP trans aperture DBs are supplied (a WARP reduction
    # root), lock the extraction to WARP's exact per-frame trace for bit
    # parity. Otherwise use the WARP-independent path: the multihole trans
    # reference trace + a frame-constant aperture solved by center search
    # (see extract.aperture_solve — this mirrors WARP's normal science path).
    frame_ap = None
    if not trans_apdbs:
        ref_trans = calib.ref_trans_apdbs
        if not ref_trans:
            raise ValueError(
                "calibration set has no multihole trans reference apertures "
                "(ap{aptrans}_{m}trans) and no per-frame trans DBs were found; "
                "cannot determine the extraction trace."
            )
        frame_ap = solve_frame_aperture(
            strips_obj, ref_trans, abba=_is_abba(nodpos),
        )

    for m, strip in strips_obj.items():
        strip_arr = strip.data
        if m in trans_apdbs:
            trans_set = ApertureSet.load(
                trans_apdbs[m], array_length=strip_arr.shape[0]
            )
            if m in trans_set.apertures:
                trans_ap = trans_set.apertures[m]
                ap_low, ap_high = float(trans_ap.entry.low), float(trans_ap.entry.high)
                trace_x = trans_ap.trace_x
            else:
                sub = solve_frame_aperture(
                    {m: strip}, calib.ref_trans_apdbs or {}, abba=_is_abba(nodpos),
                )
                trace_x = sub.traces[m]
                ap_low, ap_high = sub.ap_low, sub.ap_high
        else:
            trace_x = frame_ap.traces[m]
            ap_low, ap_high = frame_ap.ap_low, frame_ap.ap_high
        if extract == "optimal":
            prof = optimal_profile.get(m) if optimal_profile else None
            obj_1d[m] = optimal_extract(strip_arr, trace_x,
                                        ap_low=ap_low, ap_high=ap_high, profile=prof)
            if m in strips_sky_arrays:
                sky_1d[m] = optimal_extract(strips_sky_arrays[m], trace_x,
                                            ap_low=ap_low, ap_high=ap_high, profile=prof)
        else:
            obj_1d[m] = box_extract(strip_arr, trace_x, ap_low=ap_low, ap_high=ap_high,
                                    subtract_background=subtract_background)
            if m in strips_sky_arrays:
                sky_1d[m] = box_extract(
                    strips_sky_arrays[m], trace_x, ap_low=ap_low, ap_high=ap_high,
                )
    if save_intermediates:
        inter.spectra_1d = dict(obj_1d)
        inter.sky_1d = dict(sky_1d)

    strips_lambda = {m: (strips_obj[m].lambda_min, strips_obj[m].dy) for m in obj_1d}
    if save_intermediates:
        inter.strip_wcs = strips_lambda

    # Carried on the reduction so a multi-frame analysis can read mid-times,
    # pointing and instrument configuration off the reduced products.
    meta = headers.frame_meta(obj_header)
    if obj_path is not None:
        meta["OBJFRAME"] = obj_path.stem
    if sky_path is not None:
        meta["SKYFRAME"] = sky_path.stem

    r = _finalize_wavelength(
        obj_1d, sky_1d, strips_lambda, objname, obj_path, sky_path,
        calib, cfg, inter, shift_wave=shift_wave,
        save_intermediates=save_intermediates, meta=meta,
    )
    if workdir is not None:
        r.write_to(workdir, save_intermediates=save_intermediates)
    return r


def _finalize_wavelength(
    obj_1d: dict[int, NDArray],
    sky_1d: dict[int, NDArray],
    strips_lambda: dict[int, tuple[float, float]],
    objname: str,
    obj_path: Path | None,
    sky_path: Path | None,
    calib: Calibration,
    cfg: Config,
    inter: Intermediates,
    *,
    shift_wave: float,
    save_intermediates: bool,
    meta: dict | None = None,
) -> Reduction:
    """s11–s13: truncate (with optional shift) → dispcor → FSR cut → Reduction.

    Factored out of :func:`reduce` so the multi-frame driver can re-run this
    cheap tail with a per-frame ``shift_wave`` without repeating the expensive
    2D + rectify + extract work. ``strips_lambda`` maps order → the rectified
    strip's ``(lambda_min, dy)`` linear-WCS pair (the only thing the s11
    truncate needs from s06).
    """
    # --- s11 truncate (shift=shift_wave). Even at shift 0, WARP runs
    # scopy(rebin=YES) to truncate the rectified strip to the [1, 2048] Å
    # range, setting the input length for dispcor.
    obj_truncated: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    sky_truncated: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    for m, spec in obj_1d.items():
        lambda_min, dy = strips_lambda[m]
        h = _astrofits.Header()
        h["CRVAL1"] = lambda_min
        h["CDELT1"] = dy
        h["CRPIX1"] = 1.0
        obj_truncated[m] = apply_waveshift_one_order(spec, h, shift_wave=shift_wave)
        if m in sky_1d:
            # WARP never waveshifts the sky path (truncate only); sky shift = 0.
            sky_truncated[m] = apply_waveshift_one_order(sky_1d[m], h, shift_wave=0.0)

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

    out_meta = dict(meta or {})
    out_meta["WAVSHIFT"] = float(shift_wave)

    return Reduction(
        obj_name=objname,
        obj_path=obj_path,
        sky_path=sky_path,
        obj=obj_out,
        sky=sky_out if cfg.flag_skyemission else None,
        intermediates=inter,
        meta=out_meta,
    )


@dataclass(frozen=True, slots=True)
class TransitSeries:
    """A set of per-frame reductions with layered wavelength calibration.

    Attributes:
        reductions: one :class:`Reduction` per frame. The WARP-compatible
            relative alignment is always applied first when requested; an
            optional physical wavecal correction may then be layered on top.
        shifts: WARP-compatible per-frame relative shift applied (Angstrom),
            length == n frames. These values are retained unchanged when the
            physical wavecal layer is applied.
        refid: index of the reference frame (its shift is 0).
        wavecal_solution: the optional telluric/OH residual solution applied
            after the WARP-compatible pass. ``None`` means WARP-only.
    """

    reductions: list
    shifts: NDArray
    refid: int
    wavecal_solution: WavecalSolution | None = None
    wavecal_run: Any | None = None

    def write_to(
        self,
        workdir: str | Path,
        *,
        save_intermediates: bool = False,
    ) -> None:
        """Write the fully calibrated time series and its provenance.

        Each exposure is written to ``workdir/<SERIESID>/`` (normally the
        object-frame name; object/sky pair names disambiguate repeated object
        frames). Writing happens on the completed :class:`TransitSeries` rather
        than at an intermediate step, so when a wavecal solution is present the
        FITS WCS holds the telluric/OH correction on top of the WARP alignment.

        The root directory also receives ``warp_alignment.npz`` and, when
        applicable, ``wavecal_solution.npz``, so the two calibration layers can
        be inspected or reproduced independently, plus
        ``telluric_transmission.npz`` and ``oh_support.npz``. Those two carry
        the atmosphere the wavecal fitted, which a later analysis needs to mask
        telluric and airglow pixels without refitting it.
        """
        root = Path(workdir)
        root.mkdir(parents=True, exist_ok=True)

        from decanter.wavecal.series import frame_ids_from_reductions

        frame_ids = frame_ids_from_reductions(self.reductions)

        for frame_id, reduction in zip(frame_ids, self.reductions, strict=True):
            reduction.write_to(
                root / frame_id,
                save_intermediates=save_intermediates,
            )

        np.savez_compressed(
            root / "warp_alignment.npz",
            frame_ids=np.asarray(frame_ids, dtype="U64"),
            shifts=np.asarray(self.shifts, dtype=float),
            refid=np.asarray(self.refid, dtype=np.int32),
        )
        if self.wavecal_solution is not None:
            self.wavecal_solution.save_npz(root / "wavecal_solution.npz")
        if self.wavecal_run is not None:
            from decanter.wavecal.products import airglow_product, telluric_product

            if self.wavecal_run.telluric_model is not None:
                telluric_product(self.wavecal_run, root / "telluric_transmission.npz")
            if self.wavecal_run.oh_model is not None:
                airglow_product(self.wavecal_run, root / "oh_support.npz")


def _robust_location(values: NDArray) -> float:
    """Median with iterative 3-MAD clipping for atmospheric anchors."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    for _ in range(5):
        kept = values[good]
        if kept.size < 3:
            break
        center = float(np.median(kept))
        sigma = float(1.4826 * np.median(np.abs(kept - center)))
        if not np.isfinite(sigma) or sigma == 0.0:
            break
        updated = good & (np.abs(values - center) <= 3.0 * sigma)
        if np.array_equal(updated, good):
            break
        good = updated
    return float(np.median(values[good])) if np.any(good) else float("nan")


def _dilated_support(support: NDArray, pixels: int) -> NDArray:
    """Expand line support so a broad velocity search retains shifted lines."""
    kernel = np.ones(2 * max(0, int(pixels)) + 1, dtype=int)
    return np.convolve(np.asarray(support, dtype=bool).astype(int), kernel,
                       mode="same") > 0


def _correlation_bank(
    signal: NDArray,
    template: NDArray,
    support: NDArray,
    shifts_pix: NDArray,
) -> NDArray:
    """Pearson CCF for every exposure against one shifted order template."""
    pixel = np.arange(template.size, dtype=float)
    bank = np.asarray([
        np.interp(pixel - shift, pixel, template, left=np.nan, right=np.nan)[support]
        for shift in shifts_pix
    ])
    bank -= np.nanmean(bank, axis=1, keepdims=True)
    bank = np.nan_to_num(bank, nan=0.0)
    bank_norm = np.sqrt(np.sum(bank * bank, axis=1))
    data = np.asarray(signal[:, support], dtype=float).copy()
    data -= np.nanmean(data, axis=1, keepdims=True)
    data = np.nan_to_num(data, nan=0.0)
    data_norm = np.sqrt(np.sum(data * data, axis=1))
    return np.divide(
        data @ bank.T,
        data_norm[:, None] * bank_norm[None, :],
        out=np.full((signal.shape[0], shifts_pix.size), np.nan),
        where=(data_norm[:, None] > 0) & (bank_norm[None, :] > 0),
    )


def _standardized_ccf(correlation: NDArray) -> NDArray:
    """Put one order's CCF on its own MAD scale, clipped against outliers."""
    center = np.nanmedian(correlation, axis=1, keepdims=True)
    scatter = 1.4826 * np.nanmedian(
        np.abs(correlation - center), axis=1, keepdims=True
    )
    standardized = np.divide(
        correlation - center,
        scatter,
        out=np.full_like(correlation, np.nan),
        where=np.isfinite(scatter) & (scatter > 0),
    )
    return np.clip(standardized, -8.0, 12.0)


def _add_standardized_ccf(
    pooled: NDArray, contribution: NDArray, correlation: NDArray,
) -> NDArray:
    """Pool orders without allowing a high-contrast order to dominate."""
    standardized = _standardized_ccf(correlation)
    finite = np.isfinite(standardized)
    pooled += np.where(finite, standardized, 0.0)
    contribution += finite
    return standardized


def _pooled_score(pooled: NDArray, contribution: NDArray) -> NDArray:
    """Mean standardised CCF, left NaN where too few orders contributed."""
    return np.divide(
        pooled, contribution, out=np.full_like(pooled, np.nan),
        where=contribution >= 2,
    )


def _pooled_peaks(score: NDArray, grid: NDArray, *, wing_kms: float = 20.0) -> dict:
    """Peak velocity, prominence and width of one pooled CCF per exposure."""
    from decanter.wavecal.measure import parabolic_extremum

    keys = ("velocity", "score", "snr", "secondary", "separation", "fwhm")
    out = {key: np.full(score.shape[0], np.nan) for key in keys}
    for i, row in enumerate(score):
        good = np.isfinite(row)
        if not np.any(good):
            continue
        index = int(np.nanargmax(row))
        out["velocity"][i] = parabolic_extremum(grid, row, index)
        out["score"][i] = row[index]
        center = float(np.nanmedian(row))
        scatter = float(1.4826 * np.nanmedian(np.abs(row - center)))
        if scatter > 0:
            out["snr"][i] = (row[index] - center) / scatter
        # The pooled peak is broad. Step outside its wings so the runner-up is
        # a separate maximum rather than the same peak's shoulder.
        away = good & (np.abs(grid - out["velocity"][i]) >= wing_kms)
        if np.any(away):
            elsewhere = np.flatnonzero(away)
            best = int(elsewhere[np.nanargmax(row[elsewhere])])
            out["secondary"][i] = row[best]
            out["separation"][i] = abs(grid[best] - out["velocity"][i])
        half = center + 0.5 * (row[index] - center)
        left = right = index
        while left > 0 and np.isfinite(row[left - 1]) and row[left - 1] >= half:
            left -= 1
        while right + 1 < row.size and np.isfinite(row[right + 1]) and row[right + 1] >= half:
            right += 1
        out["fwhm"][i] = grid[right] - grid[left]
    return out


def _order_bootstrap(
    evidence: NDArray, grid: NDArray, *, draws: int = 400, seed: int = 2109,
) -> tuple[NDArray, NDArray, NDArray]:
    """Spread of the pooled peak when the contributing orders are resampled.

    The pooled peak is an average over a modest number of orders, so its
    uncertainty is set by how much the orders disagree rather than by the
    photon noise within any one of them.
    """
    from decanter.wavecal.measure import parabolic_extremum

    n_frames = evidence.shape[1] if evidence.ndim == 3 else 0
    sigma = np.full(n_frames, np.nan)
    low = np.full(n_frames, np.nan)
    high = np.full(n_frames, np.nan)
    if evidence.ndim != 3 or evidence.shape[0] < 3:
        return sigma, low, high
    generator = np.random.default_rng(seed)
    for i in range(n_frames):
        usable = np.flatnonzero(np.any(np.isfinite(evidence[:, i]), axis=1))
        if usable.size < 3:
            continue
        sample = generator.choice(usable, size=(draws, usable.size), replace=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows = np.nanmean(evidence[sample, i], axis=1)
        peaks = np.array([
            parabolic_extremum(grid, row, int(np.nanargmax(row)))
            if np.any(np.isfinite(row)) else np.nan
            for row in rows
        ])
        if not np.any(np.isfinite(peaks)):
            continue
        sigma[i] = float(np.nanstd(peaks))
        low[i], high[i] = (float(v) for v in np.nanpercentile(peaks, [16, 84]))
    return sigma, low, high


def _atmospheric_common_solution(run: Any, config: WavecalConfig) -> WavecalSolution:
    """Measure one broad pooled telluric+OH shift for every object/sky pair.

    Telluric absorption is measured in telluric-rich object orders. Strong OH
    emission contributes from paired sky orders only where tellurics did not
    already claim the order, matching the hybrid ladder's priority and
    avoiding double weighting. The adjacent object and sky spectra are treated
    as one atmospheric epoch; the fine physical solve subsequently measures
    and corrects order-dependent residuals.
    """
    from decanter.wavecal.solution import WavecalSolution
    from decanter.wavecal.measure import highpass
    from decanter.wavecal.solve import AtmosphericCommonMode, _normalized

    series = run.series
    search = float(config.atmospheric_search_kms)
    step = float(config.atmospheric_step_kms)
    grid = np.arange(-search, search + 0.5 * step, step)
    shape = (series.n_frames, grid.size)
    telluric_pooled = np.zeros(shape, dtype=float)
    telluric_count = np.zeros(shape, dtype=float)
    oh_pooled = np.zeros(shape, dtype=float)
    oh_count = np.zeros(shape, dtype=float)
    evidence: list[NDArray] = []

    accepted = np.asarray(run.telluric_accepted, dtype=bool)
    minimum_frames = max(4, series.n_frames // 3)
    telluric_orders = (
        np.count_nonzero(accepted, axis=0) >= minimum_frames
        if run.telluric_model is not None else
        np.zeros(series.n_orders, dtype=bool)
    )
    used_telluric: list[int] = []
    if run.telluric_model is not None:
        signal = 1.0 - _normalized(series)
        feature = 1.0 - np.asarray(run.telluric_model.native_template, dtype=float)
        static = np.asarray([
            _robust_location(run.telluric_velocity[accepted[:, j], j])
            if telluric_orders[j] else np.nan
            for j in range(series.n_orders)
        ])
        for j in np.flatnonzero(telluric_orders):
            pad = int(np.ceil((abs(static[j]) + search) / series.dv_pix_kms[j]))
            support = _dilated_support(run.telluric_support[:, j], pad)
            if np.count_nonzero(support) < 20:
                continue
            velocities = static[j] + grid
            correlation = _correlation_bank(
                signal[:, :, j], feature[:, j], support,
                velocities / series.dv_pix_kms[j],
            )
            evidence.append(
                _add_standardized_ccf(telluric_pooled, telluric_count, correlation)
            )
            used_telluric.append(int(series.orders[j]))

    used_oh: list[int] = []
    if series.sky is not None and run.oh_model is not None:
        oh_accepted = np.asarray(run.oh_accepted, dtype=bool)
        oh_orders = ((np.count_nonzero(oh_accepted, axis=0) >= minimum_frames)
                     & ~telluric_orders)
        oh_signal = np.empty_like(series.sky, dtype=float)
        oh_template = np.empty_like(run.oh_model.native_template, dtype=float)
        for j in range(series.n_orders):
            oh_template[:, j] = highpass(run.oh_model.native_template[:, j], 101)
            for i in range(series.n_frames):
                oh_signal[i, :, j] = highpass(series.sky[i, :, j], 101)
        oh_static = np.asarray(run.oh_model.parameters[:, 1], dtype=float) * series.dv_pix_kms
        for j in np.flatnonzero(oh_orders):
            pad = int(np.ceil((abs(oh_static[j]) + search) / series.dv_pix_kms[j]))
            support = _dilated_support(run.oh_model.support[:, j], pad)
            if np.count_nonzero(support) < 20:
                continue
            velocities = oh_static[j] + grid
            correlation = _correlation_bank(
                oh_signal[:, :, j], oh_template[:, j], support,
                velocities / series.dv_pix_kms[j],
            )
            evidence.append(
                _add_standardized_ccf(oh_pooled, oh_count, correlation)
            )
            used_oh.append(int(series.orders[j]))

    telluric_score = _pooled_score(telluric_pooled, telluric_count)
    oh_score = _pooled_score(oh_pooled, oh_count)
    joint_score = _pooled_score(
        telluric_pooled + oh_pooled, telluric_count + oh_count
    )
    joint = _pooled_peaks(joint_score, grid)
    common = joint["velocity"].copy()
    if not np.all(np.isfinite(common)):
        bad = np.flatnonzero(~np.isfinite(common))
        raise ValueError(
            "atmospheric pre-alignment has no pooled CCF peak for "
            f"exposure rows {bad.tolist()}"
        )
    common -= _robust_location(common)

    stacked = np.asarray(evidence) if evidence else np.empty((0, 0, 0))
    sigma, p16, p84 = _order_bootstrap(stacked, grid)
    diagnostic = AtmosphericCommonMode(
        velocity_grid_kms=grid,
        telluric_score=telluric_score,
        oh_score=oh_score,
        joint_score=joint_score,
        peak_velocity_kms=joint["velocity"],
        telluric_peak_velocity_kms=_pooled_peaks(telluric_score, grid)["velocity"],
        oh_peak_velocity_kms=_pooled_peaks(oh_score, grid)["velocity"],
        peak_snr=joint["snr"],
        peak_score=joint["score"],
        secondary_score=joint["secondary"],
        secondary_separation_kms=joint["separation"],
        fwhm_kms=joint["fwhm"],
        bootstrap_sigma_kms=sigma,
        bootstrap_p16_kms=p16,
        bootstrap_p84_kms=p84,
        common_velocity_kms=common,
        telluric_orders=tuple(used_telluric),
        oh_orders=tuple(used_oh),
        search_kms=search,
        step_kms=step,
    )

    matrix = np.repeat(common[:, None], series.n_orders, axis=1)
    return WavecalSolution(
        frame_ids=series.frame_ids,
        orders=series.orders,
        velocity=matrix,
        source=np.full(matrix.shape, "interpolated", dtype="U16"),
        bracketed=np.ones(matrix.shape, dtype=bool),
        mode=run.solution.mode,
        zero_point="relative",
        assembly="decomposition",
        meta={
            "stage": "broad_pooled_telluric_oh_ccf",
            "telluric_orders": used_telluric,
            "oh_orders": used_oh,
            "search_kms": search,
            "step_kms": step,
            "common_velocity_kms": common.tolist(),
            "peak_score": joint["score"].tolist(),
            "peak_snr": joint["snr"].tolist(),
            "peak_fwhm_kms": joint["fwhm"].tolist(),
            "bootstrap_sigma_kms": sigma.tolist(),
            "diagnostic": diagnostic,
        },
    )


def _compose_wavecal_solutions(
    coarse: WavecalSolution, fine: WavecalSolution,
) -> WavecalSolution:
    """Compose two WCS velocity rescalings without approximating their sum."""
    from decanter.wavecal.solution import C_KMS, WavecalSolution

    if coarse.frame_ids != fine.frame_ids or coarse.orders != fine.orders:
        raise ValueError("coarse and fine wavecal grids do not match")
    scale = ((1.0 + np.asarray(coarse.velocity) / C_KMS)
             * (1.0 + np.asarray(fine.velocity) / C_KMS))
    combined = C_KMS * (scale - 1.0)
    meta = dict(fine.meta)
    meta.update({
        "atmospheric_prealign": True,
        "coarse_common_velocity_kms": coarse.velocity[:, 0].tolist(),
        "fine_solution_meta": dict(fine.meta),
    })
    return WavecalSolution(
        frame_ids=fine.frame_ids,
        orders=fine.orders,
        velocity=combined,
        source=fine.source.copy(),
        bracketed=fine.bracketed.copy(),
        mode=fine.mode,
        zero_point=fine.zero_point,
        assembly=fine.assembly,
        oh_tie_kms=fine.oh_tie_kms,
        meta=meta,
    )


def calibrate_wavelengths(
    series: TransitSeries,
    config: WavecalConfig | None = None,
    *,
    verbose: bool = True,
    diagnostic_pdf: str | Path | None = None,
) -> TransitSeries:
    """Layer physical telluric/OH wavecal on a reduced series.

    ``series.shifts`` is neither replaced nor recomputed. Normally the hybrid
    solver measures the residual after WARP alignment. With
    ``config.atmospheric_prealign``, a preparatory measurement pass supplies
    templates and trusted orders for a broad pooled telluric+OH CCF; its
    wavelength solution is not applied. The pooled common mode registers the
    spectra in WCS, the atmospheric templates are rebuilt, and one final
    hybrid solution is measured. Only the pooled registration and final
    physical solution are composed into the output.
    """
    if series.wavecal_solution is not None:
        raise ValueError("this TransitSeries already has a physical wavecal solution")

    # Imported lazily so the base WARP-compatible pipeline still runs without
    # the wavecal extra installed.
    from decanter.wavecal.config import WavecalConfig
    from decanter.wavecal.series import from_reductions
    from decanter.wavecal.solve import solve

    cfg = config or WavecalConfig()
    reference = from_reductions(series.reductions, fsr_cut=cfg.fsr_cut)
    cfg = cfg.resolved_for(reference.instmode)

    def invoke(reference_grid):
        solved_run = solve(
            reference_grid, cfg, verbose=verbose, return_diagnostics=True
        )
        return solved_run, solved_run.solution

    run, solution = invoke(reference)
    if cfg.atmospheric_prealign:
        if run is None:
            raise RuntimeError(
                "atmospheric pre-alignment requires diagnostic wavecal measurements"
            )
        coarse = _atmospheric_common_solution(run, cfg)
        if verbose:
            common = coarse.velocity[:, 0]
            print(
                "  atmospheric pre-alignment: "
                f"range {np.ptp(common) * 1e3:.0f} m/s, "
                f"RMS {np.std(common) * 1e3:.0f} m/s; rebuilding templates",
                flush=True,
            )
        from decanter.wavecal.series import frame_ids_from_reductions

        frame_ids = frame_ids_from_reductions(series.reductions)
        coarse_reductions = [
            coarse.apply(reduction, frame_id=frame_id)
            for reduction, frame_id in zip(
                series.reductions, frame_ids, strict=True
            )
        ]
        fine_reference = from_reductions(
            coarse_reductions, fsr_cut=cfg.fsr_cut
        )
        run, fine_solution = invoke(fine_reference)
        solution = _compose_wavecal_solutions(coarse, fine_solution)
        if run is not None:
            run.atmospheric = coarse.meta.get("diagnostic")
    if diagnostic_pdf is not None:
        from decanter.wavecal.report import wavecal_report_pdf

        label = series.reductions[0].obj_name if series.reductions else "dataset"
        wavecal_report_pdf(run, diagnostic_pdf, dataset=label)
    from decanter.wavecal.series import frame_ids_from_reductions

    frame_ids = frame_ids_from_reductions(series.reductions)
    corrected = [
        solution.apply(reduction, frame_id=frame_id)
        for reduction, frame_id in zip(series.reductions, frame_ids, strict=True)
    ]
    return TransitSeries(
        reductions=corrected,
        shifts=series.shifts,
        refid=series.refid,
        wavecal_solution=solution,
        wavecal_run=run,
    )


#: Intermediate fields the cross-frame alignment tail reads back. The rest of
#: :class:`Intermediates` is 2D and unused once the 1D spectra exist.
_ALIGNMENT_INTERMEDIATES = (
    "spectra_1d", "sky_1d", "strip_wcs", "spectra_dispcor", "sky_dispcor",
)


def _release_2d_intermediates(reduction: Reduction) -> Reduction:
    """Drop the 2D intermediates a multi-frame run has finished with.

    One frame holds ~300 MB of them, nine full-detector arrays plus the
    per-order strips, against ~1 MB of 1D spectra the alignment tail re-reads.
    A series is freed frame by frame unless the caller asked to keep it.
    """
    inter = reduction.intermediates
    for name in vars(inter):
        if name in _ALIGNMENT_INTERMEDIATES:
            continue
        current = getattr(inter, name)
        setattr(inter, name, {} if isinstance(current, dict) else None)
    return reduction


def reduce_many(
    pairs: list[tuple],
    calib: Calibration,
    *,
    refid: int | None = None,
    config: Config | None = None,
    extract: str = "box",
    check_calib: bool = True,
    subtract_background: bool = False,
    align: bool = True,
    workdir: str | Path | None = None,
    save_intermediates: bool = False,
    wavecal_config: WavecalConfig | None = None,
    wavecal_verbose: bool = True,
    wavecal_diagnostic_pdf: str | Path | None = None,
    jobs: int = 1,
) -> TransitSeries:
    """Reduce a list of ``(obj, sky)`` frame pairs and align them in wavelength.

    Runs :func:`reduce` on each pair once (the expensive part), then measures a
    per-frame cross-frame wavelength shift by cross-correlating the extracted
    1D spectra against a reference frame (WARP's ``ccwaveshift`` / s10), and
    re-runs only the cheap wavelength-finalize tail with each frame's shift so
    every frame lands on the reference frame's grid. When ``wavecal_config``
    is supplied, the telluric/OH physical calibration is then solved from and
    applied to those WARP-aligned products. The two layers stay separately
    recorded in :class:`TransitSeries`. With ``workdir``, writing waits for
    that final state, so the saved FITS WCS carries both layers.

    Args:
        pairs: ``[(obj, sky), ...]`` — paths or arrays, as for :func:`reduce`.
        refid: reference-frame index; default = the highest-flux frame.
        extract: ``"box"`` or ``"optimal"`` (applied to every frame).
        align: if False, skip shift measurement (shifts all 0) — useful to
            get the per-frame reductions without cross-frame alignment.
        workdir: optional root for the completed time series. One directory is
            written per object frame, plus the calibration solution files.
        save_intermediates: also persist captured reduction intermediates when
            ``workdir`` is supplied. Final spectra are always written.
        wavecal_config: optional physical wavelength-calibration settings.
            ``None`` (default) preserves the original WARP-only behavior.
        wavecal_verbose: print template-fit progress when physical wavecal is
            enabled.
        wavecal_diagnostic_pdf: optional filename for the full calibration
            report: order selection, telluric and OH fits, drift by physical
            order, interpolation, and tracer compatibility.
        jobs: number of worker processes for the expensive independent
            per-frame extraction stage. Series alignment, physical wavecal,
            and final writing remain single coordinated stages.

    Returns:
        A :class:`TransitSeries`.
    """
    cfg = config or Config()

    def _finish(result: TransitSeries) -> TransitSeries:
        if wavecal_config is not None:
            result = calibrate_wavelengths(
                result,
                wavecal_config,
                verbose=wavecal_verbose,
                diagnostic_pdf=wavecal_diagnostic_pdf,
            )
        if workdir is not None:
            result.write_to(workdir, save_intermediates=save_intermediates)
        return result

    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    reduction_kwargs = {
        "config": cfg,
        "extract": extract,
        "check_calib": check_calib,
        "subtract_background": subtract_background,
        "save_intermediates": True,
        "shift_wave": 0.0,
    }
    def _keep(reduction: Reduction) -> Reduction:
        return reduction if save_intermediates else _release_2d_intermediates(reduction)

    if jobs == 1:
        base = [_keep(reduce(o, calib, sky=s, **reduction_kwargs)) for o, s in pairs]
    else:
        # Resolve futures in submission order so frame ordering is stable, and
        # free each frame as it lands rather than after the whole series is in.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(reduce, o, calib, sky=s, **reduction_kwargs)
                for o, s in pairs
            ]
            base = [_keep(future.result()) for future in futures]
    n = len(base)
    if not align or n < 2:
        result = TransitSeries(reductions=base, shifts=np.zeros(n), refid=refid or 0)
        return _finish(result)

    orders = sorted(set.intersection(*[set(r.intermediates.spectra_1d) for r in base]))
    # Truncated 1D per order per frame (the WARP `_m###c` cross-correlation input).
    trunc: dict[int, list] = {m: [] for m in orders}
    for r in base:
        it = r.intermediates
        for m in orders:
            lam, dy = it.strip_wcs[m]
            h = _astrofits.Header()
            h["CRVAL1"] = lam; h["CDELT1"] = dy; h["CRPIX1"] = 1.0
            t, _ = apply_waveshift_one_order(it.spectra_1d[m], h, shift_wave=0.0)
            trunc[m].append(np.asarray(t, float))

    ref_ord = 163 if 163 in orders else orders[len(orders) // 2]
    if refid is None:
        refid = int(np.argmax([np.nanmedian(trunc[ref_ord][i]) for i in range(n)]))
    dy = base[0].intermediates.strip_wcs[orders[0]][1]   # common across orders
    rows = []
    for m in orders:
        L = min(s.size for s in trunc[m])
        rows.append(waveshift_one_order([s[:L] for s in trunc[m]], dy, refid=refid))
    shift_avg, _, _, _ = waveshift_clip(np.array(rows))

    aligned = []
    for r, sh in zip(base, shift_avg):
        it = r.intermediates
        aligned.append(_finalize_wavelength(
            it.spectra_1d, it.sky_1d, it.strip_wcs, r.obj_name, r.obj_path,
            r.sky_path, calib, cfg, it, shift_wave=float(sh),
            save_intermediates=True, meta=r.meta,
        ))
    result = TransitSeries(reductions=aligned, shifts=shift_avg, refid=refid)
    return _finish(result)


def combine(
    reductions: list | TransitSeries,
    *,
    cut: float | None = None,
    weight: str = "flux",
) -> Reduction:
    """SNR-weighted combine of aligned per-frame reductions into a master.

    Stacks each ``(fsr_cut, order)`` spectrum present in every frame onto the
    reference grid. Intended for a master (e.g. an out-of-transit template);
    for a transit time series keep :class:`TransitSeries.reductions` separate.

    Args:
        reductions: a list of :class:`Reduction` (ideally wavelength-aligned,
            e.g. from :func:`reduce_many`) or a :class:`TransitSeries`.
        cut: restrict to one FSR cut; default combines every cut present.
        weight: ``"flux"`` (photon-weighted, ~SNR^2) or ``"uniform"``.

    Returns:
        A :class:`Reduction` whose ``obj`` holds the combined spectra.
    """
    reds = reductions.reductions if isinstance(reductions, TransitSeries) else list(reductions)
    if not reds:
        raise ValueError("combine() needs at least one reduction")
    keys = set.intersection(*[set(r.obj) for r in reds])
    if cut is not None:
        keys = {k for k in keys if k[0] == cut}
    obj_out: dict[tuple[float, int], OrderSpectrum] = {}
    for key in sorted(keys):
        specs = [r.obj[key] for r in reds]
        L = min(s.flux.size for s in specs)
        ref = specs[0]
        same_grid = all(
            np.isclose(spec.crval1, ref.crval1, rtol=0.0, atol=1e-12)
            and np.isclose(spec.cdelt1, ref.cdelt1, rtol=0.0, atol=1e-15)
            and np.isclose(spec.crpix1, ref.crpix1, rtol=0.0, atol=1e-12)
            for spec in specs[1:]
        )
        if same_grid:
            F = np.array([np.asarray(spec.flux[:L], float) for spec in specs])
        else:
            # A per-frame wavecal changes the WCS without touching flux, so
            # the frames sit on different wavelength grids. Regrid once here.
            target_wave = np.asarray(ref.wavelength[:L], dtype=float)
            rows = []
            for spec in specs:
                native_wave = np.asarray(spec.wavelength, dtype=float)
                native_flux = np.asarray(spec.flux, dtype=float)
                if native_wave[0] > native_wave[-1]:
                    native_wave = native_wave[::-1]
                    native_flux = native_flux[::-1]
                rows.append(
                    np.interp(target_wave, native_wave, native_flux, left=np.nan, right=np.nan)
                )
            F = np.asarray(rows)
        if weight == "uniform":
            w = np.ones(len(specs))
        else:
            w = np.array([max(float(np.nanmedian(s.flux[:L])), 1e-9) for s in specs])
        if np.all(np.isfinite(F)):
            stack = np.average(F, axis=0, weights=w)
        else:
            valid = np.isfinite(F)
            denominator = np.sum(valid * w[:, None], axis=0)
            stack = np.divide(
                np.nansum(F * w[:, None], axis=0),
                denominator,
                out=np.full(L, np.nan),
                where=denominator > 0,
            )
        obj_out[key] = OrderSpectrum(
            order=key[1], fsr_cut=key[0], flux=stack.astype(np.float32),
            crval1=ref.crval1, cdelt1=ref.cdelt1, crpix1=ref.crpix1,
        )
    # A stack has no single mid-time or pointing, so only the frame-independent
    # configuration keys are carried onto the combined product.
    stack_meta = {
        key: value for key, value in reds[0].meta.items()
        if key in {"OBJECT", "INSTRUME", "TELESCOP", "OBSERVAT", "INSTMODE",
                   "SETTING", "PERIOD", "SLIT", "PIPELINE"}
    }
    stack_meta["NCOMBINE"] = len(reds)

    return Reduction(
        obj_name=reds[0].obj_name, obj_path=None, sky_path=None,
        obj=obj_out, sky=None, meta=stack_meta,
    )
