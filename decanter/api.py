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

import dataclasses

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits
from numpy.typing import NDArray

from decanter._reduction import Intermediates, OrderSpectrum, OrderSpectrum2D, Reduction
from decanter.calib import Calibration
from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.extract.aperture_solve import solve_frame_aperture
from decanter.io.apdb import parse as apdb_parse
from decanter.extract.box_extract_1d import box_extract
from decanter.extract.optimal_extract import optimal_extract
from decanter.image2d import (
    fix_bad_pixels, flatfield_divide, sky_subtract, subtract_apscatter,
)
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.fsr import load as load_fsr
from decanter.utils.cosmic_ray import ndr_from_header
from decanter.utils.iraf_imio import imio_read_real
from decanter.utils.iraf_scopy import scopy_wavelength_truncate
from decanter.waveshift.apply import apply_waveshift_one_order, shift_truncated
from decanter.waveshift.measure import waveshift_clip, waveshift_one_order
from decanter.wavelength import dispcor_one_order, truncate_spectrum
from decanter.wavelength.fsr import fsr_bounds, truncate_spectrum_iraf
from decanter.wavelength.iraf_wcs import DispcorWCS, dispcor_wcs

_OBJNAME_BAD_CHARS: tuple[str, ...] = (" ", "'", "\"", "#", "/")

# Reduction modes. Both run the vectorised WARP pipeline (decanter.fast).
# "default": no cross-frame waveshift; the OH + telluric wavelength
# calibration (decanter.wavecal) puts every frame on the true telluric scale.
# "warp-clone": WARP to the bit (its cross-correlation shift as configured,
# no wavecal; docs/BITEXACT.md).
MODES: tuple[str, ...] = ("default", "warp-clone")


def resolve_mode(mode: str) -> str:
    """Check ``mode`` (``"default"`` or ``"warp-clone"``).

    Raises:
        ValueError: for any other value.
    """
    if mode not in MODES:
        hint = " (v1.0's 'warp' is now 'warp-clone')" if mode == "warp" else ""
        raise ValueError(f"mode={mode!r} is not a decanter mode{hint}; use 'default' "
                         f"(telluric-calibrated wavelengths) or 'warp-clone' (WARP to the bit)")
    return mode


def implementation(mode: str) -> dict[str, str]:
    """What runs for ``mode``: ``{"mode", "implementation", "backend"}`` (the s06 array
    backend, ``"numpy"`` or ``"mlx"``)."""
    from decanter.fast._backend import active_backend
    return {"mode": resolve_mode(mode), "implementation": "vectorised",
            "backend": active_backend()}


def _shift_for(shift_wave, m: int) -> float:
    """The s11 shift of order ``m``: one value for every order, or a per-order mapping."""
    if isinstance(shift_wave, dict):
        return float(shift_wave.get(m, 0.0))
    return float(shift_wave)


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in _OBJNAME_BAD_CHARS:
        name = name.replace(ch, "_")
    return name


def _comp_dy(comp: Path) -> float:
    """The s06 transform ``dy``: the comp file's CDELT1.

    WARP equivalent: ``warp/config.py`` ``self.dyinput = prihdr_comp["CDELT1"]``,
    which raises when the keyword is missing; so does this (no 0.5 default).
    """
    header = _fits.read_image(comp)[1]
    if "CDELT1" not in header:
        raise KeyError(f"comp file {comp} has no CDELT1 (the transform dy); "
                       f"WARP fails here too (warp/config.py: prihdr_comp['CDELT1'])")
    return float(header["CDELT1"])


def _is_abba(nodpos: str) -> bool:
    return "O" not in nodpos


def _centersearch_abba(nodpos: str) -> bool:
    """WARP's ``abbaflag`` for the PSF center search, which (unlike the
    cosmic-ray step) also excludes ``NODPOS == "NA"`` (Warp_sci.py:395)."""
    return _is_abba(nodpos) and nodpos.strip() != "NA"


def check_supported(cfg: Config) -> None:
    """Refuse parameter values WARP itself stops on.

    Every WARP parameter-file option is implemented (``Background
    Subtraction`` -> apall ``background``, ``CUTRANSFORM flux`` -> transform
    ``flux``, ``Extract 2d spectrum`` -> apall ``format=strip`` +
    ``resample2Dspec``).  WARP hands the two tokens to pyraf as typed, so a
    value runs exactly when pyraf and the IRAF task take it:

    * ``skysub_mode`` (with ``flag_skysub``): apall's ``background`` choice
      list, minimum-matched (``med`` is ``median``, ``no`` is ``none``;
      :func:`decanter.config.apall_background`);
    * ``fluxinput``: ``yes``/``no`` in any case or ``y``/``n``
      (:func:`decanter.config.pyraf_bool`; ``INDEF`` passes pyraf but
      ``transform`` stops on it).

    Raises:
        ValueError: for a value WARP stops on ("ERROR in extraction" /
            "ERROR in 2d image reduction").
        NotImplementedError: for a pyraf ``)``-indirection (not modelled).
    """
    _ = cfg.apall_background   # raises ValueError for a token pyraf refuses
    _ = cfg.transform_flux     # raises ValueError for a token WARP stops on


def background_mode(cfg: Config) -> str:
    """apall's ``background`` for the object extraction: the minimum-matched
    ``Background Subtraction`` token, or ``none`` without it
    (Warp_sci.py:437-438; :attr:`Config.apall_background`).  WARP's
    background *products* follow ``cfg.flag_skysub`` instead: with
    ``Background Subtraction: no`` apall subtracts nothing but WARP still
    writes ``_none``, ``_bg``, ``_bgc``, ``_bgcw`` and the background panel."""
    return cfg.apall_background


def _read_input(
    obj: Path | NDArray,
    sky: Path | NDArray | None,
) -> tuple[NDArray, _astrofits.Header, NDArray | None, _astrofits.Header | None,
           Path | None, Path | None]:
    """Resolve obj / sky to (data, header) tuples. Accept a path (str or
    os.PathLike) or an NDArray."""
    if isinstance(obj, (str, os.PathLike)):
        obj = Path(obj)
    if isinstance(sky, (str, os.PathLike)):
        sky = Path(sky)
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
    mode: str = "default",
    subtract_background: bool = False,
    extract: str = "box",
    optimal_profile: dict[int, tuple] | None = None,
    aperture: tuple[float, float] | None = None,
    background_region: str | None = None,
    _warp_products: bool | str = False,
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
        mode: ``"default"``: WARP's pipeline, then the OH + telluric
            wavelength calibration (:mod:`decanter.wavecal`, single-frame
            path; with a stored :class:`~decanter.wavecal.StaticSurface` of
            the calibration set when there is one) moves every order onto
            its true telluric wavelength scale through the s11 shift.
            ``"warp-clone"``: WARP to the bit (``docs/BITEXACT.md``),
            ``shift_wave`` as given. Any other value raises ``ValueError``.
            Both run the vectorised implementation (:mod:`decanter.fast`;
            the rectification on the Apple GPU via MLX when installed).
        extract: 1D extraction method. ``"box"`` (default) is the WARP-parity
            box sum; ``"optimal"`` is Horne (1986) profile-weighted extraction
            (higher SNR, an algorithmic upgrade over WARP — see
            :mod:`decanter.extract.optimal_extract`).
        aperture: manual ``(low, high)`` extraction window in pixels relative
            to the trace (WARP's ``Manual Aperture: yes`` with the listfile's
            ``ap=low:high``). Skips the center search; clamped to the
            reference apertures as WARP does. Required when
            ``config.flag_manual_aperture`` is set.
        background_region: the listfile's ``bg=`` token, IRAF ``sample``
            syntax relative to the aperture centre (e.g. ``"-6:-2,2:6"``).
            WARP writes it into the frame's aperture database (``xmin``/
            ``xmax``/``sample`` of the ``background`` section), which is what
            apall's background (``config.flag_skysub`` /
            ``config.skysub_mode``, the parameter file's ``Background
            Subtraction``) samples; ``None`` is WARP's ``INDEF``: the
            aperture window for ``average``/``median``/``minimum``, and for
            ``fit`` IRAF's ``rg_xadd`` range ``[0, INDEFR]``, every point at or
            right of the aperture centre (none for an aperture left of it,
            sky 0). The sample is parsed as IRAF does, whatever it is;
            ``reduce_to_warp_tree`` refuses beforehand the ``bg=`` tokens WARP
            stops on (``warptree.check_bg_tokens``).

    WARP's parameter-file options map to ``config``, their tokens taken as
    pyraf takes them: ``Background Subtraction`` other than ``none`` sets
    ``flag_skysub`` (``intermediates.spectra_1d_none`` keeps the plain sum,
    ``intermediates.bg_dispcor`` the dispersion-corrected background) and is
    apall's background, minimum-matched (``average``/``av``,
    ``median``/``med``, ``minimum``/``mi``, ``fit``/``f``; ``no`` is
    ``none``: nothing subtracted, the background products all zero);
    ``CUTRANSFORM flux: yes`` (any case, or ``y``) multiplies the rectified
    strips by the transform's Jacobian; ``Extract 2d spectrum: yes`` adds the
    2-D spectra (``Reduction.obj_2d`` and ``Reduction.obj_2d_resample``). All
    three follow WARP bit for bit (docs/BITEXACT.md).  Every IRAF task port
    reads its input images as IRAF's FITS kernel does (``-0.0``,
    subnormals, ``+-Inf``/NaN -> ``+0``; :mod:`decanter.utils.iraf_imio`).

    Raises:
        ValueError: for a parameter value WARP itself stops on
            (:func:`check_supported`).
        NotImplementedError: for a pyraf ``)``-indirection token.

    Returns:
        :class:`Reduction` carrying per-(fsr_cut, order) calibrated 1D
        spectra. Per-frame products keyed as ``(fsr_cut, order)`` so a
        caller can grab a specific order with ``r.obj[(1.05, 163)]``.
    """
    mode = resolve_mode(mode)
    from decanter import fast as _fast
    from decanter.fast import apsum as _fast_apsum
    from decanter.fast import rectify as _fast_rectify
    if extract not in ("box", "optimal"):
        raise ValueError(f"extract={extract!r} must be 'box' or 'optimal'")
    cfg = config or Config()
    check_supported(cfg)
    bg_mode = background_mode(cfg)
    bgsample = "INDEF" if background_region is None else str(background_region)
    # apall reads off-band pixels (background regions, 2-D strips): whole strips
    warp_opts = bg_mode != "none" or cfg.flag_extract2d
    # WARP's Background Subtraction products (_none, _bg, ...): flag_skysub,
    # also when the token is "no" (apall background none, Warp_sci.py:563-569)
    skysub = cfg.flag_skysub
    if (warp_opts or skysub) and (extract != "box" or subtract_background):
        raise ValueError(
            "WARP's Background Subtraction / Extract 2d spectrum options act on the "
            "box (apall) extraction; they cannot be combined with extract="
            f"{extract!r} / subtract_background={subtract_background!r}")
    flux = cfg.transform_flux
    if cfg.flag_manual_aperture and aperture is None:
        raise ValueError(
            "config.flag_manual_aperture is set ('Manual Aperture: yes') but no "
            "aperture=(low, high) was given (WARP reads it from the listfile's "
            "ap=low:high token)")
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

    inter = Intermediates() if save_intermediates else Intermediates()
    if save_intermediates:
        inter.obj_raw = obj_data

    # Every IRAF task below reads its input images through the FITS kernel,
    # which maps -0, subnormals, Inf and NaN to +0 (utils/iraf_imio:
    # imio_read_real); WARP's Python steps (cosmic-ray mask, center search,
    # resample2Dspec) read the files as written.
    rd = imio_read_real

    # --- s01 sky subtract -------------------------------------------------
    diff = (sky_subtract(rd(obj_data), rd(sky_data)) if sky_data is not None
            else obj_data.copy())
    if save_intermediates or _warp_products:
        inter.obj_s = diff

    # --- s02 cosmic-ray mask ----------------------------------------------
    # the vectorised detector hands over the whole-frame arrays WARP's
    # 4-OBJ_mask holds when they are wanted (it computes them anyway)
    cr_stats: dict = {"arrays": True} if _warp_products == "mask" else {}
    if cfg.flag_bpmask and sky_data is not None:
        mask = _fast.cr_mask(
            diff, obj_data, sky_data, calib, static_bp,
            nodpos=nodpos,
            ndr_obj=ndr_from_header(obj_header),
            ndr_sky=ndr_from_header(sky_header) if sky_header is not None else 1,
            config=cfg, stats=cr_stats,
        )
    else:
        mask = np.zeros_like(static_bp, dtype=np.int16)
    if save_intermediates or _warp_products:
        inter.cr_mask = mask
    if "pixnum" in cr_stats:
        inter.cr_stats = (cr_stats["pixnum"], cr_stats["sigthres"])
    if _warp_products == "mask" and cfg.flag_bpmask and sky_data is not None:
        # WARP's other cosmic-ray products (badpixmask.py:146-171): the
        # median-filter residuals of the difference frame and the noise frame
        # (written to 4-OBJ_mask only; _warp_products="mask" asks for them).
        from decanter.utils.cosmic_ray import warp_noise_from_median, warp_noise_image
        ndr_o = ndr_from_header(obj_header)
        ndr_s = ndr_from_header(sky_header) if sky_header is not None else 1
        if "mf1" in cr_stats:
            inter.extra["mf1"] = cr_stats["mf1"]
            inter.extra["mf2"] = cr_stats["mf2"]
            inter.extra["noise"] = warp_noise_from_median(cr_stats["smoothed"], ndr_o, ndr_s)
        else:
            from decanter.utils.median_filter import median_filter as _mf
            mf1 = diff - _mf(diff, size=(5, 1))
            inter.extra["mf1"] = mf1
            inter.extra["mf2"] = mf1 - _mf(mf1, footprint=np.identity(5))
            inter.extra["noise"] = warp_noise_image(obj_data, sky_data, ndr_o, ndr_s)
    # WARP maskflat_* (Warp_sci.py:352): imarith REAL sum static + CR; fixpix
    # reads it as INTEGER (truncated), > 0 bad (decanter.utils.fixpix).
    combined_mask = static_bp.astype(np.float32) + mask.astype(np.float32)
    if _warp_products:
        inter.extra["maskflat"] = combined_mask

    # --- s03 apscatter ----------------------------------------------------
    if cfg.flag_apscatter:
        obj_ssc, scatter = subtract_apscatter(rd(diff), apset_apsc)
    else:
        obj_ssc, scatter = diff, np.zeros_like(diff)
    if save_intermediates:
        inter.obj_ssc = obj_ssc
        inter.scatter_model = scatter

    # --- s04 flatfield divide --------------------------------------------
    obj_sscf = flatfield_divide(rd(obj_ssc), rd(flat))
    if save_intermediates:
        inter.obj_sscf = obj_sscf

    # --- s05 fixpix -------------------------------------------------------
    obj_sscfm = fix_bad_pixels(rd(obj_sscf), combined_mask)   # imcopy + fixpix
    if save_intermediates or _warp_products:
        inter.obj_sscfm = obj_sscfm

    # Sky 2D path (when requested)
    sky_fm = None
    if cfg.flag_skyemission and sky_data is not None:
        sky_f = flatfield_divide(rd(sky_data), rd(flat))
        sky_fm = fix_bad_pixels(rd(sky_f), combined_mask)
        if save_intermediates:
            inter.sky_f = sky_f
            inter.sky_fm = sky_fm

    # --- s06 rectify per order -------------------------------------------
    # (cutransform: imcopy of each order's columns, then transform)
    sscfm_in = rd(obj_sscfm)
    fm_in = rd(sky_fm) if sky_fm is not None else None
    orders = (
        apset_multi.echelle_orders
        if (cfg.reduce_full_data or not cfg.selected_orders)
        else tuple(m for m in apset_multi.echelle_orders if m in cfg.selected_orders)
    )
    dy = _comp_dy(calib.comp)

    # A manual window replaces any per-frame DBs (it is what WARP writes
    # into them in manual-aperture mode).
    trans_apdbs = {} if aperture is not None else (calib.trans_apdbs or {})
    # Fast s06: rectify obj and sky together; evaluate only the band of
    # each strip the extraction reads (the multihole reference window),
    # unless whole strips are wanted or an aperture leaves the band.
    # (the background regions and the 2-D strips may read beyond the
    # band; whole strips keep every option exact)
    whole = save_intermediates or extract != "box" or subtract_background or warp_opts
    full_orders = set() if whole else _fast_rectify.orders_needing_full(
        apset_multi, orders, fc_dir=calib.fc_dir, fc_refname=calib.fc_refname,
        dy=dy, ref_trans_apdbs=calib.ref_trans_apdbs, trans_apdbs=trans_apdbs)
    # The background regions and the 2-D strips (resample2Dspec reads
    # twice the aperture around the trace) can read off-band pixels,
    # where the MLX backend's float32 subnormals may differ from WARP
    # (docs/VECTORISED.md): with those options rectify on the CPU (exact).
    rect_backend = "numpy" if warp_opts else None
    rect = _fast_rectify.rectify_orders_multi(
        [sscfm_in] + ([fm_in] if fm_in is not None else []), apset_multi,
        fc_dir=calib.fc_dir, fc_refname=calib.fc_refname, dy=dy, orders=orders,
        ref_trans_apdbs=calib.ref_trans_apdbs, full_orders=full_orders, full=whole,
        flux=flux, backend=rect_backend)
    strips_obj = rect[0]
    strips_sky_full = rect[1] if sky_fm is not None else None
    if save_intermediates:
        inter.strips_obj = {m: s.data for m, s in strips_obj.items()}
    strips_sky_arrays: dict[int, NDArray] = {}
    if strips_sky_full is not None:
        strips_sky_arrays = {m: s.data for m, s in strips_sky_full.items()}
        if save_intermediates:
            inter.strips_sky = strips_sky_arrays

    # --- s07/s08 trace + 1D extract --------------------------------------
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
            strips_obj, ref_trans, abba=_centersearch_abba(nodpos), window=aperture,
            bgsample=bgsample,
        )

    ap_used: dict[int, tuple[float, float]] = {}
    ap_refs: dict[int, Path] = {}
    # WARP's Background Subtraction / Extract 2d spectrum products
    obj_1d_none: dict[int, NDArray] = {}
    strips_2d: dict[int, object] = {}
    strips_2d_resample: dict[int, NDArray] = {}

    def _warp_options(m, strip_arr, entry) -> None:
        """apall ``background=<mode>`` for the object sum (the plain sum kept
        as ``_none``) and ``format=strip`` + ``resample2Dspec``
        (Warp_sci.py:561-584)."""
        from decanter.extract import apbackground as _apb
        from decanter.extract.resample2d import resample_2d
        sky_line = None
        if skysub:
            # apall background=none (the plain sum WARP keeps as _none)
            obj_1d_none[m] = obj_1d[m]
        if bg_mode != "none":
            if _apb.background_of(entry) is not None:
                sky_line, _ = _apb.sky_values_fast(strip_arr, entry, bg_mode)
            obj_1d[m] = _apb.apsum_background_fast(strip_arr, entry, bg_mode, sky=sky_line)
        if cfg.flag_extract2d:
            strips_2d[m] = _apb.strip_extract_fast(strip_arr, entry, bg_mode, sky=sky_line)
            strips_2d_resample[m] = resample_2d(strip_arr, entry)

    def _solve_one_order(m, strip):
        # Locked-path fallback (no WARP equivalent): same center search and
        # abbaflag as the free path, on this order only.
        if not calib.ref_trans_apdbs or m not in calib.ref_trans_apdbs:
            raise ValueError(f"order {m}: no per-frame trans DB and no multihole "
                             f"trans reference aperture to solve it from")
        return solve_frame_aperture({m: strip}, calib.ref_trans_apdbs,
                                    abba=_centersearch_abba(nodpos), bgsample=bgsample)

    def _extract_orders(ms) -> None:
        for m in ms:
            strip = strips_obj[m]
            strip_arr = strip.data
            sub = None
            ap_refs[m] = Path(trans_apdbs[m]) if m in trans_apdbs else Path(
                (calib.ref_trans_apdbs or {}).get(m, ""))
            if m in trans_apdbs:
                trans_set = ApertureSet.load(
                    trans_apdbs[m], array_length=strip_arr.shape[0]
                )
                if m in trans_set.apertures:
                    trans_ap = trans_set.apertures[m]
                    ap_low, ap_high = float(trans_ap.entry.low), float(trans_ap.entry.high)
                    trace_x = trans_ap.trace_x
                else:
                    sub = _solve_one_order(m, strip)
            elif frame_ap is None:
                # Locked, but this order has no per-frame DB: solve it alone.
                sub = _solve_one_order(m, strip)
            else:
                trace_x = frame_ap.traces[m]
                ap_low, ap_high = frame_ap.ap_low, frame_ap.ap_high
            if sub is not None:
                trace_x = sub.traces[m]
                ap_low, ap_high = sub.ap_low, sub.ap_high
            ap_used[m] = (float(ap_low), float(ap_high))
            exact_entry = None
            if sub is not None and extract == "box" and not subtract_background:
                exact_entry = sub.entries.get(m)
            elif m in trans_apdbs and extract == "box" and not subtract_background:
                exact_entry = apdb_parse(trans_apdbs[m]).get(m)
            elif frame_ap is not None and extract == "box" and not subtract_background:
                # Free path: the per-frame aperture database WARP would write,
                # token for token (extract.aperture_solve.solve_frame_apdb).
                exact_entry = frame_ap.entries.get(m)
            if exact_entry is not None:
                # Same bit-exact apall, obj and sky together, geometry cached.
                pair = [strip_arr] + ([strips_sky_arrays[m]] if m in strips_sky_arrays else [])
                sums = _fast_apsum.apsum_multi(pair, exact_entry)
                obj_1d[m] = sums[0]
                if m in strips_sky_arrays:
                    sky_1d[m] = sums[1]
            elif extract == "optimal":
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
            if warp_opts or skysub:
                if exact_entry is None:
                    raise ValueError(f"order {m}: WARP's background / 2-D options need the "
                                     f"per-frame aperture database (box extraction)")
                _warp_options(m, strip_arr, exact_entry)

    _extract_orders(list(strips_obj))
    # Safety net: an order whose extraction read outside its evaluated
    # band (NaN there) is rectified in full and extracted again.
    redo = _fast_rectify.orders_to_redo(strips_obj, obj_1d, sky_1d)
    if redo:
        rect = _fast_rectify.rectify_orders_multi(
            [sscfm_in] + ([fm_in] if fm_in is not None else []), apset_multi,
            fc_dir=calib.fc_dir, fc_refname=calib.fc_refname, dy=dy, orders=redo,
            full=True, flux=flux, backend=rect_backend)
        for m in redo:
            strips_obj[m] = rect[0][m]
            if sky_fm is not None:
                strips_sky_arrays[m] = rect[1][m].data
        _extract_orders(redo)
    inter.spectra_1d = dict(obj_1d)
    inter.sky_1d = dict(sky_1d)
    inter.spectra_1d_none = dict(obj_1d_none)
    inter.strips_2d = {m: sr.data for m, sr in strips_2d.items()}
    inter.strip_2d_limits = {m: (sr.low, sr.high) for m, sr in strips_2d.items()}
    inter.strips_2d_resample = dict(strips_2d_resample)
    # sarith(none - background-subtracted): the background spectrum (SN = n)
    bg_1d = {m: (rd(obj_1d_none[m]) - rd(obj_1d[m])).astype(np.float32)
             for m in obj_1d_none}

    # s11 reads the 1-D spectrum's CRVAL1/CDELT1 back from the FITS header
    # that apall wrote from the strip's header text (IRAF ctod/imputd round
    # trips), so hand it exactly those values.
    strips_lambda = {m: _extracted_wcs(strips_obj[m]) for m in obj_1d}
    inter.strip_wcs = strips_lambda
    inter.strip_meta = {m: (strips_obj[m].crval2_text, getattr(strips_obj[m], "cdelt2_text", ""),
                            int(np.asarray(strips_obj[m].data).shape[0]),
                            int(strips_obj[m].xmin), int(strips_obj[m].xmax))
                        for m in obj_1d}
    if _warp_products:
        _centersearch_products(inter, strips_obj, calib, frame_ap, nodpos,
                               manual=aperture is not None)

    r = _finalize_wavelength(
        obj_1d, sky_1d, strips_lambda, objname, obj_path, sky_path,
        calib, cfg, inter, shift_wave=shift_wave,
        save_intermediates=save_intermediates,
        bg_1d=bg_1d, twod={"strip": inter.strips_2d, "resample": inter.strips_2d_resample},
    )
    r.obj_header = obj_header if obj_path is not None else None
    r.sky_header = sky_header if sky_path is not None else None
    r.apertures = ap_used
    r.aperture_refs = ap_refs
    if frame_ap is not None:
        r.centersearch_undefined = {m: n for m, n in frame_ap.minpack_undefined.items() if n}
    if mode == "default":
        (r,), _ = _apply_wavecal([r], calib, cfg, save_intermediates=save_intermediates)
    if workdir is not None:
        r.write_to(workdir, save_intermediates=save_intermediates)
    return r


def _centersearch_products(inter: Intermediates, strips_obj, calib, frame_ap, nodpos: str, *,
                           manual: bool) -> None:
    """WARP's center-search log values and spatial-profile ``.dat`` data.

    Free path: from the aperture solution. Locked path (WARP's per-frame
    databases supplied): the center search is run for the logs only. A NaN
    center (WARP gives up) or a manual aperture: ``make_slit_profile``
    (``Warp_sci.py:403-404, 426``).
    """
    from decanter.extract.aperture_solve import _strip_wcs, solve_frame_apdb
    from decanter.io.apdb import parse as parse_apdb
    from decanter.utils.psf import warp_slit_profile
    refs_paths = calib.ref_trans_apdbs or {}
    xcenter: dict = {}
    fwhm: dict = {}
    profiles: dict = {}
    if not manual:
        if frame_ap is not None and frame_ap.xcenter:
            xcenter, fwhm, profiles = frame_ap.xcenter, frame_ap.fwhm, dict(frame_ap.profiles)
        elif refs_paths:
            try:
                sol = solve_frame_apdb(strips_obj, refs_paths, abba=_centersearch_abba(nodpos))
                xcenter, fwhm, profiles = sol.xcenter, sol.fwhm, dict(sol.profiles)
            except ValueError:
                pass
        inter.centersearch = {m: (xcenter[m], fwhm[m]) for m in xcenter}
    for m in strips_obj:
        if m not in refs_paths:
            continue
        if manual or (m in xcenter and not np.isfinite(xcenter[m])):
            data, crval2, cdelt2, crpix2 = _strip_wcs(strips_obj[m])
            profiles[m] = warp_slit_profile(data, parse_apdb(refs_paths[m])[m],
                                            crval2=crval2, cdelt2=cdelt2, crpix2=crpix2)
    inter.slit_profiles = profiles


def _extracted_wcs(strip) -> tuple[float, float]:
    """(CRVAL1, CDELT1) as WARP's s11 ``scopy`` reads them from the apall output."""
    from decanter.utils.fpemu import iraf_ctod
    from decanter.utils.iraf_fmt import iraf_imputd_text

    def rt(text: str, fallback: float) -> float:
        if not text:
            return float(fallback)
        return iraf_ctod(iraf_imputd_text(iraf_ctod(text)))

    cdelt_text = getattr(strip, "cdelt2_text", "")
    nrows = int(np.asarray(strip.data).shape[0])
    if strip.crval2_text and cdelt_text and nrows > 1:
        # apall's extracted WCS (apmw_open: fused logical->world at pixels 1
        # and n of the strip's dispersion axis), as its header text is re-read.
        from decanter.warptree_chain import apall_dispersion
        w1, dw = apall_dispersion(strip.crval2_text, cdelt_text, nrows)
        return (iraf_ctod(iraf_imputd_text(w1)), iraf_ctod(iraf_imputd_text(dw)))
    return (rt(strip.crval2_text, strip.lambda_min), rt(cdelt_text, strip.dy))


def _truncated_c(it: Intermediates, m: int) -> NDArray[np.float32]:
    """The s11 ``_m###c`` spectrum (truncated, unshifted) of order ``m``.

    Taken from ``it.spectra_trunc`` when the reduction kept it, else computed
    from the 1-D extraction exactly as :func:`_finalize_wavelength` does.
    """
    if m in it.spectra_trunc:
        return np.asarray(it.spectra_trunc[m], dtype=np.float32)
    lambda_min, dy = it.strip_wcs[m]
    h = _astrofits.Header()
    h["CRVAL1"] = lambda_min
    h["CDELT1"] = dy
    h["CRPIX1"] = 1.0
    trunc = scopy_wavelength_truncate(np.asarray(it.spectra_1d[m]), h, w1=1.0, w2=2048.0)
    return np.asarray(trunc.data, dtype=np.float32)


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
    shift_wave: float | dict,
    save_intermediates: bool,
    bg_1d: dict[int, NDArray] | None = None,
    twod: dict[str, dict[int, NDArray]] | None = None,
) -> Reduction:
    """s11–s13: truncate (with optional shift) → dispcor → FSR cut → Reduction.

    Factored out of :func:`reduce` so the multi-frame driver can re-run this
    cheap tail with a per-frame ``shift_wave`` without repeating the expensive
    2D + rectify + extract work. ``strips_lambda`` maps order → the rectified
    strip's ``(lambda_min, dy)`` linear-WCS pair (the only thing the s11
    truncate needs from s06). ``shift_wave``: one s11 shift for every order,
    or ``{order: shift}`` (the default mode's telluric-anchored shifts).

    ``bg_1d`` (WARP's Background Subtraction): per order the background
    spectrum (``_bg``), truncated and dispersion-corrected like the object
    but never shifted (``Warp_sci.py:566-568, 646-648``) into
    ``inter.bg_trunc`` / ``inter.bg_dispcor``.  ``twod`` (Extract 2d
    spectrum): ``{"strip"|"resample": {order: (rows, lines)}}``; every row
    goes through the object's s11-s12 (``scopy`` truncate, the cross-frame
    shift, ``dispcor``: Warp_sci.py:580-584, 671-686) into
    ``inter.twod_trunc`` / ``inter.twod_shifted`` and
    ``Reduction.obj_2d`` / ``obj_2d_resample``.
    """
    from decanter.fast.wavelength import dispcor_edges_fast as edges_fn
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
        trunc = scopy_wavelength_truncate(np.asarray(spec), h, w1=1.0, w2=2048.0)
        cd_c = float(trunc.header.get("CDELT1", 1.0))
        shifted = shift_truncated(trunc.data, cd_c, _shift_for(shift_wave, m))
        hs = trunc.header.copy()
        hs["CRVAL1"] = 1.0
        hs["CRPIX1"] = 1.0
        obj_truncated[m] = (shifted.astype(np.float32), hs)
        if save_intermediates:
            inter.spectra_trunc[m] = np.asarray(trunc.data, dtype=np.float32)
            inter.spectra_shifted[m] = shifted.astype(np.float32)
        if m in sky_1d:
            # WARP never waveshifts the sky path (truncate only); sky shift = 0.
            st = scopy_wavelength_truncate(np.asarray(sky_1d[m]), h, w1=1.0, w2=2048.0)
            sky_truncated[m] = (np.asarray(st.data, dtype=np.float32), st.header)
            if save_intermediates:
                inter.sky_trunc[m] = np.asarray(st.data, dtype=np.float32)

    # --- s12 dispcor per order -------------------------------------------
    obj_dispcor: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    sky_dispcor_d: dict[int, tuple[NDArray, _astrofits.Header]] = {}
    disp_wcs: dict[int, DispcorWCS] = {}
    id_paths = {m: Path(calib.id_dir) / f"id{calib.id_refname}.{m:04d}"
                for m in obj_truncated}
    id_paths = {m: p for m, p in id_paths.items() if p.exists()}
    order_ids = tuple(id_paths[m] for m in sorted(id_paths))
    for m, (trunc_data, trunc_header) in obj_truncated.items():
        if m not in id_paths:
            continue
        id_path = id_paths[m]
        obj_dispcor[m] = dispcor_one_order(trunc_data, trunc_header, id_path,
                                           order_ids=order_ids, edges_fn=edges_fn)
        d_hdr = obj_dispcor[m][1]
        disp_wcs[m] = dispcor_wcs(float(d_hdr["CRVAL1"]), float(d_hdr["CDELT1"]),
                                  int(obj_dispcor[m][0].size))
        if m in sky_truncated:
            sky_data_t, sky_header_t = sky_truncated[m]
            sky_dispcor_d[m] = dispcor_one_order(sky_data_t, sky_header_t, id_path,
                                                 order_ids=order_ids, edges_fn=edges_fn)
    if save_intermediates:
        inter.spectra_dispcor = {m: a for m, (a, _) in obj_dispcor.items()}
        inter.sky_dispcor = {m: a for m, (a, _) in sky_dispcor_d.items()}
        inter.dispcor_wcs = dict(disp_wcs)

    def _hdr_c(m):
        lambda_min, dy = strips_lambda[m]
        h = _astrofits.Header()
        h["CRVAL1"] = lambda_min
        h["CDELT1"] = dy
        h["CRPIX1"] = 1.0
        return h

    # Background Subtraction: _bg -> _bgc (truncate) -> _bgcw (dispcor)
    for m, bg in (bg_1d or {}).items():
        if m not in obj_truncated:
            continue
        bt = scopy_wavelength_truncate(np.asarray(bg), _hdr_c(m), w1=1.0, w2=2048.0)
        inter.bg_trunc[m] = np.asarray(bt.data, dtype=np.float32)
        if m in id_paths:
            hb = bt.header.copy()
            hb["CRVAL1"] = 1.0
            hb["CRPIX1"] = 1.0
            bd, bh = dispcor_one_order(inter.bg_trunc[m], hb, id_paths[m], order_ids=order_ids,
                                       edges_fn=edges_fn)
            inter.bg_dispcor[m] = bd
            inter.bg_dispcor_wcs[m] = dispcor_wcs(float(bh["CRVAL1"]), float(bh["CDELT1"]),
                                                  int(np.asarray(bd).size))

    # Extract 2d spectrum: every row through the object's s11-s12
    obj_2d: dict[str, dict[int, OrderSpectrum2D]] = {"strip": {}, "resample": {}}
    for variant, arrays in (twod or {}).items():
        for m, arr in arrays.items():
            if m not in obj_truncated:
                continue
            rows_c, rows_s, rows_w = [], [], []
            arr = np.asarray(arr, dtype=np.float32)
            for row in arr:
                t = scopy_wavelength_truncate(row, _hdr_c(m), w1=1.0, w2=2048.0)
                tc = np.asarray(t.data, dtype=np.float32)
                cd_c = float(t.header.get("CDELT1", 1.0))
                sh = shift_truncated(tc, cd_c, _shift_for(shift_wave, m),
                                     multispec=arr.shape[0] > 1).astype(np.float32)
                rows_c.append(tc)
                rows_s.append(sh)
                if m in id_paths:
                    hs = t.header.copy()
                    hs["CRVAL1"] = 1.0
                    hs["CRPIX1"] = 1.0
                    rows_w.append(dispcor_one_order(sh, hs, id_paths[m], order_ids=order_ids,
                                                    edges_fn=edges_fn)[0])
            inter.twod_trunc[(variant, m)] = np.array(rows_c, dtype=np.float32)
            inter.twod_shifted[(variant, m)] = np.array(rows_s, dtype=np.float32)
            if rows_w and m in disp_wcs:
                dc = disp_wcs[m]
                obj_2d[variant][m] = OrderSpectrum2D(
                    order=m, flux=np.array(rows_w, dtype=np.float32),
                    crval1=float(dc.w1), cdelt1=float(dc.dw))

    # --- s13 FSR truncate -> final per-cut per-order spectra -------------
    # IRAF re-reads the dispcor WCS from the _csw header text, and the FSR
    # bounds arrive through pyraf's 12-digit real text; see
    # wavelength/iraf_wcs.py. The header WCS IRAF writes is kept with the
    # spectrum for the WARP-layout writer.
    fsr_table = load_fsr(calib.fsr_table)
    obj_out: dict[tuple[float, int], OrderSpectrum] = {}
    sky_out: dict[tuple[float, int], OrderSpectrum] = {}

    def _fsr_cut(data: NDArray, m: int, cut: float) -> OrderSpectrum | None:
        if m not in fsr_table or m not in disp_wcs:
            return None
        lo, hi = fsr_bounds(fsr_table[m].lambda_min, fsr_table[m].lambda_max, float(cut))
        dc = disp_wcs[m]
        sliced, sc = truncate_spectrum_iraf(np.asarray(data, dtype=np.float32),
                                            dc.crval1, dc.cdelt1, lo, hi)
        if sliced.size == 0:
            return None
        cards = {"CRVAL1": sc.crval1, "CRPIX1": sc.crpix1, "CDELT1": sc.cdelt1,
                 "CD1_1": sc.cdelt1, "LTM1_1": sc.ltm1_1}
        if sc.ltv1 is not None:
            cards["LTV1"] = sc.ltv1
        if sc.np2 is not None:
            cards["NP2"] = str(sc.np2)
        return OrderSpectrum(
            order=m, fsr_cut=cut, flux=sliced.astype(np.float32),
            crval1=float(sc.w0), cdelt1=float(sc.wp), crpix1=1.0,
            header_wcs=cards,
        )

    for cut in cfg.cutrange_list:
        for m, (data, hdr) in obj_dispcor.items():
            spec = _fsr_cut(data, m, cut)
            if spec is not None:
                obj_out[(cut, m)] = spec
        for m, (data, hdr) in sky_dispcor_d.items():
            spec = _fsr_cut(data, m, cut)
            if spec is not None:
                sky_out[(cut, m)] = spec

    return Reduction(
        obj_name=objname,
        obj_path=obj_path,
        sky_path=sky_path,
        obj=obj_out,
        sky=sky_out if cfg.flag_skyemission else None,
        intermediates=inter,
        shift_wave=float(shift_wave) if not isinstance(shift_wave, dict) else 0.0,
        obj_2d=obj_2d["strip"],
        obj_2d_resample=obj_2d["resample"],
    )


@dataclass(frozen=True, slots=True)
class TransitSeries:
    """A cross-frame-aligned set of per-frame reductions (a transit time series).

    Attributes:
        reductions: one :class:`Reduction` per frame: on the telluric
            wavelength scale (``mode="default"``), or aligned to
            ``reductions[refid]`` by WARP's cross-frame shift (``"warp-clone"``
            with the shift on).
        shifts: per-frame s11 shift applied (Å; the mean over orders for the
            default mode's per-order shifts), length == n frames.
        refid: index of WARP's reference frame (0 without the WARP shift).
        wavecal: the :class:`decanter.wavecal.AnchorSolution` (default mode).
    """

    reductions: list
    shifts: NDArray
    refid: int
    wavecal: Any = None     # the decanter.wavecal solution (mode="default")


def _with_waveshift(cfg: Config, waveshift: bool | None, mode: str = "warp-clone") -> Config:
    """``cfg`` with WARP's measure / correct shift flags set to ``waveshift``.
    ``None``: the mode's own choice -- off for ``"default"`` (the wavelength
    calibration replaces it), as configured for ``"warp-clone"``."""
    if waveshift is None:
        if mode == "default":
            waveshift = False
        else:
            return cfg
    return dataclasses.replace(cfg, flag_wsmeasure=bool(waveshift),
                               flag_wscorrect=bool(waveshift))


def _wavecal_shifts(reds: list, calib: Calibration, *, cut: float = 1.30):
    """The default mode's wavelength calibration of unshifted reductions: the
    :class:`decanter.wavecal.AnchorSolution`, per frame ``{order: s11 shift}``
    that moves every order onto its true (telluric) wavelengths, and the input
    :class:`~decanter.wavecal.NightSpectra` (for :func:`decanter.wavecal.telluric_check`)."""
    from decanter.wavecal.model import StaticSurface, ThArTemplate, anchor, night_from_reductions

    tmpl = ThArTemplate.from_calibration(calib)
    night = night_from_reductions(reds, tmpl, cut=cut)
    sol = anchor(night, surface=StaticSurface.find(calib, tmpl))
    orders = [m for m in reds[0].intermediates.strip_wcs if m in set(tmpl.orders)]
    shifts = []
    for f in range(len(reds)):
        sh = {}
        for m in orders:
            p = tmpl.p_range(m)
            lam = tmpl.lam(m, np.linspace(p[0], p[1], 64))
            true = sol.true_lambda(m, lam, "star", frames=[f])[0]
            # the s11 shift moves a feature at raw pixel q to label pixel q + shift / dy
            sh[m] = float(np.median(tmpl.p_of_lam(m, true) - tmpl.p_of_lam(m, lam))) * tmpl.dy
        shifts.append(sh)
    return sol, shifts, night


def _apply_wavecal(reds: list, calib: Calibration, cfg: Config, *,
                   save_intermediates: bool = False):
    """Re-run the s11-s13 tail of unshifted reductions with the wavecal's
    per-order shifts: ``(corrected reductions, solution)``."""
    sol, shifts, _ = _wavecal_shifts(reds, calib)
    out = []
    for r, sh in zip(reds, shifts):
        it = r.intermediates
        red = _finalize_wavelength(
            it.spectra_1d, it.sky_1d, it.strip_wcs, r.obj_name, r.obj_path,
            r.sky_path, calib, cfg, it, shift_wave=sh,
            save_intermediates=True,
            twod={"strip": it.strips_2d, "resample": it.strips_2d_resample})
        red.obj_header, red.sky_header = r.obj_header, r.sky_header
        red.apertures, red.aperture_refs = r.apertures, r.aperture_refs
        red.centersearch_undefined = r.centersearch_undefined
        red.meta = dict(getattr(r, "meta", {}) or {}, wavecal_shift=sh)
        out.append(red)
    return out, sol


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
    mode: str = "default",
    waveshift: bool | None = None,
    workdir: str | Path | None = None,
) -> TransitSeries:
    """Reduce a list of ``(obj, sky)`` frame pairs (one night) into a time series.

    Runs :func:`reduce` on each pair once (the expensive part), then re-runs
    only the cheap wavelength tail (s11-s13) of every frame with its shift:

    * ``mode="default"``: no WARP cross-frame shift; the OH + telluric
      wavelength calibration (:mod:`decanter.wavecal`) of the whole night puts
      every frame's spectra on their true telluric wavelength scale.
    * ``mode="warp-clone"``: WARP's cross-frame shift (s10/s11: each frame
      cross-correlated against a reference frame and moved onto its grid), as
      ``config`` says (WARP's default: on). Bit-identical to WARP.

    Args:
        pairs: ``[(obj, sky), ...]`` — paths or arrays, as for :func:`reduce`.
        refid: WARP's reference frame index (``"warp-clone"``); default =
            WARP's choice (largest sum of per-order median counts).
        extract: ``"box"`` or ``"optimal"`` (applied to every frame).
        align: if False, no shift and no wavelength calibration at all.
        mode: ``"default"`` or ``"warp-clone"`` (above).
        waveshift: WARP's cross-frame shift: ``True`` on, ``False`` off,
            ``None`` (default) the mode's choice (``"default"``: off;
            ``"warp-clone"``: ``config``).
        workdir: also write every frame's final spectra as FITS
            (:meth:`Reduction.write_to`) into ``workdir/<object>_NO<i>/``, and
            in the default mode the calibration (per-frame zero points, km/s)
            to ``workdir/decanter_wavecal.json``.

    Returns:
        A :class:`TransitSeries` (``wavecal`` holds the solution in the
        default mode).
    """
    mode = resolve_mode(mode)
    cfg = _with_waveshift(config or Config(), waveshift, mode)
    series = _reduce_series(pairs, calib, cfg, refid=refid, extract=extract,
                            check_calib=check_calib, subtract_background=subtract_background,
                            align=align, mode=mode)
    if workdir is not None:
        _write_series(series, Path(workdir))
    return series


def _write_series(series: TransitSeries, workdir: Path) -> None:
    """``reduce_many(workdir=)``: one directory of final spectra per frame, plus
    the default mode's calibration as JSON."""
    import json

    workdir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(series.reductions, start=1):
        r.write_to(workdir / f"{r.obj_name}_NO{i}")
    sol = series.wavecal
    if sol is not None:
        def arr(a):
            return None if a is None else [None if not np.isfinite(v) else float(v)
                                           for v in np.asarray(a, float)]
        rec = dict(method=sol.method, mode=sol.mode,
                   frames=[f"{r.obj_name}_NO{i}" for i, r in enumerate(series.reductions, 1)],
                   z_star_kms=arr(sol.z_star), z_star_err_kms=arr(sol.z_star_err),
                   z_sky_kms=arr(sol.z_sky), z_sky_err_kms=arr(sol.z_sky_err),
                   s11_shift_per_order=[{str(m): v for m, v in
                                         (r.meta or {}).get("wavecal_shift", {}).items()}
                                        for r in series.reductions])
        (workdir / "decanter_wavecal.json").write_text(json.dumps(rec, indent=1))


def _reduce_series(pairs, calib, cfg, *, refid, extract, check_calib, subtract_background,
                   align, mode) -> TransitSeries:
    base = [
        reduce(o, calib, sky=s, config=cfg, extract=extract,
               check_calib=check_calib, subtract_background=subtract_background,
               shift_wave=0.0, mode="warp-clone")
        for (o, s) in pairs
    ]
    n = len(base)
    if not align:
        return TransitSeries(reductions=base, shifts=np.zeros(n), refid=refid or 0)
    if mode == "default":
        out, sol = _apply_wavecal(base, calib, cfg)
        mean = np.array([np.mean(list(r.meta["wavecal_shift"].values())) for r in out])
        return TransitSeries(reductions=out, shifts=mean, refid=0, wavecal=sol)
    if n < 2:
        return TransitSeries(reductions=base, shifts=np.zeros(n), refid=refid or 0)

    orders = sorted(set.intersection(*[set(r.intermediates.spectra_1d) for r in base]))
    # WARP's s10 exactly (warp/ccwaveshift.py): float32 search on the
    # truncated _m###c spectra, WARP's reference frame and clipping; flagged
    # frames are not shifted, as in WARP.
    from decanter.waveshift.measure import warp_waveshift_log
    rows = [[_truncated_c(r.intermediates, m) for r in base] for m in orders]
    wlog = warp_waveshift_log(rows, orders, float(base[0].intermediates.strip_wcs[orders[0]][1]),
                              measure=cfg.flag_wsmeasure, correct=cfg.flag_wscorrect,
                              refid=refid)
    refid = wlog.refid
    shift_avg = np.asarray(wlog.waveShiftAdopted, dtype=float)

    aligned = []
    for r, sh in zip(base, shift_avg):
        it = r.intermediates
        red = _finalize_wavelength(
            it.spectra_1d, it.sky_1d, it.strip_wcs, r.obj_name, r.obj_path,
            r.sky_path, calib, cfg, it, shift_wave=float(sh),
            save_intermediates=True,
        )
        red.obj_header, red.sky_header = r.obj_header, r.sky_header
        red.apertures, red.aperture_refs = r.apertures, r.aperture_refs
        aligned.append(red)
    return TransitSeries(reductions=aligned, shifts=shift_avg, refid=refid)


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
        F = np.array([np.asarray(s.flux[:L], float) for s in specs])
        if weight == "uniform":
            w = np.ones(len(specs))
        else:
            w = np.array([max(float(np.nanmedian(s.flux[:L])), 1e-9) for s in specs])
        stack = np.average(F, axis=0, weights=w)
        ref = specs[0]
        obj_out[key] = OrderSpectrum(
            order=key[1], fsr_cut=key[0], flux=stack.astype(np.float32),
            crval1=ref.crval1, cdelt1=ref.cdelt1, crpix1=ref.crpix1,
        )
    return Reduction(
        obj_name=reds[0].obj_name, obj_path=None, sky_path=None,
        obj=obj_out, sky=None,
    )
