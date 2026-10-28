"""Reduce a whole night the way WARP does and write WARP's on-disk layout.

:func:`reduce_to_warp_tree` is the multi-frame counterpart of
:func:`decanter.reduce`. It reproduces what ``Warp_sci.py`` does across the
frames of a listfile and leaves a directory that every downstream consumer of
WARP output can read unchanged (``winered_pipeline`` prepare/analyze, the
Paschen-line scripts, ``Calibration.from_dir(tree)``):

* **Frames and numbering.** Every listfile line is one frame, numbered
  ``NO1..N`` in line order; its directory is ``<OBJECT>_NO<i>`` with
  ``OBJECT`` taken from that line's object frame and sanitized exactly as
  ``warp/config.py:readInputDataHeader`` does (so one night can hold several
  prefixes, e.g. ``HAT-P-26_NO37`` next to ``L98-59_NO1``). WARP drops no
  frames.
* **Cross-frame alignment.** WARP's s10 search (``warp/ccwaveshift.py``) is
  run on the ``_m###c`` spectra with WARP's float32 arithmetic and WARP's
  reference frame and clipping, and the adopted shift is applied in s11
  (:func:`decanter.waveshift.measure.warp_waveshift_log`).
* **Products: every file WARP writes** (``Warp_sci.py``, no ``-s``; the
  ``-s`` extras with ``save=True``), each with WARP's data and header text
  (only the time stamps differ):

  - per frame ``<OBJECT>_NO<i>/``: ``onedspec/{VAC,AIR}_{flux,norm,cont}``
    (FSR cut -> ``continuum`` -> ``sarith /`` -> ``dopcor``,
    :mod:`decanter.wavelength.iraf_onedspec`), ``sky_emission/{VAC,AIR}``,
    ``images/spatial_profile/*.{dat,png}``, ``images/2d_image``,
    ``images/badpixmask``, ``intermediate_files/OBJ/4-OBJ_mask``,
    ``OBJ/8A-OBJ-1DSPEC/{1..4}``, ``SKY/7-SKY_dispcor`` (and with ``save``
    every 2-D stage, the cut/rectified strips and ``SKY/5,6``);
  - ``<OBJECT>_sum/``: ``scombine`` average of the frames (2..28 frames) or
    ``scopy`` of frame 1, its continuum/air products and the all-order
    ``*_norm_comb`` (:mod:`decanter.wavelength.iraf_scombine`);
    ``SNR_dat/`` (``SNratio_estimate``), ``spectra_image/``.  With exactly
    28 frames WARP computes the SNR files and ``*_norm_comb`` but does not
    move them, so they are in the tree root (Warp_sci.py:877-890);
  - ``calibration_data/``: the calibration set, the per-frame trans and sky
    aperture databases, ``ap<OBJ>_NO<i>_s`` and ``aplast``;
  - ``reduction_log/``: aperture, waveshift, center-search, cosmic-ray,
    apscatter and cutransform logs, ``count_and_fwhm.png``, ``status.txt``
    (plus decanter's ``decanter_provenance.json``); ``rawdata_image/``;
  - in the tree root, besides the listfile copy, every other entry of the
    calibration set (WARP copies the whole set into its work directory and
    moves only its own files into ``calibration_data/``; e.g. the empty
    ``sys`` of ``calibration_LCO25a_setting5_HIRES-Y100``,
    :func:`calib_set_strays`).

  Names follow WARP's work names: ``_sscf``/``_sscfm`` become ``_sf``/``_sfm``
  with ``Apscatter: no`` (:func:`obj_suffixes`), and an OBJECT longer than 13
  characters is ``OBJECT`` where WARP keeps a ``shortFile`` name.

  The parameter file's other options add WARP's products for them:
  ``Background Subtraction`` (apall background in the listfile's ``bg=``
  regions) the ``_m###_none.####``, ``_bg``, ``_bgc``, ``_bgcw`` spectra of
  ``8A-OBJ-1DSPEC`` and the background panel of ``<OBJ>_VACflux.pdf``;
  ``Extract 2d spectrum`` ``twodspec/{VAC,AIR}`` and ``8B-OBJ-2DSPEC``
  (strip and resampled strip, truncated (``-s``), shifted);
  ``CUTRANSFORM flux`` changes the rectified strips and every product after.
  Headers are built by applying each IRAF task's card edits in turn
  (:mod:`decanter.warptree_chain`), figures by ports of ``warp/plotframes.py``
  (:mod:`decanter.warptree_plots`), other products in
  :mod:`decanter.warptree_products`. Not written: the ``-s``-only IRAF task
  logs ``reduction_log/{null,logfile}``.
* **Cost.** One process pool runs every phase (frames, s11-s13, sums,
  figures).  Beyond the spectra nearly all of a default tree's time goes into
  drawing WARP's figures (about a third of it; ``plots=False`` skips them);
  ``intermediates=False`` (no ``4-OBJ_mask``, ~100 MB per frame) and
  ``rawdata=False`` mainly save disk space.

Example::

    import decanter
    tree = decanter.reduce_to_warp_tree(
        "lists/TOI836.txt", "science_data/2025_02_15",
        "calibration_data/calibration_LCO25a_setting4_HIRES-Y100",
        "/tmp/reductions/TOI836", workers=6)
    tree.waveshift.waveShiftAdopted
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy.io import fits as _afits

from decanter.calib import Calibration, LockedAperturesWarning
from decanter.config import Config
from decanter.utils.iraf_time import fixpix_time as iraf_fixpix_time
from decanter.utils.psf import warp_minpack_label

WARP_VERSION = "3.8.14"     # the Warp_sci.py version decanter's warp mode clones
_OBJECT_NAME_LIMIT = 13     # Warp_sci.py: objectNameLimit (shortFile)
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
               "EXTEND", "BZERO", "BSCALE", "PCOUNT", "GCOUNT"}


# ---------------------------------------------------------------------------
# Listfile + frame naming (warp/config.py:inputDataList / readInputDataHeader)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ListLine:
    """One listfile line, as frame ``NO<i>`` of WARP's reduction: the
    object/sky raw frame names, the options WARP gives that frame (``ws``,
    ``ap``, ``bg``; see :func:`read_warp_listfile`) and the line's own
    tokens."""

    obj: str
    sky: str
    ws: float | None = None        # the frame's ``ws=`` manual shift (None: none given)
    raw: str = ""
    ap: tuple[float, float] | None = None   # the frame's ``ap=low:high`` manual aperture
    bg: str | None = None          # the frame's ``bg=`` background sample (None: INDEF)
    #: every ``ws=`` / ``ap=`` / ``bg=`` token of the line, in order
    ws_tokens: tuple[float, ...] = ()
    ap_tokens: tuple[tuple[float, float], ...] = ()
    bg_tokens: tuple[str, ...] = ()


def warp_flat_tokens(lines: Sequence[ListLine]) -> tuple[list, list, list]:
    """WARP's ``waveshift_man``, ``lowlim_input``/``upplim_input`` and
    ``skysub_region`` lists (warp/config.py:128-145): every token of every
    line appended to one list per kind, a line without ``ws=`` adding
    ``0.``, one without ``bg=`` adding ``"INDEF"`` (none for ``ap=``).
    Frame ``NO<i>`` gets entry ``i - 1`` of each, so a line with two tokens
    of a kind shifts every later frame's entry."""
    ws: list[float] = []
    ap: list[tuple[float, float]] = []
    bg: list[str] = []
    for ln in lines:
        ap.extend(ln.ap_tokens)
        bg.extend(ln.bg_tokens or ("INDEF",))
        ws.extend(ln.ws_tokens or (0.,))
    return ws, ap, bg


def read_warp_listfile(path: Path | str) -> list[ListLine]:
    """Parse a listfile the way ``warp/config.py:inputDataList`` does.

    Every line is a frame (WARP has no comment syntax); only empty lines are
    skipped (WARP would crash on them). The ``ap=``, ``bg=`` and ``ws=``
    tokens are read with WARP's literal ``find``/``lstrip`` parsing, every
    token of every line (a token can count for several kinds, as in WARP).
    WARP appends each token to one list per kind and gives frame ``NO<i>``
    the list's ``i``-th entry (:func:`warp_flat_tokens`;
    warp/config.py:136-145 -- its length check at :147-152 never fires,
    because ``inputDataList`` runs before the parameter file is read,
    Warp_sci.py:163 vs 231), so the returned lines carry those per-frame
    values; a line with more than one token of a kind is reproduced and
    warned about (``UserWarning``): the frames after it get shifted options.
    A frame beyond the ``ap=`` list has ``ap=None`` (WARP fails on it with
    ``Manual Aperture: yes``, see :func:`reduce_to_warp_tree`).
    """
    raw: list[ListLine] = []
    for line in Path(path).read_text().splitlines():
        toks = line.split()
        if not toks:
            continue
        if len(toks) < 2:
            raise ValueError(f"{path}: listfile line needs 'obj sky': {line!r}")
        ws, ap, bg = [], [], []
        for t in toks[2:]:
            if t.find("ap=") != -1:
                a = t.lstrip("ap=")
                ap.append((float(a.split(":")[0]), float(a.split(":")[1])))
            if t.find("bg=") != -1:
                bg.append(t.lstrip("bg="))
            if t.find("ws=") != -1:
                ws.append(float(t.lstrip("ws=")))
        raw.append(ListLine(obj=toks[0], sky=toks[1], raw=line, ws_tokens=tuple(ws),
                            ap_tokens=tuple(ap), bg_tokens=tuple(bg)))
    return resolve_warp_tokens(raw, source=str(path))


def resolve_warp_tokens(lines: Sequence[ListLine], *, source: str = "listfile") -> list[ListLine]:
    """The lines with WARP's per-frame ``ws``/``ap``/``bg`` (entry ``i - 1``
    of :func:`warp_flat_tokens`'s lists), warning about every line that has
    more than one token of a kind."""
    flat_ws, flat_ap, flat_bg = warp_flat_tokens(lines)
    for k, ln in enumerate(lines, start=1):
        for kind, toks in (("ws", ln.ws_tokens), ("ap", ln.ap_tokens), ("bg", ln.bg_tokens)):
            if len(toks) > 1:
                got = {"ws": flat_ws, "ap": flat_ap, "bg": flat_bg}[kind][:len(lines)]
                warnings.warn(
                    f"{source} line {k} ({ln.raw!r}) has {len(toks)} {kind}= tokens: WARP "
                    f"appends every token to one list and gives frame NO<i> its i-th entry "
                    f"(warp/config.py:136-145), so the frames from NO{k + 1} on get shifted "
                    f"{kind}= values; reproduced: " + ", ".join(
                        f"NO{i}={v}" for i, v in enumerate(got, start=1)), stacklevel=3)
    has_ws = any(x.ws_tokens for x in lines)
    return [dataclasses.replace(ln, ws=flat_ws[i] if has_ws else None,
                                ap=flat_ap[i] if i < len(flat_ap) else None,
                                bg=None if flat_bg[i] == "INDEF" else flat_bg[i])
            for i, ln in enumerate(lines)]


#: background limits from this magnitude on are refused as not modelled:
#: IRAF's nint() of them (apskyeval.x:143-145, apextract.x:279-280) leaves
#: 32-bit integer range near 2^31 (saturation / wrap-around)
_BG_INT_LIMIT = 2.0 ** 30
_MAX_REAL = float(np.finfo(np.float32).max)


def check_bg_tokens(lines: list[ListLine], flag_skysub: bool) -> None:
    """Refuse the ``bg=`` regions WARP stops on, before reducing anything.

    ``lines`` are the frames (:func:`read_warp_listfile`: each frame's own
    ``bg``, WARP's flat-list entry).

    * With ``Background Subtraction`` on, ``aperture_plot`` reads every
      comma-separated region of a sample other than ``INDEF`` as
      ``float(r.split(":")[0])``, ``float(r.split(":")[1])``
      (warp/plotframes.py:551-557): a region without ``:`` or with a
      non-number there stops WARP with "ERROR in PSF center search"
      (Warp_sci.py:429-431).  So a one-number token never reaches apall's
      background.
    * Whatever the option, ``write_apdatabase`` ``float()``\\ s every number of
      a sample without ``INDEF`` in it (aperture.py:176-179, decanter's
      ``aperture_solve._bg_coords``) and writes the extremes of them and the
      window as ``xmin``/``xmax`` with ``%.2f``; apall reads those with
      ``dtgetr`` (apdb.x:227-228: ``gargr`` -> ``ctod`` -> REAL): ``inf``,
      ``-inf`` (from ``inf``, ``1e309``, ...) are not numbers to ``ctod``
      ("Error in database field value") and a value beyond the REAL range
      (``1e39``) overflows the conversion and kills ``x_apextract.e``.
      Either way WARP stops, "ERROR in extraction" (Warp_sci.py:592-594),
      for every background mode including ``none``.  (A ``nan`` is never
      an extreme -- Python's ``min``/``max`` keep the finite window -- so
      it only reaches the ``sample`` text, which IRAF parses as it parses
      any text: not refused.)

    Raises:
        ValueError: for a token WARP stops on.
        NotImplementedError: for a region number of magnitude ``>= 2^30``
            (WARP does not stop, but IRAF's ``nint`` of such a limit leaves
            32-bit integer range, which is not modelled).
    """
    from decanter.extract.aperture_solve import _bg_coords
    for ln in lines:
        bg = "INDEF" if ln.bg is None else ln.bg
        if flag_skysub and bg != "INDEF":
            try:
                for r in bg.split(","):
                    float(r.split(":")[0])
                    float(r.split(":")[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"listfile line {ln.raw!r}: WARP's aperture_plot cannot read bg={bg} "
                    "as a:b regions (plotframes.py:551-557); WARP stops "
                    "('ERROR in PSF center search')") from exc
        try:
            nums = _bg_coords(bg)
        except ValueError as exc:
            raise ValueError(
                f"listfile line {ln.raw!r}: WARP's write_apdatabase cannot float() "
                f"every number of bg={bg} (aperture.py:176-179); WARP stops "
                "('ERROR in extraction')") from exc
        for v in nums:
            if v != v:                      # nan: never xmin/xmax (see above)
                continue
            if abs(v) > _MAX_REAL:
                raise ValueError(
                    f"listfile line {ln.raw!r}: bg={bg} makes WARP's database xmin/xmax "
                    f"{v:.2f}, which IRAF's apall cannot read as a REAL (apdb.x:227-228); "
                    "WARP stops ('ERROR in extraction')")
            if abs(v) >= _BG_INT_LIMIT:
                raise NotImplementedError(
                    f"listfile line {ln.raw!r}: bg={bg} has a background limit of magnitude "
                    f">= 2^30; IRAF's integer arithmetic on it (nint, apskyeval.x:143-145) "
                    "leaves 32-bit range, which decanter does not model")


def warp_objname(header) -> str:
    """WARP's per-frame name prefix from the OBJECT keyword.

    WARP equivalent: ``warp/config.py:readInputDataHeader``
    (``header_key_read(prihdr, "object")`` -> ``"N/A"`` when missing).
    """
    try:
        raw = header["OBJECT"]
    except KeyError:
        raw = "N/A"
    return (str(raw).replace(" ", "_").replace(" ", "_").replace("'", "_")
            .replace("\"", "_").replace("#", "_").replace("/", "_"))


def short_name(name: str, objname: str) -> str:
    """WARP's ``shortFile``: long object names become ``OBJECT`` in work files."""
    return name.replace(objname, "OBJECT") if len(objname) > _OBJECT_NAME_LIMIT else name


def obj_suffixes(apscatter: bool) -> tuple[str, str]:
    """WARP's object work-name suffixes after the flat field and fixpix.

    ``("sscf", "sscfm")`` with ``Apscatter: yes``, ``("sf", "sfm")`` without:
    WARP sets ``obj_ssc_list = obj_s_list`` when apscatter is off
    (Warp_sci.py:284-293) and derives every later object name from it -- the
    flat-fielded and fixpix'd frames, the cut and rectified strips, the
    per-frame aperture databases, the spatial profiles, the 2-D figure and
    the cutransform log entry (Warp_sci.py:294-308, 553).
    """
    return ("sscf", "sscfm") if apscatter else ("sf", "sfm")


def _fits_name(raw_dir: Path, frame: str) -> Path:
    p = Path(raw_dir) / f"{frame}.fits"
    if not p.is_file():
        raise FileNotFoundError(f"raw frame {p} not found")
    return p


# ---------------------------------------------------------------------------
# Per-frame aperture database (warp/aperture.py:apertureSet.write_apdatabase)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RefAperture:
    order: int
    center_x: float
    center_y: float
    ap_low: float
    ap_high: float
    ftype: int
    yorder: int
    ymin: float
    ymax: float
    coeffs: tuple[float, ...]


def _read_ref_aperture(path: Path, order: int) -> _RefAperture:
    """``warp/aperture.py:apertureSet.__init__`` for one order (last record)."""
    lines = Path(path).read_text().splitlines()
    begins = [i for i, ln in enumerate(lines) if ln.find("begin") != -1]
    begins.append(len(lines))
    rec = None
    for b0, b1 in zip(begins[:-1], begins[1:]):
        if int(lines[b0 + 2].split()[1]) == order:
            rec = (b0, b1)
    if rec is None:
        raise KeyError(f"order {order} not in {path}")
    cx = cy = lo = hi = None
    coeff: list[float] = []
    for j in range(*rec):
        if lines[j].find("center") != -1:
            cx = float(lines[j].split()[1])
            cy = float(lines[j].split()[2])
            lo = float(lines[j + 1].split()[1])
            hi = float(lines[j + 2].split()[1])
        if lines[j].find("curve") != -1:
            n = int(lines[j].split()[1])
            for k in range(1, n + 1):
                v = float(lines[j + k].split()[0])
                coeff.append(int(v) if k <= 2 else v)
    yorder = int(coeff[1])
    return _RefAperture(order=order, center_x=cx, center_y=cy, ap_low=lo, ap_high=hi,
                        ftype=int(coeff[0]), yorder=yorder, ymin=coeff[2], ymax=coeff[3],
                        coeffs=tuple(coeff[4:4 + yorder]))


def render_frame_apdb(ref: _RefAperture, image: str, ap_low: float, ap_high: float,
                      bgsample: str = "INDEF", timestamp: str | None = None) -> str:
    """Text of WARP's per-frame trans aperture database ``ap<image>``.

    WARP equivalent: ``warp/aperture.py:apertureSet.write_apdatabase`` with
    the reference aperture's ``apLow``/``apHigh`` replaced by the frame's
    aperture (``Warp_sci.py`` extraction block).
    """
    if timestamp is None:
        timestamp = time.strftime("# %a %H:%M:%S %d-%b-%Y", time.gmtime())
    m = ref.order
    apcoord = [ap_low, ap_high]
    if bgsample.find("INDEF") == -1:
        for part in bgsample.split(","):
            apcoord.extend(float(v) for v in part.split(":"))
    s = timestamp + "\n"
    s += "begin\taperture\t%s %d %.2f %.2f\n" % (image, m, ref.center_x, ref.center_y)
    s += "\timage\t%s\n" % image
    s += "\taperture\t%d\n" % m
    s += "\tbeam\t%d\n" % m
    s += "\tcenter\t%.2f %f\n" % (ref.center_x, ref.center_y)
    s += "\tlow\t%.2f %f\n" % (ap_low, -ref.center_y + 1.)
    s += "\thigh\t%.2f %f\n" % (ap_high, ref.center_y + 1.)
    s += "\tbackground\n"
    s += "\t\txmin %.2f\n" % min(apcoord)
    s += "\t\txmax %.2f\n" % max(apcoord)
    s += "\t\tfunction chebyshev\n\t\torder 1\n"
    s += "\t\tsample %s\n" % bgsample
    s += "\t\tnaverage -3\n\t\tniterate 0\n\t\tlow_reject 3.\n\t\thigh_reject 3.\n\t\tgrow 0.\n"
    s += "\taxis 1\n"
    s += "\tcurve %d\n" % (4 + ref.yorder)
    s += "\t\t%f\n" % ref.ftype
    s += "\t\t%f\n" % ref.yorder
    s += "\t\t%f\n" % ref.ymin
    s += "\t\t%f\n" % ref.ymax
    for c in ref.coeffs:
        s += "\t\t%f\n" % c
    s += "\n"
    return s


# ---------------------------------------------------------------------------
# Logs (warp/logger.py)
# ---------------------------------------------------------------------------

def _log_title(title: str, description: str) -> str:
    s = "#" * 72 + "\n" + "### %s %s\n" % (title, "#" * (72 - len(title) - 5)) + "#" * 72 + "\n\n"
    for d in description.split("\n"):
        s += "# %s\n" % d
    return s + "\n"


def warp_logged_window(root: Path | str, low: np.ndarray, high: np.ndarray,
                       orders: Sequence[int]) -> tuple[np.ndarray, np.ndarray, str]:
    """The aperture windows of a tree whose apertures are locked to the WARP
    reduction ``root``, as WARP logged them.

    WARP logs the window its center search gave (``lowlim``/``highlim``,
    floats; Warp_sci.py:547-557, 596; ``writeApertureLogNpz``) and writes
    the per-frame database with ``%.2f`` of them; a locked decanter run only
    has the database tokens.  Where ``root/reduction_log/aperture_log.npz``
    holds the same orders and frames and its value's ``%.2f`` text is the
    database token decanter extracted with, that value is WARP's logged one
    and is used; anything else keeps the database value.

    Returns:
        ``(low, high, note)``; ``note`` says what was used (for the log and
        ``decanter_provenance.json``).
    """
    p = Path(root) / "reduction_log" / "aperture_log.npz"
    if not p.is_file():
        return low, high, f"database %.2f values ({p} not found)"
    try:
        with np.load(p) as z:
            wl = np.asarray(z["apertureLow"])
            wh = np.asarray(z["apertureUpp"])
            wo = [int(o) for o in np.asarray(z["echelleOrder"]).ravel()]
    except (KeyError, ValueError, OSError) as exc:
        return low, high, f"database %.2f values ({p} unreadable: {exc})"
    low = np.asarray(low)
    high = np.asarray(high)
    if wo != [int(o) for o in orders] or wl.shape != low.shape or wh.shape != high.shape:
        return low, high, (f"database %.2f values ({p} has orders/frames {wo[:3]}.../"
                           f"{wl.shape}, this tree {list(orders)[:3]}.../{low.shape})")
    fmt = np.vectorize(lambda v: "%.2f" % v)
    use = (fmt(wl) == fmt(low)) & (fmt(wh) == fmt(high))
    out_l = np.where(use, wl, low).astype(wl.dtype)
    out_h = np.where(use, wh, high).astype(wh.dtype)
    n = int(use.sum())
    note = f"WARP's logged window ({p}) for {n} of {use.size} (frame, order)"
    if n < use.size:
        note += "; the database %.2f values where WARP's text differs"
    return out_l, out_h, note


def write_aperture_log(path_txt: Path, path_npz: Path, low: np.ndarray, high: np.ndarray,
                       orders: Sequence[int]) -> None:
    """``warpLog.writeApertureLogText`` / ``writeApertureLogNpz``."""
    n = low.shape[0]
    s = _log_title("Log of Apertures", "Lower and uppper limit (pix) of the apertures.")
    s += "Files\t" + "".join("No.%d\t" % (i + 1) for i in range(n)) + "\n\n"
    for j, m in enumerate(orders):
        s += "m=%d\t" % m + "".join("%.2f:%.2f\t" % (low[i][j], high[i][j]) + "\t"
                                    for i in range(n)) + "\n"
    Path(path_txt).write_text(s)
    np.savez(path_npz, apertureLow=low, apertureUpp=high,
             echelleOrder=np.array(list(orders)), frameNum=n)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def _num_card(key: str, text: str) -> _afits.Card:
    """A numeric card carrying IRAF's text verbatim (value field cols 11-30)."""
    return _afits.Card.fromstring(f"{key:<8s}= {text:>20s}")


def _apnum(ref: _RefAperture, low: float, high: float) -> str:
    """``APNUM1`` as WARP's apall writes it for the frame aperture ``(low, high)``.

    apall reads the per-frame database WARP wrote (``%.2f`` center and limit
    tokens, IRAF ``ctod`` -> REAL), swaps inverted limits (``apextract.x:139``)
    and stores ``cen + low`` and ``low + (high - low)`` (``apextract.x:1195``);
    ``apmw_saveim`` prints them with ``"%d %d %.2f %.2f"`` (IRAF ``dtoc``).
    """
    from decanter.extract.apsum_exact import apnum_limits
    from decanter.io.apdb import parse_text
    from decanter.utils.iraf_fmt import dtoc3
    entry = parse_text(render_frame_apdb(ref, "apnum", low, high, timestamp="#"))[ref.order]
    a, b = apnum_limits(entry)
    return f"{ref.order} {ref.order} {dtoc3(float(a), 2, 'f')} {dtoc3(float(b), 2, 'f')}"


def _base_header(raw: _afits.Header) -> _afits.Header:
    h = _afits.Header()
    for c in raw.cards:
        if c.keyword in _STRUCTURAL or c.keyword in ("ORIGIN", "IRAF-TLM", "DATE"):
            continue
        h.append(c, bottom=True)
    h["PIPELINE"] = WARP_VERSION
    return h


@dataclass(frozen=True)
class _HeaderContext:
    flat: str
    apsc: str
    comp_stem: str
    fixpix_time: str         # IRAF's cnvdate text (utils/iraf_time.fixpix_time)
    mask: str = ""           # static bad-pixel mask file name
    dy_text: str = ""        # the transform dy as WARP logs it (str of the comp CDELT1)


def _processing_cards(h: _afits.Header, ctx: _HeaderContext, frame_short: str, *,
                      obj: bool) -> None:
    if obj:
        h["APSCATTE"] = "Scattered light subtracted"
        h["APSCMASK"] = ctx.apsc
    h["FLAT"] = ctx.flat
    h["FIXPIX"] = f"{ctx.fixpix_time} Bad pixel file is maskflat_{frame_short}_s"
    h["BP_MASK"] = f"maskflat_{frame_short}_s"


def _wcs_cards(h: _afits.Header, m: int, cards: dict[str, str], apnum: str, ctx: _HeaderContext,
               *, ncombine: bool, dispcor: bool = True) -> None:
    h.append(_num_card("WCSDIM", "1"))
    h.append(_num_card("LTM1_1", cards.get("LTM1_1", "1.")))
    h["WAT0_001"] = "system=equispec"
    h["WAT1_001"] = "wtype=linear label=Wavelength units=angstroms"
    h["DCLOG1"] = "Transform"
    h.append(_num_card("DC-FLAG", "0"))
    h["CTYPE1"] = "LINEAR"
    for key in ("CRVAL1", "CRPIX1", "CDELT1", "CD1_1"):
        h.append(_num_card(key, cards[key]))
    h["APNUM1"] = apnum
    if ncombine:
        h.append(_num_card("NCOMBINE", "1"))
    if dispcor:
        h["DCLOG2"] = f"REFSPEC1 = {ctx.comp_stem}.{m:04d}"
        h["AIRORVAC"] = "vac"
    h["BANDID1"] = "spectrum: background none, weights none, clean no"
    if "LTV1" in cards:
        h.append(_num_card("LTV1", cards["LTV1"]))
    if "NP2" in cards:
        h.append(_num_card("NP2", cards["NP2"]))


def _write(path: Path, data: np.ndarray, header: _afits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = _afits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header)
    # IRAF's FITS kernel writes EXTEND = F after NAXIS1 (astropy drops it
    # from the header passed in, so set it on the HDU).
    hdu.header.set("EXTEND", False, "File may contain extensions", after="NAXIS1")
    hdu.writeto(path, overwrite=True, output_verify="silentfix")


# ---------------------------------------------------------------------------
# Per-frame reduction worker (runs in a process pool)
# ---------------------------------------------------------------------------

@dataclass
class _FramePart:
    """The light, picklable part of a single-frame reduction."""

    index: int
    objname: str
    obj_header: _afits.Header
    sky_header: _afits.Header | None
    spectra_1d: dict
    sky_1d: dict
    strip_wcs: dict
    apertures: dict
    aperture_refs: dict
    elapsed: float = 0.0
    centersearch_undefined: dict = dataclasses.field(default_factory=dict)
    strip_meta: dict = dataclasses.field(default_factory=dict)
    centersearch: dict = dataclasses.field(default_factory=dict)
    slit_profiles: dict = dataclasses.field(default_factory=dict)
    cr_stats: tuple | None = None
    # Background Subtraction: the plain (background none) object sums
    spectra_1d_none: dict = dataclasses.field(default_factory=dict)
    # Extract 2d spectrum: apall strips (rows x lines), their REAL limits,
    # and resample2Dspec's arrays
    strips_2d: dict = dataclasses.field(default_factory=dict)
    strip_2d_limits: dict = dataclasses.field(default_factory=dict)
    strips_2d_resample: dict = dataclasses.field(default_factory=dict)


def _backend_label(name: str | None) -> str | None:
    """``"numpy"``, ``"mlx <version>"``, or None."""
    if name == "mlx":
        try:
            from importlib.metadata import version
            return f"mlx {version('mlx')}"
        except Exception:
            return "mlx"
    return name


def _frame_worker(job: dict) -> _FramePart:
    import warnings

    import decanter
    warnings.filterwarnings("ignore")
    if job.get("fast_backend"):
        # spawn workers do not inherit the parent's set_backend(): pin the
        # backend the parent resolved (and records in the provenance).
        from decanter.fast._backend import set_backend
        set_backend(job["fast_backend"])
    t0 = time.time()
    # the 1-D extractions are always kept; WARP's -s (save) needs every 2-D
    # intermediate, whole strips included
    save_inter = job.get("save", False)
    # the 4-OBJ_mask cosmic-ray extras (mf1, mf2, noise) only when written
    wp = "mask" if (job.get("intermediates") or job.get("save")) else True
    r = decanter.reduce(Path(job["obj"]), job["calib"], sky=Path(job["sky"]),
                        config=job["config"], mode="warp-clone",
                        save_intermediates=save_inter,
                        check_calib=job["check_calib"],
                        aperture=job.get("aperture"), background_region=job.get("bg"),
                        _warp_products=wp)
    it = r.intermediates
    if job.get("dest") is not None:
        _write_frame_2d(job, r)
        if job.get("plots"):
            _plot_frame_2d(job, r)
    return _FramePart(index=job["index"], objname=job["objname"],
                      obj_header=r.obj_header, sky_header=r.sky_header,
                      spectra_1d={m: np.asarray(v, np.float32) for m, v in it.spectra_1d.items()},
                      sky_1d={m: np.asarray(v, np.float32) for m, v in it.sky_1d.items()},
                      strip_wcs=dict(it.strip_wcs), apertures=dict(r.apertures),
                      aperture_refs={m: str(p) for m, p in r.aperture_refs.items()},
                      elapsed=time.time() - t0,
                      centersearch_undefined=dict(r.centersearch_undefined),
                      strip_meta=dict(it.strip_meta), centersearch=dict(it.centersearch),
                      slit_profiles=dict(it.slit_profiles), cr_stats=it.cr_stats,
                      spectra_1d_none={m: np.asarray(v, np.float32)
                                       for m, v in it.spectra_1d_none.items()},
                      strips_2d=dict(it.strips_2d), strip_2d_limits=dict(it.strip_2d_limits),
                      strips_2d_resample=dict(it.strips_2d_resample))


def _frame_uas(job: dict, obj_header, sky_header) -> dict:
    """The UAs of the per-frame 2-D chain (:mod:`decanter.warptree_chain`)."""
    from decanter.warptree_chain import ua_dispaxis, ua_fixpix, ua_flat, ua_ssc
    from decanter.warptree_io import raw_ua
    ctx: _HeaderContext = job["ctx"]
    cfg: Config = job["config"]
    short = short_name(f"{job['objname']}_NO{job['index']}", job["objname"])
    # pyfixpix gets the maskflat work name, or conf.mask_file (the static
    # mask's file name, .fits included) without cosmic-ray detection
    bp = f"maskflat_{short}_s" if cfg.flag_bpmask else ctx.mask
    ua: dict = {}
    ua["obj"] = raw_ua(obj_header, pipeline=WARP_VERSION)
    ua["s"] = ua["obj"]
    ua["ssc"] = ua_ssc(ua["s"], ctx.apsc) if cfg.flag_apscatter else ua["s"]
    ua["sscf"] = ua_flat(ua["ssc"], ctx.flat)
    ua["sscfm"] = ua_dispaxis(ua_fixpix(ua["sscf"], ctx.fixpix_time, bp))
    if sky_header is not None:
        ua["sky"] = raw_ua(sky_header, pipeline=WARP_VERSION)
        ua["f"] = ua_flat(ua["sky"], ctx.flat)
        ua["fm"] = ua_dispaxis(ua_fixpix(ua["f"], ctx.fixpix_time, bp))
    return ua


_STATIC_MASK: dict = {}


def _static_mask_data(path) -> np.ndarray:
    """The static bad-pixel mask's pixels as ``fits.open(...)[0].data`` gives
    them (read once per process)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _STATIC_MASK:
        _STATIC_MASK.clear()
        with _afits.open(p) as h:
            _STATIC_MASK[key] = np.array(h[0].data)
    return _STATIC_MASK[key]


def _plot_frame_2d(job: dict, r) -> None:
    """``images/2d_image/*_sscfm.png`` and ``images/badpixmask/mask_*.{png,pdf}``
    (Warp_sci.py:351, 791-798).

    WARP draws them from the frame's FITS files (the float32 ``_sscfm`` and
    the int16 CR mask); the same arrays are drawn here, without the files'
    round trip, and the multihole geometry is computed once per process.
    """
    from decanter.warptree_plots import cosmicRay2dImages, plot_2dimages, plot_2dimages_mask
    it = r.intermediates
    cfg: Config = job["config"]
    name, i = job["objname"], job["index"]
    short = short_name(f"{name}_NO{i}", name)
    fm = obj_suffixes(cfg.flag_apscatter)[1]
    fdir = Path(job["dest"]) / f"{name}_NO{i}" / "images"
    (fdir / "2d_image").mkdir(parents=True, exist_ok=True)
    plot_2dimages(np.asarray(it.obj_sscfm, dtype=np.float32),
                  fdir / "2d_image" / f"{name}_NO{i}_{fm}.png", f"{short}_{fm}.fits")
    if cfg.flag_bpmask and it.cr_mask is not None:
        (fdir / "badpixmask").mkdir(parents=True, exist_ok=True)
        cal = job["calib"]
        crm = np.asarray(it.cr_mask, dtype=np.int16)
        cosmicRay2dImages(crm, fdir / "badpixmask" / f"mask_{name}_NO{i}_s.pdf", f"mask_{short}_s",
                          cal.apdb_multihole, _static_mask_data(cal.static_bp_mask))
        plot_2dimages_mask(crm, fdir / "badpixmask" / f"mask_{name}_NO{i}_s.png",
                           f"mask_{short}_s.fits")


def _write_frame_2d(job: dict, r) -> None:
    """The frame's 2-D files: ``4-OBJ_mask`` (always) and, with ``save``,
    every other 2-D intermediate (``Warp_sci.py`` clean-up block)."""
    from decanter.warptree_chain import ua_cut, ua_trans
    from decanter.warptree_io import raw_ua, write_astropy_image, write_iraf_fits
    if not (job.get("intermediates") or job.get("save")):
        return
    it = r.intermediates
    cfg: Config = job["config"]
    name, i = job["objname"], job["index"]
    fdir = Path(job["dest"]) / f"{name}_NO{i}"
    io_ = fdir / "intermediate_files" / "OBJ"
    is_ = fdir / "intermediate_files" / "SKY"
    for d in ("1-OBJ_sky_subs", "2-OBJ_scatter_subs", "3-OBJ_flat", "4-OBJ_mask", "5-OBJ_cut",
              "6-OBJ_transform", "7-OBJ_mask2"):
        (io_ / d).mkdir(parents=True, exist_ok=True)
    if cfg.flag_skyemission:
        for d in ("1-SKY_flat", "2-SKY_mask", "3-SKY_cut", "4-SKY_transform", "5-SKY_extract",
                  "6-SKY_truncate", "7-SKY_dispcor"):
            (is_ / d).mkdir(parents=True, exist_ok=True)
    ua = _frame_uas(job, r.obj_header, r.sky_header)
    f_sfx, fm_sfx = obj_suffixes(cfg.flag_apscatter)
    mo = io_ / "4-OBJ_mask"
    write_iraf_fits(mo / f"{name}_NO{i}_{fm_sfx}.fits", it.obj_sscfm, ua["sscfm"])
    if cfg.flag_bpmask and it.cr_mask is not None:
        write_astropy_image(mo / f"mask_{name}_NO{i}_s.fits", np.asarray(it.cr_mask, np.int16))
        mask_hdr = _afits.getheader(job["calib"].static_bp_mask)
        write_iraf_fits(mo / f"maskflat_{name}_NO{i}_s.fits", it.extra["maskflat"],
                        raw_ua(mask_hdr))
        for key in ("mf1", "mf2", "noise"):
            if key in it.extra:
                write_astropy_image(mo / f"{key}_{name}_NO{i}_s.fits", it.extra[key])
    if not job.get("save"):
        return
    write_iraf_fits(io_ / "1-OBJ_sky_subs" / f"{name}_NO{i}_s.fits", it.obj_s, ua["s"])
    if cfg.flag_apscatter:
        write_iraf_fits(io_ / "2-OBJ_scatter_subs" / f"{name}_NO{i}_ssc.fits", it.obj_ssc,
                        ua["ssc"])
        write_iraf_fits(io_ / "2-OBJ_scatter_subs" / f"scatter_{name}_NO{i}_s.fits",
                        it.scatter_model, ua["s"])
    write_iraf_fits(io_ / "3-OBJ_flat" / f"{name}_NO{i}_{f_sfx}.fits", it.obj_sscf, ua["sscf"])
    meta = it.strip_meta
    for m, strip in it.strips_obj.items():
        crval2, cdelt2, _nrows, xmin, xmax = meta[m]
        cut = ua_cut(ua["sscfm"], xmin)
        write_iraf_fits(io_ / "5-OBJ_cut" / f"{name}_NO{i}_{fm_sfx}_m{m}cut.fits",
                        np.asarray(it.obj_sscfm)[:, xmin - 1:xmax], cut)
        write_iraf_fits(io_ / "6-OBJ_transform" / f"{name}_NO{i}_{fm_sfx}_m{m}trans.fits", strip,
                        ua_trans(cut, crval2, cdelt2))
    if cfg.flag_skyemission and it.sky_fm is not None:
        write_iraf_fits(is_ / "1-SKY_flat" / f"{name}_skyNO{i}_f.fits", it.sky_f, ua["f"])
        write_iraf_fits(is_ / "2-SKY_mask" / f"{name}_skyNO{i}_fm.fits", it.sky_fm, ua["fm"])
        for m, strip in it.strips_sky.items():
            crval2, cdelt2, _nrows, xmin, xmax = meta[m]
            cut = ua_cut(ua["fm"], xmin)
            write_iraf_fits(is_ / "3-SKY_cut" / f"{name}_skyNO{i}_fm_m{m}cut.fits",
                            np.asarray(it.sky_fm)[:, xmin - 1:xmax], cut)
            write_iraf_fits(is_ / "4-SKY_transform" / f"{name}_skyNO{i}_fm_m{m}trans.fits",
                            strip, ua_trans(cut, crval2, cdelt2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class WarpTreeFrame:
    """One frame of a written tree."""

    index: int                 # 1-based WARP frame number (NO<i>)
    objname: str               # directory prefix (sanitized OBJECT)
    obj_raw: str
    sky_raw: str
    shift: float               # adopted waveshift applied in s11 (Å)

    @property
    def dirname(self) -> str:
        return f"{self.objname}_NO{self.index}"


@dataclass
class WarpTree:
    """Result of :func:`reduce_to_warp_tree`."""

    root: Path
    frames: list[WarpTreeFrame]
    waveshift: object            # decanter.waveshift.measure.WarpWaveshiftLog
    orders: tuple[int, ...]
    cuts: tuple[float, ...]
    mode: str
    aperture_source: str
    elapsed: float = 0.0
    reductions: list = field(default_factory=list, repr=False)

    def vac_flux(self, index: int, order: int, cut: float = 1.30) -> Path:
        f = self.frames[index - 1]
        return (self.root / f.dirname / "onedspec" / "VAC_flux" / f"fsr{cut:.2f}"
                / f"{f.dirname}_m{order}_fsr{cut:.2f}_VAC.fits")


def _frame_db_stem(objname: str, index: int, apscatter: bool = True) -> str:
    """Per-frame aperture DB image stem, ``<obj>_NO<i>_sscfm`` (``_sfm``
    without apscatter, :func:`obj_suffixes`).

    WARP passes the ``shortFile`` work name to ``write_apdatabase``
    (Warp_sci.py:553), so for an OBJECT longer than 13 characters both the
    file name and the ``begin``/``image`` lines use ``OBJECT`` instead; only
    FITS products are renamed back to the long name when moved (Warp_sci.py:95).
    """
    return short_name(f"{objname}_NO{index}_{obj_suffixes(apscatter)[1]}", objname)


def _frame_trans_dbs(root: Path, objname: str, index: int,
                     apscatter: bool = True) -> dict[int, Path]:
    """Frame ``index``'s per-order trans aperture databases in a WARP tree.

    The names of a reduction made with ``Apscatter: yes`` (``_sscfm_``) or
    ``no`` (``_sfm_``) are both recognised; ``apscatter`` (this reduction's
    setting) says which one is tried first.
    """
    db = Path(root) / "calibration_data" / "database"
    for ap in (apscatter, not apscatter):
        fm = obj_suffixes(ap)[1]
        out: dict[int, Path] = {}
        for cand in db.glob(f"ap{_frame_db_stem(objname, index, ap)}_m*trans"):
            mm = re.search(rf"_{fm}_m(\d+)trans$", cand.name)
            if mm:
                out[int(mm.group(1))] = cand
        if out:
            return out
    return {}


def _resolve_calib(calib) -> tuple[Calibration, Path | None]:
    if isinstance(calib, Calibration):
        return calib, None
    p = Path(calib)
    with warnings.catch_warnings():
        # the NO1 DBs a reduction root carries are dropped below (apertures=)
        warnings.simplefilter("ignore", LockedAperturesWarning)
        return Calibration.from_dir(p), p


def _calib_set_dir(calib: Calibration, src_dir: Path | None) -> Path | None:
    """The directory WARP's ``-c`` names: the one holding ``input_files.txt``
    (a reduction root's ``calibration_data/``)."""
    for d in (x for x in (src_dir, Path(calib.comp).parent) if x is not None):
        d = d / "calibration_data" if (d / "calibration_data" / "input_files.txt").is_file() else d
        if (d / "input_files.txt").is_file():
            return d
    return None


#: what Warp_sci.py:1099-1108 moves from the work directory into
#: calibration_data/ (or the trash) besides input_files.txt's flat, mask, comp
_CALIB_MOVED = ("database", "input_files.txt", "calibration_parameters.txt", "readme.txt")
#: top-level names WARP's run uses itself (a calibration-set entry of that name
#: makes WARP stop, mixes it into a product, or is moved by WARP's clean-up:
#: ``null``, ``logfile`` and ``winered_logo.pdf`` go to reduction_log/,
#: Warp_sci.py:1125-1130)
_ROOT_NAMES = ("calibration_data", "reduction_log", "rawdata_image", "spectra_image", "SNR_dat",
               "slit_viewer", "Trash", "null", "logfile", "winered_logo.pdf")


def calib_set_strays(set_dir: Path | None, skip: Sequence[str] = ()) -> list[Path]:
    """Entries of a calibration set that WARP copies to the tree root and leaves there.

    ``Warp_sci.py:202-209`` copies every entry of ``glob(calibpath + "*")``
    (no dot files) into the work directory -- unless that name is already
    there (``skip``: the listfile copy, ``status.txt``, the raw frames) --
    files with ``shutil.copy``, directories with ``shutil.copytree``;
    ``Warp_sci.py:1099-1108`` then moves only the flat, mask and comp files
    named by ``input_files.txt`` (lines 2-4, first token), ``database``,
    ``input_files.txt``, ``calibration_parameters.txt`` and ``readme.txt``
    into ``calibration_data/`` (the flat and mask to the trash without
    ``-s``).  Any other file or directory -- e.g. the empty ``sys`` of
    ``calibration_LCO25a_setting5_HIRES-Y100`` -- stays in the tree root.
    Entries that are neither a file nor a directory (a broken link) are not
    copied (``os.path.isfile`` / ``isdir``).
    """
    if set_dir is None:
        return []
    set_dir = Path(set_dir)
    toks = [ln.split() for ln in (set_dir / "input_files.txt").read_text().splitlines()]
    named = {t[0] for t in toks[1:4] if t}                  # warp/config.py:readInputCalib
    keep = set(_CALIB_MOVED) | named | set(skip)
    return sorted(p for p in set_dir.iterdir()
                  if not p.name.startswith(".") and p.name not in keep
                  and (os.path.isfile(p) or os.path.isdir(p)))


def _copy_calibration_strays(strays: Sequence[Path], dest: Path) -> list[str]:
    """Copy :func:`calib_set_strays` to the tree root as WARP does."""
    out = []
    for p in strays:
        if os.path.isfile(p):
            shutil.copy(p, dest)                 # content and mode bits, a new mtime
        else:
            shutil.copytree(p, dest / p.name)
        out.append(p.name)
    return out


def _copy_calibration(calib: Calibration, src_dir: Path | None, dest: Path, *,
                      save: bool = False) -> None:
    """``calibration_data/``: input_files.txt, the comp file and the database
    (plus, with ``save``, the flat and the static mask, Warp_sci.py:1100-1102).

    Per-frame databases of some other reduction (``ap*_NO<i>_*``,
    ``ap*_skyNO<i>_*``, ``aplast``) are not copied.
    """
    cd = dest / "calibration_data"
    (cd / "database").mkdir(parents=True, exist_ok=True)
    d = _calib_set_dir(calib, src_dir)
    if d is not None:
        shutil.copy2(d / "input_files.txt", cd / "input_files.txt")
        for extra in ("calibration_parameters.txt", "readme.txt"):
            if (d / extra).is_file():
                shutil.copy2(d / extra, cd / extra)
    shutil.copy2(calib.comp, cd / Path(calib.comp).name)
    if save:
        shutil.copy2(calib.flat, cd / Path(calib.flat).name)
        shutil.copy2(calib.static_bp_mask, cd / Path(calib.static_bp_mask).name)
    pat = re.compile(r"^ap.*_(sky)?NO\d+_")
    for f in Path(calib.fc_dir).iterdir():
        if f.is_file() and not pat.match(f.name):
            shutil.copy2(f, cd / "database" / f.name)


def reduce_to_warp_tree(
    listfile: Path | str | Sequence[tuple[str, str]],
    raw_dir: Path | str,
    calib: Calibration | Path | str,
    dest_dir: Path | str,
    *,
    mode: str = "default",
    config: Config | None = None,
    paramfile: Path | str | None = None,
    apertures: str | Path = "solve",
    workers: int = 1,
    intermediates: bool = True,
    overwrite: bool = False,
    check_calib: bool = True,
    listname: str | None = None,
    log: Callable[[str], None] | None = None,
    save: bool = False,
    rawdata: bool = True,
    plots: bool = True,
    waveshift: bool | None = None,
    wavecal_check: bool = True,
) -> WarpTree:
    """Reduce every frame of a listfile and write WARP's directory layout.

    Args:
        listfile: WARP listfile (one ``obj sky`` pair per line) or a list of
            ``(obj, sky)`` raw-frame names.
        raw_dir: directory holding the raw ``<frame>.fits`` files.
        calib: a :class:`Calibration`, a calibration-set directory, or a WARP
            reduction root (its NO1 per-frame databases are *not* used unless
            ``apertures`` says so).
        dest_dir: output root (``reductions/<testname>``). Must not exist
            unless ``overwrite`` (then it is replaced).
        mode: ``"default"``: WARP's pipeline without its cross-frame shift,
            then the night's OH + telluric wavelength calibration
            (:mod:`decanter.wavecal`) applied in place through the s11 shift,
            per frame and order, so every product of the tree (VAC / AIR /
            norm / cont, the sums) is on the true telluric wavelength scale;
            ``reduction_log/`` gets ``decanter_wavecal.json`` (per-frame
            velocities) and ``decanter_wavecal_telluric.png`` (a strong H2O
            line in every frame before / after). ``"warp-clone"``: WARP's
            tree to the byte (with ``waveshift=None`` or ``True`` and WARP's
            options). Each spectrum's ``DECANTER`` card names the mode.
        wavecal_check: default mode: also draw
            ``reduction_log/decanter_wavecal_telluric.png``
            (:func:`decanter.wavecal.telluric_check`).
        config / paramfile: pipeline configuration; ``paramfile`` is a WARP
            ``-p`` parameter file (:meth:`Config.from_parameter_file`).
        apertures: ``"solve"`` (decanter finds each frame's aperture, as WARP
            does) or a WARP reduction root whose per-frame trans aperture
            databases are used instead (locked extraction, for parity runs).
        workers: processes for the per-frame reductions.
        intermediates: write the intermediate files WARP always keeps
            (``intermediate_files/OBJ/4-OBJ_mask`` -- six 2048x2048 images per
            frame --, ``OBJ/8A-OBJ-1DSPEC/{1..4}``, ``SKY/7-SKY_dispcor``).
            ``False`` skips them (saves ~100 MB per frame).
        save: WARP's ``-s``: also every other intermediate (the 2-D stages,
            the per-order cut and rectified strips, ``SKY/5,6``) and the flat
            and static mask in ``calibration_data/``. On the MLX backend the
            rectification then runs on the NumPy backend (the MLX backend's
            float32 subnormals off the extraction bands would show in the
            written strips).
        rawdata: copy the raw frames into ``rawdata_image/`` (WARP does).
        plots: draw WARP's figures (``*.png``/``*.pdf``,
            :mod:`decanter.warptree_plots`); ``False`` skips them.
        waveshift: WARP's cross-frame shift (s10/s11): ``True`` on, ``False``
            off, ``None`` (default) the mode's choice -- off in ``"default"``
            (the wavelength calibration replaces it), ``config`` / ``paramfile``
            (``Measure Shift`` / ``Correct Shift``; WARP's default: on) in
            ``"warp-clone"``.
        listname: file name for the listfile copy when ``listfile`` is a list
            of pairs (default ``<dest name>.txt``).
        log: progress callback (default: print, flushed). Pass
            ``lambda s: None`` to silence.

    Returns:
        A :class:`WarpTree` describing what was written.

    WARP equivalent: ``Warp_sci.py:Warp_sci`` (``--noreport``; ``-s`` with
    ``save=True``). Not written: the ``-s``-only ``reduction_log/null`` and
    ``reduction_log/logfile`` (IRAF task logs: user, host, times, temporary
    file names). The calibration set's entries other than WARP's own files
    are copied to the tree root as WARP leaves them (:func:`calib_set_strays`;
    one with the name of a tree product raises ``ValueError``).
    """
    t_start = time.time()
    start_ctime = time.ctime()
    say = log or (lambda s: print(s, flush=True))
    if config is not None and paramfile is not None:
        raise ValueError("pass config or paramfile, not both")
    cfg = Config.from_parameter_file(Path(paramfile)) if paramfile else (config or Config())
    from decanter.api import _with_waveshift, check_supported, resolve_mode
    mode = resolve_mode(mode)
    cfg = _with_waveshift(cfg, waveshift, mode)
    check_supported(cfg)

    if isinstance(listfile, (str, os.PathLike)):
        lines = read_warp_listfile(listfile)
    else:
        lines = [ListLine(obj=o, sky=s, raw=f"{o} {s}") for o, s in listfile]
    if not lines:
        raise ValueError("empty listfile")
    if cfg.flag_manual_aperture and any(ln.ap is None for ln in lines):
        # WARP indexes its flat ap= list by frame (Warp_sci.py:421-422): a
        # frame beyond it stops WARP ("ERROR in PSF center search").
        raise ValueError("Manual Aperture: yes needs an ap=low:high token for every "
                         "frame (WARP gives frame NO<i> the i-th ap= token of the listfile)")
    check_bg_tokens(lines, cfg.flag_skysub)

    dest = Path(dest_dir)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} exists (WARP aborts too); pass overwrite=True")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if isinstance(listfile, (str, os.PathLike)):
        list_copy = Path(listfile).name
        shutil.copy2(listfile, dest / list_copy)
    else:
        list_copy = listname or f"{dest.name}.txt"
        (dest / list_copy).write_text("\n".join(ln.raw for ln in lines) + "\n")

    cal, cal_src = _resolve_calib(calib)
    raw_dir = Path(raw_dir)
    lock_root = None if (isinstance(apertures, str) and apertures == "solve") else Path(apertures)

    from decanter.api import background_mode
    from decanter.fast._backend import active_backend
    fast_backend = active_backend()
    if fast_backend == "mlx" and (save or background_mode(cfg) != "none"
                                  or cfg.flag_extract2d):
        # The MLX rectification differs from WARP in float32 subnormals of
        # strip pixels outside every extraction band;
        # -s writes those whole strips, and the background regions and
        # 2-D strips may read them, so rectify on the CPU (exact).
        fast_backend = "numpy"

    flat_name = Path(cal.flat).name
    apsc_name = Path(cal.apdb_apsc).name[2:]
    # FIXPIX's run time: IRAF's cnvdate (clktime(0)) (t_fixpix.x:124), e.g. 'Oct 22  2:52':
    # hour space-padded, IRAF's clock (local standard time; utils/iraf_time.py)
    ctx = _HeaderContext(flat=flat_name, apsc=apsc_name, comp_stem=Path(cal.comp).stem,
                         fixpix_time=iraf_fixpix_time(),
                         mask=Path(cal.static_bp_mask).name,
                         dy_text=str(float(_afits.getheader(cal.comp)["CDELT1"])))
    _copy_calibration(cal, cal_src, dest, save=save)
    # the calibration set's other entries, which WARP leaves in the tree root;
    # the names already there when WARP copies them are not copied
    strays = calib_set_strays(_calib_set_dir(cal, cal_src),
                              skip=[list_copy, "status.txt"]
                              + [f"{f}.fits" for ln in lines for f in (ln.obj, ln.sky)])

    jobs = []
    names = []
    for i, ln in enumerate(lines, start=1):
        objp = _fits_name(raw_dir, ln.obj)
        skyp = _fits_name(raw_dir, ln.sky)
        objname = warp_objname(_afits.getheader(objp))
        names.append(objname)
        if lock_root is not None:
            dbs = _frame_trans_dbs(lock_root, objname, i, cfg.flag_apscatter)
            if not dbs:
                raise FileNotFoundError(
                    f"no per-frame trans DBs ap{_frame_db_stem(objname, i, cfg.flag_apscatter)}"
                    f"_m*trans under {lock_root}")
            c_i = dataclasses.replace(cal, trans_apdbs=dbs)
        else:
            c_i = dataclasses.replace(cal, trans_apdbs=None)
        jobs.append(dict(index=i, obj=str(objp), sky=str(skyp), objname=objname,
                         calib=c_i, config=cfg, mode=mode, check_calib=check_calib,
                         aperture=ln.ap if cfg.flag_manual_aperture else None,
                         bg=ln.bg, fast_backend=fast_backend, dest=str(dest), ctx=ctx,
                         intermediates=intermediates, save=save, plots=plots))

    reserved = (set(_ROOT_NAMES) | {f"{nm}_NO{i}" for i, nm in enumerate(names, start=1)}
                | {f"{names[0]}_sum"})
    clash = [p.name for p in strays if p.name in reserved or p.name.startswith("SNratio_")
             or "_norm_comb" in p.name]
    if clash:
        raise ValueError(f"calibration set entries {clash} have the names of WARP-tree products: "
                         "WARP copies them to the tree root (Warp_sci.py:202-209), where its own "
                         "products of that name then fail or mix with them")
    root_strays = _copy_calibration_strays(strays, dest)
    say(f"[decanter] {len(jobs)} frames -> {dest} (mode={mode}, workers={workers}, "
        f"apertures={'solved' if lock_root is None else 'locked to ' + str(lock_root)})")
    # one process pool for every phase (frames, s11-s13, sums, figures)
    pool = _Pool(workers)
    try:
        parts: dict[int, _FramePart] = {}
        for part in pool.map(_frame_worker, jobs):
            parts[part.index] = part
            say(f"[decanter]   NO{part.index} {part.objname} ({part.elapsed:.0f} s)")
        ordered = [parts[i] for i in range(1, len(jobs) + 1)]

        # --- s10: cross-frame shift, WARP-exact -----------------------------
        from decanter.utils.iraf_scopy import scopy_wavelength_truncate
        from decanter.waveshift.measure import warp_waveshift_log
        orders = tuple(sorted(set.intersection(*[set(p.spectra_1d) for p in ordered])))
        c_spectra: list[list[np.ndarray]] = []
        cdelt_c = None
        for m in orders:
            row = []
            for p in ordered:
                lam, dy = p.strip_wcs[m]
                h = _afits.Header()
                h["CRVAL1"] = lam
                h["CDELT1"] = dy
                h["CRPIX1"] = 1.0
                t = scopy_wavelength_truncate(p.spectra_1d[m], h, w1=1.0, w2=2048.0)
                row.append(np.asarray(t.data, dtype=np.float32))
                if cdelt_c is None:
                    cdelt_c = float(t.header["CDELT1"])
            c_spectra.append(row)
        # WARP's conf.waveshift_man: one entry per line, every ws= token of a
        # line with several (warp_flat_tokens); frame NO<i> uses entry i-1
        flat_ws = warp_flat_tokens(lines)[0]
        wlog = warp_waveshift_log(c_spectra, orders, cdelt_c, measure=cfg.flag_wsmeasure,
                                  correct=cfg.flag_wscorrect, n_manual=len(flat_ws))
        adopted = np.asarray(wlog.waveShiftAdopted, dtype=float)
        if cfg.flag_wsmanual:
            adopted = np.array(flat_ws, dtype=float)       # shift_cor = conf.waveshift_man
            wlog = dataclasses.replace(wlog, waveShiftAdopted=adopted)
        applied = (adopted[:len(lines)] if (len(lines) > 1 and cfg.flag_wscorrect)
                   else np.zeros(len(lines)))
        say(f"[decanter] waveshift: refid NO{wlog.refid + 1}, "
            f"{int(np.count_nonzero(applied))} frames shifted")
        wavecal = None
        if mode == "default":
            # the night's OH + telluric wavelength calibration on the unshifted
            # spectra; its per-order shifts go through WARP's s11 shift
            say("[decanter] wavelength calibration (OH + telluric)")
            wavecal, applied, wc_night = _tree_wavecal(ordered, cal, cfg)

        # --- s11-s13 + 1-D products per frame ---------------------------------
        stamp = time.strftime("# %a %H:%M:%S %d-%b-%Y", time.gmtime())
        cuts = tuple(cfg.cutrange_list)
        shifted = (len(lines) > 1 and cfg.flag_wscorrect) or wavecal is not None
        n = len(lines)
        # what <OBJ>_sum is made of: every frame's VAC (2..28 frames, scombine)
        # or frame 1's six products (scopy); handed over in memory
        sum_keys = ("VAC",) if 1 < n <= FRAME_NUMBER_LIMIT else tuple(_SUFFIX)
        # the VACflux figure of a background-subtracted night reads the _bgcw
        # spectra; without the intermediates they go to a scratch directory
        bg_plot_dir = (dest / ".decanter_bgcw" if cfg.flag_skysub and plots
                       and not (intermediates or save) else None)
        wjobs = [dict(part=p, shift=applied[k] if wavecal is not None else float(applied[k]), cal=cal, cfg=cfg, ctx=ctx, dest=str(dest),
                      stamp=stamp, orders=orders, locked=lock_root is not None, mode=mode,
                      ncombine=shifted, intermediates=intermediates, save=save,
                      bg_plot_dir=None if bg_plot_dir is None else str(bg_plot_dir),
                      bg=lines[k].bg or "INDEF", fast_backend=fast_backend, plots=plots,
                      manual_ap=lines[k].ap if cfg.flag_manual_aperture else None,
                      sum_keys=sum_keys if (1 < n <= FRAME_NUMBER_LIMIT or k == 0) else ())
                 for k, p in enumerate(ordered)]
        written = list(pool.map(_write_frame, wjobs))
        ap_low = np.array([w[0] for w in written])
        ap_high = np.array([w[1] for w in written])
        reds = [w[2] for w in written]
        frames_out = [WarpTreeFrame(index=p.index, objname=p.objname, obj_raw=ln.obj,
                                    sky_raw=ln.sky, shift=(float(np.mean(list(applied[k].values())))
                                                           if wavecal is not None
                                                           else float(applied[k])))
                      for k, (ln, p) in enumerate(zip(lines, ordered, strict=True))]
        say(f"[decanter] wrote {len(frames_out)} frame directories")

        # --- <OBJ>_sum, SNR_dat, raw data, figures ----------------------------
        rep = warp_objname(_afits.getheader(_fits_name(raw_dir, lines[0].obj)))
        images = sorted({ln.obj for ln in lines} | {ln.sky for ln in lines})
        night = []
        if rawdata:
            night += [pool.submit(_rawdata_job, str(dest / "rawdata_image"),
                                  str(_fits_name(raw_dir, nm)), plots) for nm in images]
        if plots:
            night += [pool.submit(_night_plot_job, t)
                      for t in _night_plot_tasks(dest, rep, frames_out, orders, cuts, cfg, cal,
                                                 ordered, bg_plot_dir)]
        snr = _write_sums(pool, dest, rep, frames_out, orders, cuts, [w[3] for w in written],
                          mode=mode)
        say(f"[decanter] wrote {rep}_sum")
        if plots:
            night += [pool.submit(_night_plot_job, t)
                      for t in _sum_plot_tasks(dest, rep, frames_out, orders, cuts, cal, snr)]

        # --- databases, logs ----------------------------------------------------
        db_dir = dest / "calibration_data" / "database"
        last = ordered[-1]
        m_last = orders[-1]
        last_img = f"{_frame_db_stem(last.objname, last.index, cfg.flag_apscatter)}_m{m_last}trans"
        if cfg.flag_skyemission:
            last_img = short_name(f"{last.objname}_skyNO{last.index}_fm_m{m_last}trans",
                                  last.objname)
        last_src = db_dir / f"ap{last_img}"
        from decanter.warptree_products import iraf_apdb_text
        if last_src.is_file():
            (db_dir / "aplast").write_text(iraf_apdb_text(last_src.read_text(), "last"))
        rl = dest / "reduction_log"
        rl.mkdir(exist_ok=True)
        window_note = ("the listfile's ap= tokens" if cfg.flag_manual_aperture
                       else "decanter's center search (WARP's floats)")
        if lock_root is not None:
            # a locked run has the databases' %.2f tokens; WARP's own log of
            # the unrounded window, where it matches them
            ap_low, ap_high, window_note = warp_logged_window(lock_root, ap_low, ap_high, orders)
            say(f"[decanter] aperture_log: {window_note}")
        _write_logs(rl, dest, ordered, orders, cfg, ctx, cal, ap_low, ap_high, wlog, lines,
                    lock_root is not None)
        if wavecal is not None:
            wc_night = dataclasses.replace(wc_night, name=dest.name)
            _write_wavecal_log(rl, wavecal, wc_night, applied, ordered, plot=wavecal_check)
        for fut in night:
            fut.result()
        if bg_plot_dir is not None:
            shutil.rmtree(bg_plot_dir, ignore_errors=True)
    finally:
        pool.close()
    from decanter.warptree_products import status_text
    (rl / "status.txt").write_text(status_text(
        version=WARP_VERSION, start=start_ctime, end=time.ctime(),
        elapsed=time.time() - t_start, lines=lines, images=images,
        image_headers=[_afits.getheader(_fits_name(raw_dir, n)) for n in images],
        comp_header=_afits.getheader(cal.comp), flat_header=_afits.getheader(cal.flat),
        cfg=cfg))
    prov = dict(decanter=_version(), mode=mode,
                implementation="vectorised",
                fast_backend=_backend_label(fast_backend),
                centersearch_minpack=(warp_minpack_label() if lock_root is None
                                      and not cfg.flag_manual_aperture else None),
                # {"NO<i>": {"<order>": n}}: center-search fits that hit the C
                # MINPACK qrfac read past the Jacobian (minpack_exact
                # STATS["undefined"]); WARP's own float logs (aperture_log,
                # centersearch_log) are not reproducible for these frames.
                centersearch_minpack_undefined={
                    f"NO{p.index}": {str(m): int(n) for m, n in sorted(p.centersearch_undefined.items())}
                    for p in ordered if p.centersearch_undefined},
                warp_version_cloned=WARP_VERSION,
                apertures="solved" if lock_root is None else f"locked:{lock_root}",
                aperture_log=window_note,
                raw_dir=str(raw_dir), calibration=str(cal_src or cal.comp.parent),
                calibration_strays=root_strays,
                frames=len(lines), refid=wlog.refid + 1,
                config={f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
                products=dict(intermediates=intermediates, save=save, rawdata=rawdata),
                elapsed_s=round(time.time() - t_start, 1),
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (rl / "decanter_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    say(f"[decanter] wrote {dest} in {time.time() - t_start:.0f} s")
    return WarpTree(root=dest, frames=frames_out, waveshift=wlog, orders=orders, cuts=cuts,
                    mode=mode, aperture_source=prov["apertures"],
                    elapsed=time.time() - t_start, reductions=reds)


def _write_wavecal_log(rl: Path, sol, night, shifts: list, parts: list, *, plot: bool) -> None:
    """The default mode's ``reduction_log/decanter_wavecal.json`` (per frame: the
    telluric (star) and OH (sky) zero points with their errors, km/s, and the s11
    shift applied per order) and, with ``plot``, ``decanter_wavecal_telluric.png``."""
    import json

    def arr(a):
        return None if a is None else [None if not np.isfinite(v) else float(v)
                                       for v in np.asarray(a, float)]
    rec = dict(method=sol.method, mode=sol.mode,
               frames=[f"{p.objname}_NO{p.index}" for p in parts],
               z_star_kms=arr(sol.z_star), z_star_err_kms=arr(sol.z_star_err),
               z_sky_kms=arr(sol.z_sky), z_sky_err_kms=arr(sol.z_sky_err),
               s11_shift_per_order=[{str(m): v for m, v in sh.items()} for sh in shifts])
    if plot:
        from decanter.wavecal.plots import telluric_check
        try:
            rec["telluric_check"] = telluric_check(night, sol, rl / "decanter_wavecal_telluric.png")
        except RuntimeError as e:          # no usable H2O line (e.g. a dry night in WIDE)
            rec["telluric_check"] = dict(error=str(e))
    (rl / "decanter_wavecal.json").write_text(json.dumps(rec, indent=1))


def _tree_wavecal(parts: list, cal: Calibration, cfg: Config):
    """The default mode's wavelength calibration of a tree's frames: the
    solution and, per frame, ``{order: s11 shift}`` (:func:`decanter.api._wavecal_shifts`
    on the frames' unshifted s11-s13 tail)."""
    from decanter._reduction import Intermediates
    from decanter.api import _finalize_wavelength, _wavecal_shifts

    reds = []
    for p in parts:
        it = Intermediates()
        it.spectra_1d, it.sky_1d, it.strip_wcs = p.spectra_1d, p.sky_1d, p.strip_wcs
        r = _finalize_wavelength(p.spectra_1d, p.sky_1d, p.strip_wcs, p.objname, None, None,
                                 cal, cfg, it, shift_wave=0.0, save_intermediates=False)
        r.obj_header, r.sky_header = p.obj_header, p.sky_header
        reds.append(r)
    return _wavecal_shifts(reds, cal)       # (solution, per-frame shifts, night)


def _spectrum(data, ua):
    from decanter.wavelength.iraf_onedspec import Spectrum1D
    return Spectrum1D(np.asarray(data, dtype=np.float32), ua)


_SUFFIX = {"VAC": "VAC", "VAC_norm": "VAC_norm", "VAC_cont": "VAC_cont", "AIR": "AIR",
           "AIR_norm": "AIR_norm", "AIR_cont": "AIR_cont"}
_ONEDSPEC_DIR = {"VAC": "VAC_flux", "VAC_norm": "VAC_norm", "VAC_cont": "VAC_cont",
                 "AIR": "AIR_flux", "AIR_norm": "AIR_norm", "AIR_cont": "AIR_cont"}


def _write_frame(job: dict):
    """s11-s13 for one frame (with its adopted shift) and all of its 1-D files.

    Returns ``(aperture lows, aperture highs, light Reduction, sum inputs)``;
    the sum inputs are ``{(cut, order): {key: Spectrum1D}}`` for the
    ``job["sum_keys"]`` products (the spectra the files hold, for
    ``<OBJ>_sum``).
    """
    import warnings

    from decanter._reduction import Intermediates, Reduction
    from decanter.api import _finalize_wavelength, _shift_for, background_mode
    from decanter.io.fsr import load as load_fsr
    from decanter.warptree_chain import (ua_apall, ua_cut, ua_dispcor, ua_fsr, ua_refspec,
                                         ua_scopy_copy, ua_shift, ua_trans, ua_truncate)
    from decanter.warptree_io import write_iraf_fits
    from decanter.warptree_products import (iraf_apdb_text, spatial_profile_dat,
                                            spectrum_products, spectrum_products_many)
    from decanter.wavelength.fsr import fsr_bounds
    from decanter.wavelength.iraf_onedspec import Spectrum1D, sa_1d
    from decanter.wavelength.iraf_wcs import scopy_wcs
    warnings.filterwarnings("ignore")
    if job.get("fast_backend"):
        from decanter.fast._backend import set_backend
        set_backend(job["fast_backend"])
    p: _FramePart = job["part"]
    cal, cfg, ctx = job["cal"], job["cfg"], job["ctx"]
    dest = Path(job["dest"])
    orders, mode, ncombine = job["orders"], job["mode"], job["ncombine"]
    i, name = p.index, p.objname
    fdir = dest / f"{name}_NO{i}"
    prov = f"decanter {_version()} mode={mode}"
    inter = Intermediates()
    from decanter.utils.iraf_imio import imio_read_real
    bgm = background_mode(cfg)
    # sarith none - subtracted (both read through the FITS kernel)
    bg_1d = {m: (imio_read_real(p.spectra_1d_none[m])
                 - imio_read_real(p.spectra_1d[m])).astype(np.float32)
             for m in p.spectra_1d_none}
    r = _finalize_wavelength(p.spectra_1d, p.sky_1d, p.strip_wcs, name, None, None,
                             cal, cfg, inter, shift_wave=job["shift"], save_intermediates=True,
                             bg_1d=bg_1d,
                             twod={"strip": p.strips_2d, "resample": p.strips_2d_resample})
    refs = {m: _read_ref_aperture(cal.ref_trans_apdbs[m] if job["locked"]
                                  else Path(p.aperture_refs[m]), m) for m in orders}
    db_dir = dest / "calibration_data" / "database"
    # WARP clamps the frame aperture to the reference window before writing
    # it (Warp_sci.py: lowlim = max(apLow, lowtrans), highlim = min(apHigh,
    # hightrans)); a no-op whenever reduce() already did.
    aps = {m: (max(refs[m].ap_low, p.apertures[m][0]), min(refs[m].ap_high, p.apertures[m][1]))
           for m in orders}
    lows, highs = [], []
    frame_db_text = {}
    fm = obj_suffixes(cfg.flag_apscatter)[1]
    for m in orders:
        lo, hi = aps[m]
        lows.append(lo)
        highs.append(hi)
        img = f"{_frame_db_stem(name, i, cfg.flag_apscatter)}_m{m}trans"
        # WARP writes the listfile's bg= region into the DB even without
        # background subtraction (Warp_sci.py:433, aperture.py:175).
        txt = render_frame_apdb(refs[m], img, lo, hi, bgsample=job.get("bg", "INDEF"),
                                timestamp=job["stamp"])
        (db_dir / f"ap{img}").write_text(txt)
        frame_db_text[m] = txt
    # apscatter's database of the _s frame (references = the apscatter mask)
    if cfg.flag_apscatter:
        s_img = short_name(f"{name}_NO{i}_s", name)
        (db_dir / f"ap{s_img}").write_text(
            iraf_apdb_text(Path(cal.apdb_apsc).read_text(), s_img))

    # --- header chain ---------------------------------------------------------
    job2 = dict(objname=name, index=i, ctx=ctx, config=cfg)
    ua = _frame_uas(job2, p.obj_header, p.sky_header)
    fsr = load_fsr(cal.fsr_table)
    obj_vac: dict = {}
    sky_vac: dict = {}
    sky_sc: dict = {}
    base_o = fdir / "intermediate_files" / "OBJ" / "8A-OBJ-1DSPEC"
    base_s = fdir / "intermediate_files" / "SKY"
    write_inter = job["intermediates"] or job["save"]
    for m in orders:
        crval2, cdelt2, nrows, xmin, _xmax = p.strip_meta[m]
        apn = _apnum(refs[m], *aps[m])
        comp_id = f"{ctx.comp_stem}.{m:04d}"
        dc = inter.dispcor_wcs.get(m)
        # obj (apall background=<mode>: BANDID1 names it, Warp_sci.py:561-562)
        tr_ua = ua_trans(ua_cut(ua["sscfm"], xmin), crval2, cdelt2)
        ext_ua, _, _ = ua_apall(tr_ua, nrows, apn, background=bgm)
        c_ua = ua_truncate(ext_ua, nrows)
        n_c = int(np.asarray(inter.spectra_trunc[m]).size)
        cs_ua = (ua_shift(c_ua, n_c, _shift_for(job["shift"], m)) if ncombine
                 else ua_scopy_copy(c_ua, n_c))
        cs_ua = ua_refspec(cs_ua, comp_id)
        if write_inter:
            write_iraf_fits(base_o / "1-OBJ-1DSPEC_extract" / f"{name}_NO{i}_m{m}.{m:04d}.fits",
                            p.spectra_1d[m], ext_ua, provenance=prov)
            write_iraf_fits(base_o / "2-OBJ-1DSPEC_truncate" / f"{name}_NO{i}_m{m}c.fits",
                            inter.spectra_trunc[m], c_ua, provenance=prov)
            write_iraf_fits(base_o / "3-OBJ-1DSPEC_shift" / f"{name}_NO{i}_m{m}cs.fits",
                            inter.spectra_shifted[m], cs_ua, provenance=prov)
        if cfg.flag_skysub and m in p.spectra_1d_none:
            # Background Subtraction (Warp_sci.py:563-569, 646-648; on for
            # every token but "none", so also for "no", whose apall
            # background is none): the background-none extraction, sarith
            # none - subtracted, truncate, dispcor_single (REFSPEC1 on its
            # input) + AIRORVAC
            none_ua, _, _ = ua_apall(tr_ua, nrows, apn)
            bg = sa_1d(Spectrum1D(p.spectra_1d_none[m], none_ua), "-",
                       Spectrum1D(p.spectra_1d[m], ext_ua))
            bgc_ua = ua_refspec(ua_truncate(bg.header, nrows), comp_id)
            bdc = inter.bg_dispcor_wcs.get(m)
            if write_inter or job.get("bg_plot_dir"):
                bgcw_dir = (base_o / "4-OBJ-1DSPEC_dispcor" if write_inter
                            else Path(job["bg_plot_dir"]))
                if bdc is not None:
                    write_iraf_fits(bgcw_dir / f"{name}_NO{i}_m{m}_bgcw.fits",
                                    inter.bg_dispcor[m],
                                    ua_dispcor(bgc_ua, bdc.crval1, bdc.cdelt1), provenance=prov)
            if write_inter:
                ex_dir = base_o / "1-OBJ-1DSPEC_extract"
                write_iraf_fits(ex_dir / f"{name}_NO{i}_m{m}_none.{m:04d}.fits",
                                p.spectra_1d_none[m], none_ua, provenance=prov)
                write_iraf_fits(ex_dir / f"{name}_NO{i}_m{m}_bg.fits", bg.data, bg.header,
                                provenance=prov)
                write_iraf_fits(base_o / "2-OBJ-1DSPEC_truncate" / f"{name}_NO{i}_m{m}_bgc.fits",
                                inter.bg_trunc[m], bgc_ua, provenance=prov)
        if cfg.flag_extract2d and m in p.strips_2d:
            _write_frame_twod(job, fdir, name, i, fm, m, tr_ua, nrows, apn, comp_id, inter, r,
                              p, ncombine, prov, write_inter)
        if dc is None:
            continue
        csw_ua = ua_dispcor(cs_ua, dc.crval1, dc.cdelt1)
        if write_inter:
            write_iraf_fits(base_o / "4-OBJ-1DSPEC_dispcor" / f"{name}_NO{i}_m{m}csw.fits",
                            inter.spectra_dispcor[m], csw_ua, provenance=prov)
        for cut in job["cfg"].cutrange_list:
            if (cut, m) not in r.obj:
                continue
            lo_w, hi_w = fsr_bounds(fsr[m].lambda_min, fsr[m].lambda_max, float(cut))
            sc = scopy_wcs(dc.crval1, dc.cdelt1, dc.nw, lo_w, hi_w)
            obj_vac[(cut, m)] = _spectrum(r.obj[(cut, m)].flux, ua_fsr(csw_ua, sc))
            sky_sc[(cut, m)] = sc
        # sky
        if m in p.sky_1d and "fm" in ua:
            sext_ua, _, _ = ua_apall(ua_trans(ua_cut(ua["fm"], xmin), crval2, cdelt2), nrows, apn)
            sc_ua = ua_refspec(ua_truncate(sext_ua, nrows), comp_id)
            if job["save"]:
                # WARP keeps the sky extraction under its shortFile work
                # name (Warp_sci.py:574: the name is built from the short
                # str, so remove_or_move_sf has no long name to restore).
                sext = short_name(f"{name}_skyNO{i}_fm_m{m}trans1d", name)
                write_iraf_fits(base_s / "5-SKY_extract" / f"{sext}.{m:04d}.fits",
                                p.sky_1d[m], sext_ua, provenance=prov)
                write_iraf_fits(base_s / "6-SKY_truncate" / f"{name}_skyNO{i}_fm_m{m}trans1dcut.fits",
                                inter.sky_trunc[m], sc_ua, provenance=prov)
            scw_ua = ua_dispcor(sc_ua, dc.crval1, dc.cdelt1)
            if write_inter and m in inter.sky_dispcor:
                write_iraf_fits(base_s / "7-SKY_dispcor" / f"{name}_skyNO{i}_fm_m{m}trans1dcutw.fits",
                                inter.sky_dispcor[m], scw_ua, provenance=prov)
            for cut in job["cfg"].cutrange_list:
                if r.sky and (cut, m) in r.sky and (cut, m) in sky_sc:
                    sky_vac[(cut, m)] = _spectrum(r.sky[(cut, m)].flux,
                                                  ua_fsr(scw_ua, sky_sc[(cut, m)]))
            # the sky aperture database apall writes (reference: the obj DB)
            sky_img = short_name(f"{name}_skyNO{i}_fm_m{m}trans", name)
            (db_dir / f"ap{sky_img}").write_text(iraf_apdb_text(frame_db_text[m], sky_img))

    # --- final spectra --------------------------------------------------------
    sum_keys = job.get("sum_keys", ())
    sum_in: dict = {}
    # the continuum fits of all of the frame's spectra at once in the
    # vectorised mode (bit for bit the same as one at a time)
    keys = list(obj_vac)
    derived = spectrum_products_many([obj_vac[k] for k in keys], [k[1] for k in keys],
                                     batched=True)
    for (cut, m), dprods in zip(keys, derived, strict=True):
        vac = obj_vac[(cut, m)]
        prods = {"VAC": vac}
        prods.update(dprods)
        for key, spec in prods.items():
            out = (fdir / "onedspec" / _ONEDSPEC_DIR[key] / f"fsr{cut:.2f}"
                   / f"{name}_NO{i}_m{m}_fsr{cut:.2f}_{_SUFFIX[key]}.fits")
            write_iraf_fits(out, spec.data, spec.header, provenance=prov)
        if sum_keys:
            # what <OBJ>_sum reads of this frame (the files' data and UA)
            sum_in[(cut, m)] = {k: prods[k] for k in sum_keys}
    for (cut, m), vac in sky_vac.items():
        for key, spec in (("VAC", vac), ("AIR", spectrum_products(vac, m, sky=True)["AIR"])):
            out = (fdir / "sky_emission" / key / f"fsr{cut:.2f}"
                   / f"{name}_skyNO{i}_fm_m{m}trans1dcutw_fsr{cut:.2f}_{key}.fits")
            write_iraf_fits(out, spec.data, spec.header, provenance=prov)

    # --- images/spatial_profile/*.dat ----------------------------------------
    sp_dir = fdir / "images" / "spatial_profile"
    sp_dir.mkdir(parents=True, exist_ok=True)
    for m in orders:
        if m in p.slit_profiles:
            mx, my = p.slit_profiles[m]
            (sp_dir / f"{name}_NO{i}_{fm}_m{m}trans.dat").write_text(spatial_profile_dat(mx, my))
    if job.get("plots"):
        # aperture_plot (Warp_sci.py:411-428): the frame window, unclamped
        from decanter.warptree_plots import aperture_plot
        if job.get("manual_ap") is not None:
            lowtrans, hightrans = job["manual_ap"]
        else:
            xs = [p.centersearch[m][0] for m in orders]
            gs = [p.centersearch[m][1] for m in orders]
            lowtrans = -1. * np.nanmedian(gs) + np.nanmedian(xs)
            hightrans = np.nanmedian(gs) + np.nanmedian(xs)
        for m in orders:
            dat = sp_dir / f"{name}_NO{i}_{fm}_m{m}trans.dat"
            if dat.is_file():
                aperture_plot(dat, sp_dir / f"{name}_NO{i}_{fm}_m{m}trans.png", lowtrans,
                              hightrans, job.get("bg", "INDEF"), cfg.flag_skysub)
    for d in ("2d_image",) + (("badpixmask",) if cfg.flag_bpmask else ()):
        (fdir / "images" / d).mkdir(parents=True, exist_ok=True)
    if cfg.flag_extract2d:
        # WARP makes these whether or not -s leaves anything in them
        for d in ("AIR", "VAC"):
            (fdir / "twodspec" / d).mkdir(parents=True, exist_ok=True)
        if write_inter:
            for d in ("1-OBJ-2DSPEC_extract", "2-OBJ-2DSPEC_truncate", "3-OBJ-2DSPEC_shift"):
                (fdir / "intermediate_files" / "OBJ" / "8B-OBJ-2DSPEC" / d).mkdir(
                    parents=True, exist_ok=True)

    light = Reduction(obj_name=name, obj_path=None, sky_path=None, obj=r.obj, sky=r.sky,
                      obj_header=p.obj_header, sky_header=p.sky_header,
                      apertures=dict(p.apertures),
                      shift_wave=0.0 if isinstance(job["shift"], dict) else job["shift"])
    return lows, highs, light, sum_in


def _write_frame_twod(job: dict, fdir: Path, name: str, i: int, fm: str, m: int, tr_ua,
                      nrows: int, apn: str, comp_id: str, inter, r, p, ncombine: bool,
                      prov: str, write_inter: bool) -> None:
    """One order's ``Extract 2d spectrum`` files (Warp_sci.py:575-584, 671-686,
    923-938, 1016-1038): the ``apall`` strip and ``resample2Dspec`` output
    (``-s``), their truncation (``-s``), shift (``3-OBJ-2DSPEC_shift``) and
    ``twodspec/{VAC,AIR}``; the 2-D header chain is
    :mod:`decanter.warptree_chain` (``ua_strip`` ... ``ua_dispcor_2d``)."""
    from decanter.api import _shift_for
    from decanter.warptree_chain import (
        ua_copy_2d,
        ua_dispcor_2d,
        ua_refspec,
        ua_shift_2d,
        ua_strip,
        ua_truncate_2d,
    )
    from decanter.warptree_io import iraf_fits_bytes, write_astropy_replaced, write_iraf_fits
    from decanter.wavelength.iraf_onedspec import Spectrum1D, vac2air
    stem = f"{name}_NO{i}_{fm}_m{m}trans"
    base = fdir / "intermediate_files" / "OBJ" / "8B-OBJ-2DSPEC"
    save = job["save"]
    dc = inter.dispcor_wcs.get(m)
    s_ua = ua_strip(tr_ua, nrows, apn, m)
    strip = np.asarray(p.strips_2d[m], np.float32)
    if save:
        write_iraf_fits(base / "1-OBJ-2DSPEC_extract" / f"{stem}2d.{m:04d}.fits", strip, s_ua,
                        provenance=prov)
        # resample2Dspec rewrites the strip file with astropy
        write_astropy_replaced(base / "1-OBJ-2DSPEC_extract" / f"{stem}2d_resample.{m:04d}.fits",
                               iraf_fits_bytes(strip, s_ua, provenance=prov),
                               p.strips_2d_resample[m])
    for variant, suf, out2d in (("strip", "", r.obj_2d), ("resample", "_resample",
                                                          r.obj_2d_resample)):
        key = (variant, m)
        if key not in inter.twod_trunc:
            continue
        cut = inter.twod_trunc[key]
        nl, n_c = cut.shape
        cut_ua = ua_truncate_2d(s_ua, nrows, nl)
        cuts_ua = (ua_shift_2d(cut_ua, n_c, _shift_for(job["shift"], m), nl) if ncombine
                   else ua_copy_2d(cut_ua, n_c, nl))
        cuts_ua = ua_refspec(cuts_ua, comp_id)
        if save:
            write_iraf_fits(base / "2-OBJ-2DSPEC_truncate" / f"{stem}2dcut{suf}.fits", cut, cut_ua,
                            provenance=prov)
        if write_inter:
            write_iraf_fits(base / "3-OBJ-2DSPEC_shift" / f"{stem}2dcuts{suf}.fits",
                            inter.twod_shifted[key], cuts_ua, provenance=prov)
        if dc is None or m not in out2d:
            continue
        vac_ua = ua_dispcor_2d(cuts_ua, dc.crval1, dc.cdelt1, nl)
        vac = np.asarray(out2d[m].flux, np.float32)
        air = vac2air(Spectrum1D(vac, vac_ua), multispec=nl > 1)
        write_iraf_fits(fdir / "twodspec" / "VAC" / f"{stem}2dcutsw{suf}_VAC.fits", vac, vac_ua,
                        provenance=prov)
        write_iraf_fits(fdir / "twodspec" / "AIR" / f"{stem}2dcutsw{suf}_AIR.fits", air.data,
                        air.header, provenance=prov)


def _version() -> str:
    import decanter
    return decanter.__version__


# ---------------------------------------------------------------------------
# <OBJ>_sum, SNR_dat
# ---------------------------------------------------------------------------

FRAME_NUMBER_LIMIT = 28      # warp/config.py: frameNumberLimit


class _Pool:
    """One process pool shared by every phase of a tree (``workers > 1``), or
    inline execution in this process."""

    def __init__(self, workers: int):
        self._ex = None
        if workers > 1:
            import multiprocessing as mp
            self._ex = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=mp.get_context("spawn"))

    def map(self, fn, items):
        if self._ex is None:
            return (fn(x) for x in items)
        return self._ex.map(fn, items)

    def submit(self, fn, *args):
        if self._ex is not None:
            return self._ex.submit(fn, *args)
        from concurrent.futures import Future
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:      # re-raised by .result(), like the pool
            fut.set_exception(exc)
        return fut

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)


def _snr_place(dest: Path, c: str, n: int) -> Path | None:
    """Where WARP leaves ``SNratio_*_fsr<c>.{dat,png}`` (Warp_sci.py:736-741,
    800-802, 884-890): moved to ``SNR_dat/fsr<c>/`` for 2..27 frames, left
    in the tree root for exactly ``frameNumberLimit`` (28) frames, not
    written otherwise."""
    if 1 < n < FRAME_NUMBER_LIMIT:
        return dest / "SNR_dat" / f"fsr{c}"
    if n == FRAME_NUMBER_LIMIT:
        return dest
    return None


def _sum_job(job: dict):
    """One (cut, order) of ``<OBJ>_sum`` (and its SNR estimate).

    ``job["specs"]``: every frame's ``VAC`` spectrum (2..28 frames:
    ``PyScombine`` + continuum/air, ``snestimate``) or frame 1's six
    products (``scopy``), Warp_sci.py:735-766.  Returns ``(cut, order, SNR
    wavelengths, SNR values, {VAC_norm, AIR_norm} of the sum)``.
    """
    import warnings

    from decanter.warptree_io import write_iraf_fits
    from decanter.warptree_products import snr_estimate, spectrum_products_many, sum_copy
    from decanter.wavelength.iraf_scombine import scombine_average
    warnings.filterwarnings("ignore")
    dest, rep, cut, m = Path(job["dest"]), job["rep"], job["cut"], job["m"]
    frames = job["frames"]                  # [(dirname, objname)] in frame order
    batched = True
    prov = f"decanter {_version()}"
    c = f"{cut:.2f}"
    n = len(frames)
    combine = 1 < n <= FRAME_NUMBER_LIMIT
    if combine:
        vacs = job["specs"]
        vac = scombine_average(vacs)
        prods = {"VAC": vac}
        prods.update(spectrum_products_many([vac], [m], batched=batched)[0])
    else:
        prods = sum_copy(job["specs"])
    for key, spec in prods.items():
        out = (dest / f"{rep}_sum" / _ONEDSPEC_DIR[key] / f"fsr{c}"
               / f"{rep}_sum_m{m}_fsr{c}_{_SUFFIX[key]}.fits")
        write_iraf_fits(out, spec.data, spec.header, provenance=prov)
    lam = sn = None
    if combine:
        # snestimate's input list: WARP's shortFile work names
        names = [short_name(f"{d}_m{m}_fsr{c}_VAC", o) for d, o in frames]
        txt, lam, sn = snr_estimate(vacs, names, batched=batched)
        place = _snr_place(dest, c, n)
        place.mkdir(parents=True, exist_ok=True)
        (place / f"SNratio_m{m}_fsr{c}.dat").write_text(txt)
    norms = {k: prods[k] for k in ("VAC_norm", "AIR_norm")} if combine else {}
    return cut, m, lam, sn, norms


def _norm_comb_job(job: dict) -> None:
    """``PyScombine`` of every order's sum ``VAC_norm`` / ``AIR_norm``
    (Warp_sci.py:756-757): ``<OBJ>_sum_fsr<cut>_<key>_comb``, moved into
    ``<OBJ>_sum/<key>/fsr<cut>/`` below 28 frames (Warp_sci.py:877-880) and
    left in the tree root under its shortFile work name at 28."""
    import warnings

    from decanter.warptree_io import write_iraf_fits
    from decanter.wavelength.iraf_scombine import scombine_average
    warnings.filterwarnings("ignore")
    dest, rep, c, key = Path(job["dest"]), job["rep"], job["c"], job["key"]
    comb = scombine_average(job["specs"])
    stem = f"{rep}_sum_fsr{c}_{key}_comb"
    if job["n"] < FRAME_NUMBER_LIMIT:
        out = dest / f"{rep}_sum" / key / f"fsr{c}" / f"{stem}.fits"
    else:
        out = dest / f"{short_name(stem, rep)}.fits"
    write_iraf_fits(out, comb.data, comb.header, provenance=f"decanter {_version()}")


def _write_sums(pool: _Pool, dest: Path, rep: str, frames: list, orders, cuts,
                sum_inputs: list[dict], *, mode: str = "default") -> dict:
    """``<OBJ>_sum`` (Warp_sci.py:735-766, 870-880), ``SNR_dat`` and the
    ``*_norm_comb`` spectra, from the frames' spectra (``sum_inputs``: per
    frame, what :func:`_write_frame` handed over).  Returns the SNR values
    for the figures, ``{(cut, order): (wavelengths, S/N)}``."""
    names = [(f.dirname, f.objname) for f in frames]
    n = len(frames)
    combine = 1 < n <= FRAME_NUMBER_LIMIT
    for key in _ONEDSPEC_DIR.values():
        for cut in cuts:
            (dest / f"{rep}_sum" / key / f"fsr{cut:.2f}").mkdir(parents=True, exist_ok=True)
    jobs = []
    for cut in cuts:
        for m in orders:
            if combine:
                specs = [si[(cut, m)]["VAC"] for si in sum_inputs]
            else:
                specs = dict(sum_inputs[0][(cut, m)])
            jobs.append(dict(dest=str(dest), rep=rep, cut=cut, m=m, frames=names, specs=specs,
                             mode=mode))
    res = list(pool.map(_sum_job, jobs))
    snr = {(cut, m): (lam, sn) for cut, m, lam, sn, _ in res}
    if combine:
        norms = {(cut, m): nm for cut, m, _, _, nm in res}
        futs = [pool.submit(_norm_comb_job, dict(
                    dest=str(dest), rep=rep, c=f"{cut:.2f}", key=key, n=n,
                    specs=[norms[(cut, m)][key] for m in orders]))
                for cut in cuts for key in ("VAC_norm", "AIR_norm")]
        for f in futs:
            f.result()
    return snr


# ---------------------------------------------------------------------------
# Raw data and the night's figures
# ---------------------------------------------------------------------------

def _rawdata_job(out: str, raw: str, plots: bool) -> None:
    """One raw frame into ``rawdata_image/`` as WARP copies it (``hedit
    PIPELINE`` in place: the raw ``DATE``, ``BZERO`` and ``BSCALE`` stay;
    Warp_sci.py:165-167, 1084-1087), and its ``plot_2dimages`` figure
    (Warp_sci.py:794-795; drawn from the copied pixels)."""
    import warnings

    from decanter.warptree_io import raw_ua, write_iraf_fits
    warnings.filterwarnings("ignore")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(raw).stem
    with _afits.open(raw) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
        ua = raw_ua(h[0].header, pipeline=WARP_VERSION, keep_scale=True)
    write_iraf_fits(out_dir / f"{name}.fits", data, ua, keep_date=True)
    if plots:
        from decanter.warptree_plots import plot_2dimages
        plot_2dimages(data, out_dir / f"{name}.png", f"{name}.fits")


def _fsr_ranges(cal: Calibration) -> dict:
    from decanter.io.fsr import load as load_fsr
    tab = load_fsr(cal.fsr_table)
    return {m: [tab[m].lambda_min, tab[m].lambda_max] for m in tab}


def _frame_spec(dest: Path, fr: str, key: str, m: int, c: str) -> str:
    return str(dest / fr / "onedspec" / _ONEDSPEC_DIR[key] / f"fsr{c}"
               / f"{fr}_m{m}_fsr{c}_{_SUFFIX[key]}.fits")


def _night_plot_tasks(dest: Path, rep: str, frames: list, orders, cuts, cfg: Config,
                      cal: Calibration, ordered: list,
                      bg_plot_dir: Path | None = None) -> list[dict]:
    """The night's figures drawn from the frames' files alone
    (Warp_sci.py:773-781, 804-813): ``spectra_image/<OBJ>_VAC{norm,flux}.pdf``
    (with ``Background Subtraction``, ``plot_all_frames_flux_BG`` of the
    ``_bgcw`` spectra) and ``reduction_log/count_and_fwhm.png``."""
    fsr = _fsr_ranges(cal)
    c0 = f"{cuts[0]:.2f}"
    names = [f.dirname for f in frames]
    apnum = list(orders)
    si = dest / "spectra_image"
    rl = dest / "reduction_log"
    si.mkdir(exist_ok=True)
    rl.mkdir(exist_ok=True)
    norm105 = [[_frame_spec(dest, fr, "VAC_norm", m, c0) for m in orders] for fr in names]
    flux105 = [[_frame_spec(dest, fr, "VAC", m, c0) for m in orders] for fr in names]
    mode = str(_afits.getheader(cal.comp).get("INSTMODE", "WIDE")).strip()
    fwhm = None
    if not cfg.flag_manual_aperture and all(p.centersearch for p in ordered):
        width = np.array([[p.centersearch[m][1] for m in orders] for p in ordered])
        fwhm = np.median(width, axis=1)
    n = len(frames)
    flux_task = dict(kind="flux", splist=flux105, out=str(si / f"{rep}_VACflux.pdf"),
                     apnum=apnum, n=n, fsr=fsr)
    if cfg.flag_skysub:
        def bgcw(fr: str, m: int) -> str:
            d = (Path(bg_plot_dir) if bg_plot_dir is not None else
                 dest / fr / "intermediate_files" / "OBJ" / "8A-OBJ-1DSPEC" / "4-OBJ-1DSPEC_dispcor")
            return str(d / f"{fr}_m{m}_bgcw.fits")
        flux_task.update(kind="flux_bg", bgspec=[[bgcw(fr, m) for m in orders] for fr in names])
    return [dict(kind="norm", splist=norm105, out=str(si / f"{rep}_VACnorm.pdf"), apnum=apnum,
                 n=n, fsr=fsr),
            flux_task,
            dict(kind="count_fwhm", splist=flux105, out=str(rl / "count_and_fwhm.png"),
                 apnum=apnum, n=n, fwhm=fwhm, mode=mode)]


def _sum_plot_tasks(dest: Path, rep: str, frames: list, orders, cuts, cal: Calibration,
                    snr: dict) -> list[dict]:
    """The figures drawn from ``<OBJ>_sum`` (Warp_sci.py:783, 800-802):
    ``spectra_image/<OBJ>_AIRnorm_combined_m*.png`` and the
    ``SNratio_fsr<cut>.png`` (2..28 frames, :func:`_snr_place`)."""
    fsr = _fsr_ranges(cal)
    c0 = f"{cuts[0]:.2f}"
    si = dest / "spectra_image"
    apnum = list(orders)
    comb = [str(dest / f"{rep}_sum" / "AIR_norm" / f"fsr{c0}"
                / f"{rep}_sum_m{m}_fsr{c0}_AIR_norm.fits") for m in orders]
    tasks = [dict(kind="combined", splist=comb,
                  outs=[str(si / f"{rep}_AIRnorm_combined_m{m}.png") for m in orders],
                  apnum=apnum, fsr=fsr)]
    n = len(frames)
    for cut in cuts:
        c = f"{cut:.2f}"
        place = _snr_place(dest, c, n)
        if place is None:
            continue
        place.mkdir(parents=True, exist_ok=True)
        tasks.append(dict(kind="snr", lams=[snr[(cut, m)][0] for m in orders],
                          vals=[snr[(cut, m)][1] for m in orders],
                          sumv=[str(dest / f"{rep}_sum" / "VAC_flux" / f"fsr{c}"
                                    / f"{rep}_sum_m{m}_fsr{c}_VAC.fits") for m in orders],
                          out=str(place / f"SNratio_fsr{c}.png")))
    return tasks


def _night_plot_job(t: dict) -> None:
    """Draw one of the night's figures (:mod:`decanter.warptree_plots`)."""
    import warnings

    from decanter.warptree_plots import (peak_count_fwhm, plot_all_frames_flux,
                                         plot_all_frames_flux_BG, plot_all_frames_norm,
                                         plot_combined_norm, snr_plots)
    warnings.filterwarnings("ignore")
    kind = t["kind"]
    if kind == "norm":
        plot_all_frames_norm(t["splist"], t["out"], t["apnum"], t["n"], t["fsr"])
    elif kind == "flux":
        plot_all_frames_flux(t["splist"], t["out"], t["apnum"], t["n"], t["fsr"])
    elif kind == "flux_bg":
        plot_all_frames_flux_BG(t["splist"], t["bgspec"], t["out"], t["apnum"], t["n"], t["fsr"])
    elif kind == "count_fwhm":
        if t["fwhm"] is not None:
            peak_count_fwhm(t["splist"], t["out"], t["apnum"], t["n"], fwhm=t["fwhm"],
                            mode=t["mode"])
        else:
            peak_count_fwhm(t["splist"], t["out"], t["apnum"], t["n"], mode=t["mode"])
    elif kind == "combined":
        plot_combined_norm(t["splist"], t["outs"], t["apnum"], t["fsr"])
    elif kind == "snr":
        snr_plots(t["lams"], t["vals"], t["sumv"], len(t["sumv"]), t["out"])
    else:                                   # pragma: no cover
        raise ValueError(kind)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def _write_logs(rl: Path, dest: Path, ordered: list, orders, cfg: Config, ctx: _HeaderContext,
                cal: Calibration, ap_low, ap_high, wlog, lines, locked: bool) -> None:
    """``reduction_log/`` (Warp_sci.py:1065-1080): the aperture, waveshift,
    center-search, cosmic-ray, apscatter and cutransform logs."""
    from decanter.warptree_products import (apscatter_log_entry, cutransform_log_entry,
                                            write_centersearch_log, write_cosmicray_log)
    write_aperture_log(rl / "aperture_log.txt", rl / "aperture_log.npz", ap_low, ap_high, orders)
    if len(lines) > 1:
        wlog.save_npz(rl / "waveshift_log.npz")
        wlog.save_text(rl / "waveshift_log.txt")
    else:
        # one frame: WARP writes the log but only moves it when objnum > 1
        wlog.save_npz(dest / "waveshift_log.npz")
        wlog.save_text(dest / "waveshift_log.txt")
    if not cfg.flag_manual_aperture and all(p.centersearch for p in ordered):
        center = np.array([[p.centersearch[m][0] for m in orders] for p in ordered])
        width = np.array([[p.centersearch[m][1] for m in orders] for p in ordered])
        write_centersearch_log(rl / "centersearch_log.txt", rl / "centersearch_log.npz",
                               center, width, orders)
    if cfg.flag_bpmask:
        sig = [p.cr_stats[1] if p.cr_stats else 0. for p in ordered]
        num = [p.cr_stats[0] if p.cr_stats else 0 for p in ordered]
        write_cosmicray_log(rl / "cosmicray_log.txt", rl / "cosmicray_log.npz", sig, num, orders)
    ref = Path(cal.apdb_multihole).name[2:]
    apsc = Path(cal.apdb_apsc).name[2:]
    aps_txt = ""
    cut_txt = ""
    for p in ordered:
        s_name = short_name(f"{p.objname}_NO{p.index}_s", p.objname)
        if cfg.flag_apscatter:
            aps_txt += apscatter_log_entry(s_name, short_name(f"{p.objname}_NO{p.index}_ssc",
                                                              p.objname), apsc)
        fm = obj_suffixes(cfg.flag_apscatter)[1]
        cut_txt += cutransform_log_entry(short_name(f"{p.objname}_NO{p.index}_{fm}", p.objname),
                                         ref, ctx.dy_text, cfg.fluxinput)
        if cfg.flag_skyemission:
            cut_txt += cutransform_log_entry(short_name(f"{p.objname}_skyNO{p.index}_fm", p.objname),
                                             ref, ctx.dy_text, cfg.fluxinput)
    if cfg.flag_apscatter:
        (rl / "apscatter_log.txt").write_text(aps_txt)
    (rl / "cutransform_log.txt").write_text(cut_txt)
