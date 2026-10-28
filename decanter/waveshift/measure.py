"""Stage 10 — measure inter-frame wavelength offsets.

WARP equivalent: ``warp/ccwaveshift.py:waveshift_oneorder`` (line 140)
and ``waveshiftClip`` (line 183).

For each (frame, order), shift the extracted 1D spectrum against the
reference frame's spectrum via cubic-spline interpolation
(``scipy.ndimage.shift``), evaluate squared-residual over the central
region, and pick the shift minimizing it. The search is three-stage
progressive: coarse (width 2.0, step 0.5), medium (0.4, 0.1), fine
(0.07, 0.01) — matching WARP exactly.

Spike C resolved: WARP uses argmin of the SSE grid at fine step 0.01
without an explicit parabola fit. The shift is reported in wavelength
units (multiplied by CDELT1).

Per-frame median shift is computed via WARP's iterative sigma-clip
(``waveshiftClip``): one 1σ pass, then four 2σ passes; rejects orders
with shift far from the per-frame mean.

Output: ``waveshift_log.npz`` with one shift per (frame, order) plus
the per-frame averages and a quality flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as nd_shift

from decanter.calib.aperture import ApertureSet
from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile
from decanter.utils.iraf_scopy import scopy_wavelength_truncate

# WARP truncates the 1D extract to wavelength range [1, 2048] before the
# cross-correlation (``Warp_sci.py:468 file_matrix_forwaveshift`` points
# at ``_m###c.fits`` = post-truncate). The same hard-coded bounds as
# s11's truncate step.
_TRUNC_W1 = 1.0
_TRUNC_W2 = 2048.0


@dataclass(frozen=True, slots=True)
class WaveshiftTable:
    """Per-(frame, order) wavelength shifts + per-frame averages."""

    frame_ids: list[str]
    orders: tuple[int, ...]
    shift_matrix: NDArray[np.float64]  # (n_orders, n_frames) in wavelength units
    shift_average: NDArray[np.float64]  # (n_frames,) — sigma-clipped mean
    shift_stddev: NDArray[np.float64]   # (n_frames,)
    shift_calcnum: NDArray[np.int64]    # (n_frames,) — orders kept after clip
    flag: NDArray[np.int64]             # (n_frames,) — 0=good, ≥1=flagged

    def save_npz(self, path: Path) -> None:
        np.savez(
            path,
            frame_ids=np.array(self.frame_ids, dtype=object),
            orders=np.asarray(self.orders),
            shift_matrix=self.shift_matrix,
            shift_average=self.shift_average,
            shift_stddev=self.shift_stddev,
            shift_calcnum=self.shift_calcnum,
            flag=self.flag,
        )


def _sanitize_objname(raw: str) -> str:
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def cc_shift_search(
    ref_flux: NDArray[np.floating],
    sp_flux: NDArray[np.floating],
    *,
    cshift: float,
    width: float,
    step: float,
    div: int = 100,
    ec_short: int = 800,
    ec_long: int = 300,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One step of the WARP progressive cross-correlation search.

    Args:
        ref_flux: reference 1D spectrum (same dtype/length as ``sp_flux``).
        sp_flux: spectrum to be shifted to align with ``ref_flux``.
        cshift: center of the shift search (in pixels).
        width, step: shift grid is ``[cshift - width, cshift + width]``
            stepped by ``step``.
        div, ec_short, ec_long: as in ``warp/ccwaveshift.py:91-93``.

    Returns:
        ``(shifts, dify)`` — the shift grid and the mean squared
        residual at each shift.
    """
    ref = np.asarray(ref_flux, dtype=np.float64)
    sp = np.asarray(sp_flux, dtype=np.float64)
    naxis1 = ref.size
    ref_med = float(np.median(ref))
    sp_med = float(np.median(sp))
    if ref_med == 0:
        ref_med = 1.0
    if sp_med == 0:
        sp_med = 1.0
    ref_norm = ref / ref_med
    start_sp1 = int(naxis1 / div)
    end_sp1 = int(naxis1 / div * (div - 1))
    lo = max(0, start_sp1 + ec_short)
    hi = min(naxis1, end_sp1 - ec_long)
    if hi <= lo:
        # Spectrum too short for the WARP-style trim; widen the window.
        lo = naxis1 // 4
        hi = naxis1 * 3 // 4

    shiftnum = int(width / step * 2 + 1)
    shifts = np.array([cshift - width + i * step for i in range(shiftnum)], dtype=np.float64)
    dify = np.empty(shiftnum, dtype=np.float64)
    for i, s in enumerate(shifts):
        sp_shifted = nd_shift(sp, float(s), order=3)
        sp_norm = sp_shifted / sp_med
        diff = ref_norm[lo:hi] - sp_norm[lo:hi]
        dify[i] = float(np.mean(diff**2))
    return shifts, dify


def waveshift_one_order(
    spectra: list[NDArray[np.floating]],
    cdelt1: float,
    refid: int = 0,
) -> NDArray[np.float64]:
    """Compute wavelength shift vector for one echelle order across frames.

    Args:
        spectra: list of 1D spectra, one per frame. All must share the
            same length and CDELT1.
        cdelt1: wavelength step per pixel (output shift is in same units).
        refid: index of the reference frame; its shift is 0 by definition.

    Returns:
        Array of length ``len(spectra)`` with the wavelength shift
        (in CDELT1 units) of each frame relative to ``spectra[refid]``.
    """
    n = len(spectra)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == refid:
            continue
        # Three-stage progressive search (WARP cc_spec_subp_sampling 1→2→3).
        shifts1, dify1 = cc_shift_search(spectra[refid], spectra[i], cshift=0.001, width=2.0, step=0.5)
        c1 = float(shifts1[int(np.argmin(dify1))])
        shifts2, dify2 = cc_shift_search(spectra[refid], spectra[i], cshift=c1, width=0.4, step=0.1)
        c2 = float(shifts2[int(np.argmin(dify2))])
        shifts3, dify3 = cc_shift_search(spectra[refid], spectra[i], cshift=c2, width=0.07, step=0.01)
        c3 = float(shifts3[int(np.argmin(dify3))])
        out[i] = c3 * cdelt1
    return out


def waveshift_clip(
    shift_matrix: NDArray[np.float64],
    *,
    sigma_1st: float = 1.0,
    sigma: float = 2.0,
    iterate: int = 5,
    std_thres: float = 0.1,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Sigma-clip per-frame median shift across orders.

    Args:
        shift_matrix: shape ``(n_orders, n_frames)`` of per-order shifts.

    Returns:
        ``(shift_average, shift_calcnum, shift_stddev, flag)`` —
        per-frame mean, kept-order count, stddev, and a quality flag
        (0 = good, ≥1 = flagged).
    """
    n_orders, n_frames = shift_matrix.shape
    shift_average = np.zeros(n_frames, dtype=np.float64)
    shift_calcnum = np.zeros(n_frames, dtype=np.int64)
    shift_stddev = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        vec = shift_matrix[:, f].copy()
        clip = np.zeros(n_orders, dtype=bool)
        avg = float(np.mean(vec))
        sd = float(np.std(vec))
        for k in range(iterate):
            kept = vec[~clip]
            if kept.size == 0:
                break
            avg = float(np.mean(kept))
            sd = float(np.std(kept))
            if sd == 0:
                break
            if k == 0:
                clip |= np.abs(vec - avg) / sd > sigma_1st
            elif k != iterate - 1:
                clip |= np.abs(vec - avg) / sd > sigma
        shift_average[f] = avg
        shift_stddev[f] = sd
        shift_calcnum[f] = int((~clip).sum())

    flag = np.zeros(n_frames, dtype=np.int64)
    flag[shift_stddev > std_thres] += 1
    flag[shift_calcnum < max(1, n_orders // 2)] += 1
    return shift_average, shift_calcnum, shift_stddev, flag


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    apdb_path: Path | str | None = None,
    refid: int | None = None,
    **_unused: Any,
) -> None:
    """Measure wavelength shifts across all frames and orders.

    Args:
        config: gated on ``flag_wsmeasure`` (default True).
        workdir: reads ``{obj}_NO{i}_sscfm_m{m}_1d.fits`` per frame
            per order (s08 output). Each spectrum is truncated in-memory
            to wavelength range [1, 2048] before cross-correlation —
            matches WARP's ``_m###c.fits`` input (``Warp_sci.py:468``).
            Writes ``waveshift_log.npz``.
        listfile: WARP-style input list.
        apdb_path: aperture database to enumerate orders.
        refid: index of the reference frame (0-indexed). If ``None``
            (default), picks ``argmax(per-frame median count summed
            over orders)`` when ``n_frames > 2``, else 0. Matches
            ``Warp_sci.py:603-610``.
    """
    if not config.flag_wsmeasure:
        return
    if apdb_path is None:
        raise ValueError("s10_waveshift_measure requires apdb_path until s00 wires in calib")

    cal_apset = ApertureSet.load(apdb_path)
    orders = (
        cal_apset.echelle_orders
        if (config.reduce_full_data or not config.selected_orders)
        else tuple(m for m in cal_apset.echelle_orders if m in config.selected_orders)
    )
    pairs = parse_listfile(listfile)

    # Load every 1D spectrum and its CDELT1.
    objnames: list[str] = []
    headers_list = []
    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objnames.append(_sanitize_objname(str(raw_objname)))
        headers_list.append(obj_header)

    n_frames = len(pairs)
    n_orders = len(orders)
    shift_matrix = np.zeros((n_orders, n_frames), dtype=np.float64)
    frame_ids = [f"{objnames[i]}_NO{i + 1}" for i in range(n_frames)]

    # Load + truncate every (frame, order) spectrum up-front. Truncating
    # in-memory here mirrors WARP's pipeline order (truncate → waveshift
    # → specshift+scombine) without needing a separate s09.5 file write.
    # Per-order spectra list, plus per-frame total median (used to pick
    # refid à la ``Warp_sci.py:603-610``).
    spectra_per_order: list[list[NDArray | None]] = [[None] * n_frames for _ in range(n_orders)]
    cdelt1_per_order: list[float | None] = [None] * n_orders
    counts = np.zeros(n_frames, dtype=np.float64)
    for oi, m in enumerate(orders):
        for i in range(n_frames):
            p = workdir / f"{objnames[i]}_NO{i + 1}_sscfm_m{m}_1d.fits"
            if not p.exists():
                continue
            data, hdr = _fits.read_image(p)
            trunc = scopy_wavelength_truncate(
                np.asarray(data, dtype=np.float64), hdr,
                w1=_TRUNC_W1, w2=_TRUNC_W2, reset_wcs=True,
            )
            spectra_per_order[oi][i] = trunc.data
            counts[i] += float(np.median(trunc.data))
            if cdelt1_per_order[oi] is None:
                cdelt1_per_order[oi] = float(trunc.header.get("CDELT1", 1.0))

    # Auto-pick refid from per-frame median counts when not specified.
    # WARP: ``refid = np.argmax(counts) if objnum > 2 else 0`` (``Warp_sci.py:610``).
    if refid is None:
        if n_frames > 2:
            refid = int(np.argmax(counts))
        else:
            refid = 0

    for oi, m in enumerate(orders):
        cdelt1 = cdelt1_per_order[oi]
        if cdelt1 is None:
            continue
        spectra = list(spectra_per_order[oi])
        valid_idx = [i for i, s in enumerate(spectra) if s is not None]
        if not valid_idx:
            continue
        actual_refid = refid if refid in valid_idx else valid_idx[0]
        ref_spec = spectra[actual_refid]
        for i, s in enumerate(spectra):
            if s is None:
                spectra[i] = ref_spec.copy()
        shifts = waveshift_one_order(spectra, cdelt1, refid=actual_refid)
        shift_matrix[oi, :] = shifts

    shift_avg, shift_n, shift_sd, flag = waveshift_clip(shift_matrix)
    table = WaveshiftTable(
        frame_ids=frame_ids,
        orders=orders,
        shift_matrix=shift_matrix,
        shift_average=shift_avg,
        shift_stddev=shift_sd,
        shift_calcnum=shift_n,
        flag=flag,
    )
    table.save_npz(workdir / "waveshift_log.npz")


# ---------------------------------------------------------------------------
# WARP-exact cross-frame shift (``warp/ccwaveshift.py``), bit for bit.
#
# The functions above are a float64 re-implementation. WARP itself runs the
# search in float32: the spectra are the float32 ``_m###c`` FITS data, the
# medians/normalisations/residuals stay float32 under NumPy 2 promotion, and
# the residual mean is NumPy's pairwise float32 sum. The argmin over the
# 0.01-pixel grid is only stable against WARP if the arithmetic is the same,
# so the functions below call the very same numpy/scipy operations on the
# very same dtypes.
# ---------------------------------------------------------------------------

def warp_cc_spec_subp_sampling(
    sp1y: NDArray[np.float32],
    sp2y: NDArray[np.float32],
    cshift: float,
    width: float,
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """``warp/ccwaveshift.py:cc_spec_subp_sampling`` (ver 2.3).

    ``sp1y`` is the reference spectrum after WARP's ``PySpecshift(sp1, tmp,
    0.)`` (scopy + specshift 0 + scombine onto its own grid, which IRAF
    short-circuits to a copy), ``sp2y`` the raw ``_m###c`` data of the other
    frame; both float32.
    """
    import scipy.ndimage

    sp1y = np.asarray(sp1y, dtype=np.float32)
    sp2y = np.asarray(sp2y, dtype=np.float32)
    normalization_sp2 = np.median(sp2y)
    sp1y_norm = sp1y / np.median(sp1y)
    div = 100
    ec_short = 800
    ec_long = 300
    naxis1 = len(sp1y)
    start_sp1 = int(naxis1 / div)
    end_sp1 = int(naxis1 / div * (div - 1))
    shiftnum = int(width / step * 2 + 1)
    shift = [cshift - width + float(i) * step for i in range(shiftnum)]
    dify = []
    for i in range(shiftnum):
        sp2y_shifted = scipy.ndimage.shift(sp2y, shift[i], order=3)
        sp2y_norm_shifted = sp2y_shifted / normalization_sp2
        tmpydif = (sp1y_norm[start_sp1 + ec_short:end_sp1 - ec_long]
                   - sp2y_norm_shifted[start_sp1 + ec_short:end_sp1 - ec_long]) ** 2.
        dify.append(np.average(tmpydif))
    return np.array(shift), np.array(dify)


def warp_waveshift_oneorder(
    spectra: list[NDArray[np.float32]],
    refid: int,
    cdelt1: float,
) -> list[float]:
    """``warp/ccwaveshift.py:waveshift_oneorder``: shift (Å) of every frame.

    Args:
        spectra: per-frame float32 ``_m###c`` spectra of one order.
        refid: reference frame index (its entry stays exactly 0.0).
        cdelt1: the reference file's CDELT1 as WARP reads it (float).
    """
    wshift_vec: list = [0.0 for _ in range(len(spectra))]
    sp1y = np.asarray(spectra[refid], dtype=np.float32)
    for i in range(len(spectra)):
        if i == refid:
            continue
        sp2y = np.asarray(spectra[i], dtype=np.float32)
        shift_1, dify_1 = warp_cc_spec_subp_sampling(sp1y, sp2y, 0.001, 2.0, 0.5)
        cshift_p = shift_1[np.argmin(dify_1)]
        shift_2, dify_2 = warp_cc_spec_subp_sampling(sp1y, sp2y, cshift_p, 0.4, 0.1)
        cshift_sp = shift_2[np.argmin(dify_2)]
        shift_3, dify_3 = warp_cc_spec_subp_sampling(sp1y, sp2y, cshift_sp, 0.07, 0.01)
        cshift_ssp = shift_3[np.argmin(dify_3)]
        wshift_vec[i] = cshift_ssp * cdelt1
    return wshift_vec


def warp_waveshift_clip(wshift_matrix, objnum: int, aplength: int, sigma_1st=1.,
                        sigma=2., iterate=5, stdthres=0.1):
    """``warp/ccwaveshift.py:waveshiftClip`` (ver 2.3), verbatim."""
    shift_average = []
    shift_calcnum = []
    shift_stddev = []
    with np.errstate(all="ignore"):
        for i in range(objnum):
            shiftvec = []
            shiftclip = np.array([0 for jj in range(aplength)])
            for j in range(aplength):
                shiftvec.append(wshift_matrix[j][i])
            shiftvecarray = np.array(shiftvec)
            for k in range(iterate):
                shift_average_ite = np.average(shiftvecarray[shiftclip == 0])
                shift_std_ite = np.std(shiftvecarray[shiftclip == 0])
                if k == 0:
                    shiftclip[np.absolute(shiftvecarray - shift_average_ite)
                              / shift_std_ite > sigma_1st] += 1
                elif k != iterate - 1:
                    shiftclip[np.absolute(shiftvecarray - shift_average_ite)
                              / shift_std_ite > sigma] += 1
            shift_average.append(shift_average_ite)
            shift_stddev.append(shift_std_ite)
            shift_calcnum.append(np.sum(shiftclip == 0))
        shift_stddev_arr = np.array(shift_stddev)
        shift_calcnum_arr = np.array(shift_calcnum)
        wshift_flag = np.zeros(objnum)
        wshift_flag[shift_stddev_arr > stdthres] += 1
        wshift_flag[shift_calcnum_arr < aplength / 2] += 1
    return shift_average, shift_calcnum, shift_stddev, wshift_flag


def warp_reference_frame(spectra_by_order: list[list[NDArray[np.float32]]]) -> int:
    """WARP's reference frame for the shift search (``Warp_sci.py`` s10).

    ``counts[i] += np.median(spy)`` over orders (float32 under NumPy 2,
    because the ``0.`` start is a weak Python float), then
    ``refid = argmax(counts)`` when there are more than two frames, else 0.
    """
    nframes = len(spectra_by_order[0])
    counts = [0. for _ in range(nframes)]
    for i in range(nframes):
        for row in spectra_by_order:
            counts[i] += np.median(np.asarray(row[i], dtype=np.float32))
    return int(np.argmax(counts)) if nframes > 2 else 0


@dataclass(frozen=True)
class WarpWaveshiftLog:
    """Contents of WARP's ``reduction_log/waveshift_log.npz`` (``warp/logger.py``)."""

    waveShift: NDArray          # (n_orders, n_frames) Å
    waveShiftAve: NDArray       # (n_frames,)
    waveShiftStd: NDArray
    waveShiftNum: NDArray       # int
    waveShiftAdopted: NDArray   # shift actually applied in s11
    echelleOrder: NDArray
    frameNum: int
    refid: int
    flag: NDArray

    def save_npz(self, path: Path) -> None:
        """Write the npz with WARP's keys (``warpLog.writeWaveshiftLogNpz``)."""
        np.savez(path, waveShift=self.waveShift, waveShiftAve=self.waveShiftAve,
                 waveShiftStd=self.waveShiftStd, waveShiftNum=self.waveShiftNum,
                 waveShiftAdopted=self.waveShiftAdopted,
                 echelleOrder=self.echelleOrder, frameNum=self.frameNum)

    def save_text(self, path: Path) -> None:
        """Write the text log (``warpLog.writeWaveshiftLogText``)."""
        n = self.frameNum
        title = "Log of Waveshift"
        lines = ["#" * 72, "### %s %s" % (title, "#" * (72 - len(title) - 5)), "#" * 72, "",
                 "# Pixel shift relative to 1st frame.", ""]
        out = "\n".join(lines) + "\n"
        out += "Files\t" + "".join("No.%d\t" % (i + 1) for i in range(n)) + "\n\n"
        for j, m in enumerate(self.echelleOrder):
            out += "m=%d\t" % m + "".join("%.4f\t" % self.waveShift[j][i] + "\t"
                                          for i in range(n)) + "\n"
        # WARP prints frameNum entries; the npz keeps whole lists (longer
        # when a listfile line has several ws= tokens)
        out += "\nAverage:\t" + "".join("%.4f\t" % self.waveShiftAve[i] for i in range(n))
        out += "\nStd dev:\t" + "".join("%.4f\t" % self.waveShiftStd[i] for i in range(n))
        out += "\nNumber of orders:\t" + "".join("%d\t" % self.waveShiftNum[i]
                                                 for i in range(n))
        out += "\n\nAdopted shift:\t" + "".join("%.4f\t" % self.waveShiftAdopted[i]
                                               for i in range(n))
        Path(path).write_text(out)


def warp_waveshift_log(
    spectra_by_order: list[list[NDArray[np.float32]]],
    orders: tuple[int, ...] | list[int],
    cdelt1: float,
    *,
    measure: bool = True,
    correct: bool = True,
    refid: int | None = None,
    n_manual: int | None = None,
) -> WarpWaveshiftLog:
    """WARP's whole s10 block (``Warp_sci.py`` "Shift correction"), exactly.

    Args:
        spectra_by_order: ``[order][frame]`` float32 ``_m###c`` spectra.
        orders: echelle orders, same order as ``spectra_by_order``.
        cdelt1: CDELT1 of the ``_m###c`` files (0.5 for HIRES, 1.0 for WIDE).
        measure / correct: WARP's ``flag_wsmeasure`` / ``flag_wscorrect``
            (``Measure Shift`` / ``Correct Shift`` in the parameter file).
        refid: override the reference frame (default: WARP's choice).
        n_manual: the length of WARP's ``conf.waveshift_man`` (one entry per
            listfile line, more when a line has several ``ws=`` tokens;
            ``decanter.warptree.warp_flat_tokens``); WARP sizes its zero
            lists by it (Warp_sci.py:622-631).  Default: the frame count.
    """
    nframes = len(spectra_by_order[0])
    aplength = len(orders)
    nman = nframes if n_manual is None else int(n_manual)
    if nframes > 1 and measure:
        if refid is None:
            refid = warp_reference_frame(spectra_by_order)
        wshift_matrix = [warp_waveshift_oneorder(row, refid, cdelt1)
                         for row in spectra_by_order]
        ave, num, std, flag = warp_waveshift_clip(wshift_matrix, nframes, aplength)
        if correct:
            cor = [ave[i] if flag[i] == 0 else 0. for i in range(nframes)]
        else:
            cor = [0. for _ in range(nman)]
    else:
        refid = 0 if refid is None else refid
        wshift_matrix = [[0. for _ in range(nframes)] for _ in range(aplength)]
        ave = [0. for _ in range(nman)]
        std = [0. for _ in range(nman)]
        num = [0 for _ in range(nman)]
        cor = [0. for _ in range(nman)]
        flag = np.zeros(nframes)
    return WarpWaveshiftLog(
        waveShift=np.array(wshift_matrix), waveShiftAve=np.array(ave),
        waveShiftStd=np.array(std), waveShiftNum=np.array(num),
        waveShiftAdopted=np.array(cor), echelleOrder=np.array(list(orders)),
        frameNum=nframes, refid=int(refid), flag=np.asarray(flag),
    )
