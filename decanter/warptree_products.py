"""WARP's per-night products beyond the VAC spectra (``Warp_sci.py``).

* :func:`iraf_apdb_text` -- the aperture databases IRAF itself writes
  (``apextract/apdb.x:ap_dbwrite``): ``ap<OBJ>_skyNO<i>_fm_m<m>trans``
  (``apall`` of the sky strip with the object's per-frame database as
  reference), ``ap<OBJ>_NO<i>_s`` (``apscatter``) and ``aplast``.
* :func:`spectrum_products` -- per FSR-cut spectrum: ``VAC_norm``,
  ``VAC_cont``, ``AIR``, ``AIR_norm``, ``AIR_cont`` (obj), ``AIR`` (sky).
* :func:`sum_products` -- ``<OBJ>_sum``: ``scombine`` average of the frames
  (2..28 frames) or ``scopy`` of frame 1, continuum, air, and the
  ``*_norm_comb`` all-order combinations.
* :func:`snr_estimate` -- ``SNR_dat/fsr<cut>/SNratio_m<m>_fsr<cut>.dat``
  (``warp/SNratio_estimate.py``).
* the text logs (``warp/logger.py``, ``apscat_log``, ``cutransform_log``).
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np

from decanter.utils.fpemu import iraf_ctod
from decanter.utils.iraf_fmt import dtoc3
from decanter.utils.iraf_time import dtptime
from decanter.wavelength import iraf_onedspec as od

R = np.float32


# ---------------------------------------------------------------------------
# IRAF aperture databases
# ---------------------------------------------------------------------------

def _g(tok: str) -> str:
    """``%g`` of the REAL IRAF parses from database token ``tok`` (pargr)."""
    return dtoc3(float(R(iraf_ctod(tok))), 7, "g")


def iraf_dtime() -> str:
    """``dtptime`` (xtools/dttext.x:564-576): ``'# ' + cnvtime (clktime (0))``,
    ``# Dow hh:mm:ss dd-Mmm-yyyy`` on IRAF's clock (local standard time, no
    daylight saving; :mod:`decanter.utils.iraf_time`)."""
    return dtptime()


def _blocks(text: str) -> list[list[str]]:
    lines = text.splitlines()
    out: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if ln.startswith("begin"):
            cur = [ln]
            out.append(cur)
        elif cur is not None and ln.strip() and not ln.startswith("#"):
            cur.append(ln)
    return out


def iraf_apdb_text(source_text: str, image: str, *, stamp: str | None = None,
                   orders: list[int] | None = None) -> str:
    """``ap_dbwrite(image, aps)`` for the apertures read from ``source_text``.

    The reference database is parsed as ``ap_dbread`` does (REAL values
    through ``ctod``) and every aperture is written back with ``%g``
    (7 significant digits) under the new image name, each record preceded
    by a ``dtptime`` line.
    """
    stamp = stamp or iraf_dtime()
    out = []
    for blk in _blocks(source_text):
        b = blk[0].split()
        ap = int(b[3])
        if orders is not None and ap not in orders:
            continue
        kv: dict[str, list[str]] = {}
        curve: list[str] = []
        bg: dict[str, list[str]] = {}
        i = 1
        in_bg = False
        while i < len(blk):
            toks = blk[i].split()
            key = toks[0]
            if key == "background":
                in_bg = True
            elif key == "axis":
                in_bg = False
                kv["axis"] = toks[1:]
            elif key == "curve":
                n = int(toks[1])
                curve = [blk[i + k].split()[0] for k in range(1, n + 1)]
                i += n
            elif in_bg:
                bg[key] = toks[1:]
            else:
                kv[key] = toks[1:]
            i += 1
        cen = kv["center"]
        s = [stamp + "\n"]
        s.append(f"begin\taperture {image} {ap} {_g(cen[0])} {_g(cen[1])}\n")
        if "title" in kv:
            s.append(f"\ttitle\t{' '.join(kv['title'])}\n")
        s.append(f"\timage\t{image}\n")
        s.append(f"\taperture\t{ap}\n")
        s.append(f"\tbeam\t{int(kv['beam'][0])}\n")
        s.append(f"\tcenter\t{_g(cen[0])} {_g(cen[1])}\n")
        s.append(f"\tlow\t{_g(kv['low'][0])} {_g(kv['low'][1])}\n")
        s.append(f"\thigh\t{_g(kv['high'][0])} {_g(kv['high'][1])}\n")
        if bg:
            s.append("\tbackground\n")
            s.append(f"\t\txmin {_g(bg['xmin'][0])}\n")
            s.append(f"\t\txmax {_g(bg['xmax'][0])}\n")
            s.append(f"\t\tfunction {bg['function'][0]}\n")
            s.append(f"\t\torder {int(bg['order'][0])}\n")
            s.append(f"\t\tsample {' '.join(bg['sample'])}\n")
            s.append(f"\t\tnaverage {int(bg['naverage'][0])}\n")
            s.append(f"\t\tniterate {int(bg['niterate'][0])}\n")
            s.append(f"\t\tlow_reject {_g(bg['low_reject'][0])}\n")
            s.append(f"\t\thigh_reject {_g(bg['high_reject'][0])}\n")
            s.append(f"\t\tgrow {_g(bg['grow'][0])}\n")
        s.append(f"\taxis\t{int(kv['axis'][0])}\n")
        s.append(f"\tcurve\t{len(curve)}\n")
        for c in curve:
            s.append(f"\t\t{_g(c)}\n")
        s.append("\n")
        out.append("".join(s))
    return "".join(out)


# ---------------------------------------------------------------------------
# Per-spectrum products
# ---------------------------------------------------------------------------

def spectrum_products(vac: od.Spectrum1D, order: int, *, sky: bool = False) -> dict:
    """WARP's derived spectra of one ``_fsr<cut>_VAC`` spectrum.

    Obj: ``{"VAC_norm", "VAC_cont", "AIR", "AIR_norm", "AIR_cont"}``
    (``PyContinuum``: ``continuum`` + ``sarith /``; ``vac2air_spec``);
    sky: ``{"AIR"}``.
    """
    if sky:
        return {"AIR": od.vac2air(vac)}
    norm = od.continuum(vac, **od.continuum_params(order))
    return _products_from_norm(vac, norm)


def _products_from_norm(vac: od.Spectrum1D, norm: od.Spectrum1D) -> dict:
    cont = od.sarith_divide(vac, norm)
    return {"VAC_norm": norm, "VAC_cont": cont, "AIR": od.vac2air(vac),
            "AIR_norm": od.vac2air(norm), "AIR_cont": od.vac2air(cont)}


def spectrum_products_many(vacs: list, orders: list, *, batched: bool = True) -> list:
    """:func:`spectrum_products` (obj) of several spectra; ``batched`` fits
    their continua together (:func:`decanter.wavelength.iraf_onedspec.continuum_many`,
    the same bits), else one at a time."""
    if not batched:
        return [spectrum_products(v, m) for v, m in zip(vacs, orders, strict=True)]
    norms = od.continuum_many(vacs, [od.continuum_params(m) for m in orders])
    return [_products_from_norm(v, n) for v, n in zip(vacs, norms, strict=True)]


def sum_spectra(frames_vac: list[od.Spectrum1D], order: int, *, combine: bool) -> dict:
    """``<OBJ>_sum`` for one (order, cut).

    ``combine`` (``1 < frames <= frameNumberLimit``): ``PyScombine`` of the
    frames' VAC spectra, then continuum/air as for a frame.  Otherwise
    ``scopy`` of frame 1's six products.
    """
    from decanter.wavelength.iraf_scombine import scombine_average
    if combine:
        vac = scombine_average(frames_vac)
        out = {"VAC": vac}
        out.update(spectrum_products(vac, order))
        return out
    raise ValueError("use sum_copy for the scopy path")


def sum_copy(frame1: dict) -> dict:
    """``iraf.scopy`` of frame 1's ``{VAC, VAC_norm, VAC_cont, AIR, AIR_norm, AIR_cont}``."""
    return {k: od.scopy(v) for k, v in frame1.items()}


# ---------------------------------------------------------------------------
# SNR estimate (warp/SNratio_estimate.py)
# ---------------------------------------------------------------------------

def _openspec_data(spec: od.Spectrum1D):
    h = spec.header
    n = spec.n
    c = float(h["CRVAL1"])
    d = float(h["CDELT1"])
    p = float(h["CRPIX1"])
    return np.array([c + d * (l - p + 1.) for l in range(n)]), np.asarray(spec.data, dtype=R)


def snr_estimate(specs: list[od.Spectrum1D], names: list[str], *,
                 batched: bool = False) -> tuple[str, list, list]:
    """``snestimate`` on the frames' VAC spectra of one (order, cut).

    Returns the ``.dat`` text and ``(lamcen, sntotal)``.  Every IRAF step
    (``sarith /`` by the median, ``scombine`` of the first 10, ``continuum``
    ``order=10``, ``sarith -``, ``continuum type=difference``) is the
    bit-exact port; the statistics are WARP's numpy calls.  ``batched`` fits
    the average's and the differences' continua together
    (:func:`decanter.wavelength.iraf_onedspec.continuum_many`, the same bits).
    """
    from decanter.wavelength.iraf_scombine import scombine_average
    clipsigma = 5
    thres_low = 0.5
    thres_high = 1.3
    dividerange = 3
    binning_pix = 1
    binnum = 20
    spnorm = []
    for s in specs:
        med = np.median(np.asarray(s.data, dtype=R))
        spnorm.append(od.sa_1d_const(s, "/", str(med)))
    ave = scombine_average(specs[:10])
    ave_prm = dict(order=10, niterate=10, low=2.0, high=3.0, grow=1.0)
    dif_prm = dict(order=10, niterate=10, low=3.0, high=3.0, grow=1.0, out_type="difference")
    difs = [od.sa_1d(spnorm[i], "-", spnorm[0 if i == len(specs) - 1 else i + 1])
            for i in range(len(specs))]
    if batched:
        ave_norm, *difns = od.continuum_many([ave, *difs], [ave_prm] + [dif_prm] * len(difs))
    else:
        ave_norm = od.continuum(ave, **ave_prm)
        difns = [od.continuum(d, **dif_prm) for d in difs]
    avelamx_tmp, avedata_tmp = _openspec_data(ave_norm)
    avelamx, avedata = _binning(avelamx_tmp, avedata_tmp, binning_pix)
    minpix = 0
    maxpix = len(avelamx) - 1
    widthpix = int((maxpix - minpix) / dividerange)
    minr = [minpix + widthpix * i for i in range(dividerange)]
    maxr = [minpix + widthpix * (i + 1) for i in range(dividerange)]
    lamcen = [(avelamx[maxr[i]] + avelamx[minr[i]]) / 2. for i in range(dividerange)]
    sn_vec: list[list] = [[] for _ in range(dividerange)]
    pairs = []
    stddev: list[list] = [[] for _ in range(len(specs))]
    for i in range(len(specs)):
        j = 0 if i == len(specs) - 1 else i + 1
        lamx_tmp, normdata_tmp = _openspec_data(difns[i])
        lamx, normdata = _binning(lamx_tmp, normdata_tmp, binning_pix)
        normlength = len(lamx)
        pairs.append("%d--%d:\t\t" % ((i + 1), (j + 1)))
        for k in range(dividerange):
            hi = min(maxr[k], normlength)
            nd = normdata[minr[k]:hi]
            ad = avedata[minr[k]:hi]
            meddif = np.median(nd)
            stddif = np.std(nd)
            clipped = []
            for n in range(len(nd)):
                if math.fabs(nd[n] - meddif) < stddif * clipsigma and thres_high > ad[n] > thres_low:
                    clipped.append(nd[n])
            std_cl = np.std(clipped)
            stddev[i].append(std_cl)
            sn_vec[k].append(std_cl ** 2)
    sn_sum = [sum(sn_vec[k]) / 2. for k in range(dividerange)]
    sntotal = [len(specs) / math.sqrt(sn_sum[k]) for k in range(dividerange)]
    w = [f"Time: {datetime.datetime.today()}\n\nInput: \t\n"]
    for nm in names:
        w.append("\t\t%s\n" % nm)
    w.append("\nSigma for clipping: \t%d\n" % clipsigma)
    w.append("Threshold: \t%.2f-%.2f\n" % (thres_low, thres_high))
    w.append("# of division: \t%d\n" % dividerange)
    w.append("Binning size in pix: \t%d\n" % binning_pix)
    w.append("# of bins: \t%d\n" % binnum)
    w.append("\n\n")
    w.append("Frame pairs\t")
    for i in range(dividerange):
        w.append("%.1f\t" % lamcen[i])
    w.append("\n\n")
    for i in range(len(specs)):
        w.append(pairs[i])
        for k in range(dividerange):
            w.append("%.1e\t" % stddev[i][k])
        w.append("\n")
    w.append("\n\nS/N (total): \t")
    for k in range(dividerange):
        w.append("%.1f\t" % sntotal[k])
    w.append("\n\n")
    return "".join(w), lamcen, sntotal


def _binning(lambdax, fluxy, binning_size):
    """``Spec1Dtools.binning_spec`` (Python scalar sums)."""
    lx, fy = [], []
    for i in range(len(lambdax) // binning_size):
        tmp_x = 0.
        tmp_y = 0.
        for j in range(binning_size):
            tmp_x += lambdax[binning_size * i + j]
            tmp_y += fluxy[binning_size * i + j]
        lx.append(tmp_x / binning_size)
        fy.append(tmp_y / binning_size)
    return np.array(lx), np.array(fy)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def _log_title(title: str, description: str) -> str:
    s = "#" * 72 + "\n" + "### %s %s\n" % (title, "#" * (72 - len(title) - 5)) + "#" * 72 + "\n\n"
    for d in description.split("\n"):
        s += "# %s\n" % d
    return s + "\n"


def _files_header(n: int) -> str:
    return "Files\t" + "".join("No.%d\t" % (i + 1) for i in range(n)) + "\n\n"


def write_centersearch_log(txt: Path, npz: Path, center: np.ndarray, width: np.ndarray,
                           orders) -> None:
    """``warpLog.writePsfLogText`` / ``writePsfLogNpz``."""
    nf, no = center.shape
    s = _log_title("Log of Center Search", "Gaussian profile is assumed.\n" +
                   "Peak position (pix) from the center position in the aperture definition.")
    s += _files_header(nf)
    for j, m in enumerate(orders):
        s += "m=%d\t" % m + "".join(("%.4f\t" % center[i][j]) + "\t" for i in range(nf)) + "\n"
    s += "\nMedian:\t" + "".join("%.4f\t" % np.nanmedian(center[i]) for i in range(nf))
    s += "\n\n# FWHM (pix) of fitted Gaussian profiles. \n\n"
    s += _files_header(nf)
    for j, m in enumerate(orders):
        s += "m=%d\t" % m + "".join(("%.4f\t" % width[i][j]) + "\t" for i in range(nf)) + "\n"
    s += "\nMedian:\t" + "".join("%.4f\t" % np.nanmedian(width[i]) for i in range(nf))
    Path(txt).write_text(s)
    np.savez(npz, psfCenter=center, psfWidth=width, echelleOrder=list(orders), frameNum=nf)


def write_cosmicray_log(txt: Path, npz: Path, sigma: list, num: list, orders) -> None:
    """``warpLog.writeCosmicRayLogText`` / ``writeCosmicRayLogNpz``."""
    s = _log_title("Log of cosmic ray detection", "Number of cosmic rays and threshold.")
    for i in range(len(num)):
        s += "No.%d: %d\t%.1f\n" % (i + 1, num[i], sigma[i])
    Path(txt).write_text(s)
    np.savez(npz, cosmicRaySigma=np.array(sigma), cosmicRayNum=np.array(num),
             echelleOrder=list(orders), frameNum=len(num))


def apscatter_log_entry(inp: str, out: str, base: str) -> str:
    """``warp/apscatter.py:apscat_log`` (one call)."""
    s = "########################\n### Log of apscatter ###\n########################\n\n"
    s += "Files: \n\tinput:\t %s\n\toutput:\t %s\n\treference:\t %s\n\n" % (inp, out, base)
    s += "Parameters: \n"
    s += "\tapextract.apscatter.apscat1.order=%d\n" % 3
    s += "\tapextract.apscatter.apscat1.sample=%s\n" % "10:2000"
    s += "\tapextract.apscatter.apscat1.low_reject=%.1f\n" % 3.0
    s += "\tapextract.apscatter.apscat1.high_reject=%.1f\n" % 2.0
    s += "\tapextract.apscatter.apscat1.niterate=%d\n" % 100
    s += "\tapextract.apscatter.apscat1.function=%s\n" % "legendre"
    s += "\tapextract.apscatter.apscat2.order=%d\n" % 5
    s += "\tapextract.apscatter.apscat2.sample=%s\n" % "10:2000"
    s += "\tapextract.apscatter.apscat2.niterate=%d\n" % 20
    s += "\tapextract.apscatter.apscat2.function=%s\n" % "legendre"
    return s + "\n\n\n"


def cutransform_log_entry(inp: str, ref: str, dy: str, flux: str) -> str:
    """``warp/cutransform.py:cutransform_log`` (one call)."""
    s = "##########################\n### Log of cutransform ###\n##########################\n\n"
    s += "Files: \n\tinput:\t %s\n\treference:\t %s\n\n" % (inp, ref)
    s += "Parameters: \n\txmargin=%d\n\tinterptype=%s\n\tdy=%s\n\tflux=%s\n" % (
        100, "spline3", dy, flux)
    return s + "\n\n\n"


def spatial_profile_dat(med_x, med_y) -> str:
    """``centersearch_fortrans`` / ``make_slit_profile`` ``.dat`` text."""
    return "".join("%.4f\t%.4f\n" % (med_x[i], med_y[i]) for i in range(len(med_x)))


def _hkr(h, key):
    """``Spec2Dtools.header_key_read``: the value, or ``"N/A"``."""
    try:
        return h[key]
    except KeyError:
        return "N/A"


def status_text(*, version: str, start: str, end: str, elapsed: float, lines, images: list,
                image_headers: list, comp_header, flat_header, cfg) -> str:
    """``reduction_log/status.txt`` (``warp/config.py``: ``writeStatus``,
    ``writeInputDataList``, ``checkDataStatus(statusOutput=True)``,
    ``writeParam``, ``writeElapsedTime``).

    The per-image arrays are built with ``np.append`` and printed with
    ``str.format`` as WARP does (numpy's array printing).
    """
    s = "WARP ver.%s\n" % version
    s += "Starting time: %s\n" % start
    s += "\n"
    s += "Input data:\n"
    for ln in lines:
        # writeInputDataList runs before the parameter file is read, so the
        # flags are WARP's defaults: no ap=/bg=/ws= tokens.
        s += "    {} {}\n".format(ln.obj, ln.sky)
    s += "\n"
    setting = np.array([])
    period = np.array([])
    slit = np.array([])
    instmode = np.array([])
    for h in image_headers:
        setting = np.append(setting, _hkr(h, "SETTING"))
        period = np.append(period, _hkr(h, "PERIOD"))
        slit = np.append(slit, _hkr(h, "SLIT"))
        instmode = np.append(instmode, _hkr(h, "INSTMODE"))
    s += "Data status:\n"
    s += "# Instrument setting IDs from fits header\n"
    s += "    Inputs: {}\n".format(setting)
    s += "    Calibs (comp,flat): {}, {}\n".format(_hkr(comp_header, "SETTING"),
                                                   _hkr(flat_header, "SETTING"))
    s += "# Period IDs from fits header\n"
    s += "    Inputs: {}\n".format(period)
    s += "    Calibs (comp,flat): {}, {}\n".format(_hkr(comp_header, "PERIOD"),
                                                   _hkr(flat_header, "PERIOD"))
    s += "# Slit from fits header\n"
    s += "    Inputs: {}\n".format(slit)
    s += "    Calibs (comp, flat): {}, {}\n".format(_hkr(comp_header, "SLIT"),
                                                    _hkr(flat_header, "SLIT"))
    s += "# Mode from fits header\n"
    s += "    Inputs: {}\n".format(instmode)
    s += "    Calibs (comp, flat): {}, {}\n".format(_hkr(comp_header, "INSTMODE"),
                                                    _hkr(flat_header, "INSTMODE"))
    s += "\n"
    s += "WARP Settings and Parameters:\n"
    s += "    flag_apscatter: {}\n".format(cfg.flag_apscatter)
    s += "    flag_manual_aperture: {}\n".format(cfg.flag_manual_aperture)
    s += "    flag_skysub: {}\n".format(cfg.flag_skysub)
    s += "    flag_bpmask: {}\n".format(cfg.flag_bpmask)
    s += "    flag_skyemission: {}\n".format(cfg.flag_skyemission)
    s += "    flag_wsmeasure: {}\n".format(cfg.flag_wsmeasure)
    s += "    flag_wscorrect: {}\n".format(cfg.flag_wscorrect)
    s += "    flag_wsmanual: {}\n".format(cfg.flag_wsmanual)
    s += "    flag_extract2d: {}\n".format(cfg.flag_extract2d)
    s += "    skysub_mode: {}\n".format(cfg.skysub_mode)
    s += "    skysubModeList: {}\n".format(["none", "average", "median", "minimum", "fit"])
    s += "    cutrange_list: {}\n".format([float(c) for c in cfg.cutrange_list])
    s += "    fluxinput: {}\n".format(cfg.fluxinput)
    s += "    CRthreshold: {}\n".format(float(cfg.CR_threshold))
    s += "    CRvaratio: {}\n".format(float(cfg.CR_var_ratio))
    s += "    CRslitposratio: {}\n".format(float(cfg.CR_slitpos_ratio))
    s += "    CRmaxsigma: {}\n".format(float(cfg.CR_max_sigma))
    s += "    CRfixsigma: {}\n".format(cfg.CR_fix_sigma)
    s += "    reduceFullData: {}\n".format(cfg.reduce_full_data)
    s += "    selectedOrders: {}\n".format([int(m) for m in cfg.selected_orders])
    s += "\n"
    s += "Termination time: %s\n" % end
    s += "Elapsed time: %dm%.1fs\n\n" % (int(elapsed / 60), elapsed % 60)
    s += "Pipeline status: {}".format("Successfully finished.")
    return s
