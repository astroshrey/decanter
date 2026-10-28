"""WARP's figures (``warp/plotframes.py``), drawn from the tree's own files.

Every PNG/PDF ``Warp_sci.py`` keeps is drawn here by a line-for-line port of
the WARP function that draws it, fed the same files WARP reads (the FITS
products and ``.dat`` profiles of the tree, which are WARP's to the bit), so
with the same matplotlib/Pillow the PNGs are WARP's bytes; PDFs differ only
in their ``/CreationDate``.

==============================================  ===============================
file                                            WARP function
==============================================  ===============================
``rawdata_image/<raw>.png``                     ``plot_2dimages`` (raw copy)
``<OBJ>_NO<i>/images/2d_image/*_sscfm.png``     ``plot_2dimages``
``<OBJ>_NO<i>/images/badpixmask/mask_*.png``    ``plot_2dimages_mask``
``<OBJ>_NO<i>/images/badpixmask/mask_*.pdf``    ``cosmicRay2dImages``
``<OBJ>_NO<i>/images/spatial_profile/*.png``    ``aperture_plot``
``spectra_image/<OBJ>_VACnorm.pdf``             ``plot_all_frames_norm``
``spectra_image/<OBJ>_VACflux.pdf``             ``plot_all_frames_flux``
``spectra_image/<OBJ>_AIRnorm_combined_m*.png`` ``plot_combined_norm``
``SNR_dat/fsr<cut>/SNratio_fsr<cut>.png``       ``snr_plots``
``reduction_log/count_and_fwhm.png``            ``peak_count_fwhm``
==============================================  ===============================

Titles carry WARP's work names (``shortFile``: an OBJECT longer than 13
characters is ``OBJECT``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

colornames = ["b", "g", "r", "c", "m", "y"]
maxfnum = 6


def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "macosx", "pdf"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _savefig(plt, *args, **kwargs) -> None:
    """WARP's ``plt.savefig(...)``: the current figure's ``savefig``.

    ``pyplot.savefig`` is ``gcf().savefig(...)`` followed by
    ``canvas.draw_idle()``, which on a non-interactive canvas (Agg) renders
    the whole figure a second time after the file is written; that render
    changes nothing in the file, so it is skipped.
    """
    plt.gcf().savefig(*args, **kwargs)


def _savefig_small(plt, pngim, **kwargs) -> None:
    """WARP's ``plt.savefig(pngim, ...)`` followed by ``savefig_small(pngim)``.

    ``savefig_small`` re-reads the PNG matplotlib wrote and saves its RGB
    quantised to an adaptive palette; the PNG's pixels are the RGBA buffer
    ``savefig`` renders, so that buffer is quantised directly (no PNG encode
    and decode in between; the file is the same).  Falls back to the two
    steps if the buffer is not the expected size.
    """
    import io

    import matplotlib as mpl
    from PIL import Image
    fig = plt.gcf()
    kwargs.pop("format", None)
    dpi = kwargs.get("dpi")
    if dpi is None or dpi == "figure":
        dpi = mpl.rcParams["savefig.dpi"]
        if dpi == "figure":
            dpi = fig.dpi
    wi, hi = fig.get_size_inches()
    w, h = int(wi * dpi), int(hi * dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", **kwargs)
    data = buf.getvalue()
    if len(data) != w * h * 4:
        fig.savefig(pngim, format="png", **kwargs)
        savefig_small(pngim)
        return
    im = Image.frombuffer("RGBA", (w, h), data, "raw", "RGBA", 0, 1)
    im.convert('RGB').convert('P', palette=Image.ADAPTIVE).save(pngim, format="PNG")


def _image_data(src):
    """The pixel array WARP's ``fits.open(name)[0].data`` gives: ``src`` itself
    when it is already the array (the same values as the file), else read."""
    if isinstance(src, np.ndarray):
        return src
    from astropy.io import fits
    fimg = fits.open(src)
    pixdata = fimg[0].data
    fimg.close()
    return pixdata


_GEOMETRY: dict = {}


def cr_plot_geometry(apdb_path, xlim1=-30, xlim2=30):
    """``cosmicRay2dImages``'s calibration-only arrays for an aperture file:
    ``(maskap, slitcoord, array_length, echelle_orders)``, computed once per
    process (``apmaskArray`` / ``slitcoordArray`` of the multihole set)."""
    from decanter.calib.aperture import ApertureSet
    p = Path(apdb_path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size, float(xlim1), float(xlim2))
    if key not in _GEOMETRY:
        apset = ApertureSet.load(p)
        if len(_GEOMETRY) >= 4:
            _GEOMETRY.pop(next(iter(_GEOMETRY)))
        _GEOMETRY[key] = (apset.apmask_array(low_lim=xlim1, upp_lim=xlim2),
                          apset.slitcoord_array(low_lim=xlim1, upp_lim=xlim2),
                          apset.array_length, list(apset.echelle_orders))
    return _GEOMETRY[key]


def openspecfits(fitsfile):
    """``warp/Spec1Dtools.py:openspecfits``."""
    from astropy.io import fits
    fitsfile = str(fitsfile)
    if fitsfile.find("fits") == -1:
        fitsfile += ".fits"
    spfits = fits.open(fitsfile)
    splength = spfits[0].header["NAXIS1"]
    spdata = spfits[0].data
    rcrval1 = float(spfits[0].header["CRVAL1"])
    rcdelt1 = float(spfits[0].header["CDELT1"])
    rcrpix1 = float(spfits[0].header["CRPIX1"])
    lamx = np.array([rcrval1 + rcdelt1 * (l - rcrpix1 + 1.) for l in range(splength)])
    spfits.close()
    return lamx, spdata, rcrval1, rcdelt1, rcrpix1


def savefig_small(pngim):
    from PIL import Image
    im = Image.open(pngim)
    im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
    im2.save(pngim)


def spectrum_plot(ax, spx, spy, xrange, yrange, spx_func="NA", spy_func="NA",
                  colors=["k", "b", "r", "g"], linew=[1.], lines=["-"], labels=[""],
                  colors_func=["k", "b", "r", "g"], linew_func=[1.], lines_func=["--"],
                  labels_func=[""], yshift=[0], yfactor=[1.], xlshift=[0], xvshift=[0],
                  xticks_val="NA", yticks_val="NA", xticks_label="NA", yticks_label="NA",
                  xaxis_label="NA", yaxis_label="NA", grid_flag=False, legend_flag=False,
                  legend_loc=0):
    """``warp/plotframes.py:spectrum_plot``."""
    import scipy.constants
    Nspec = len(spx)
    colors = colors * math.ceil(Nspec / len(colors)) if len(colors) < Nspec else colors
    linew = linew * math.ceil(Nspec / len(linew)) if len(linew) < Nspec else linew
    lines = lines * math.ceil(Nspec / len(lines)) if len(lines) < Nspec else lines
    labels = labels * math.ceil(Nspec / len(labels)) if len(labels) < Nspec else labels
    yshift = yshift + [0] * (Nspec - len(yshift)) if len(yshift) < Nspec else yshift
    yfactor = yfactor + [1] * (Nspec - len(yfactor)) if len(yfactor) < Nspec else yfactor
    xlshift = xlshift + [0] * (Nspec - len(xlshift)) if len(xlshift) < Nspec else xlshift
    xvshift = xvshift + [0] * (Nspec - len(xvshift)) if len(xvshift) < Nspec else xvshift
    for i in range(Nspec):
        spx_plot = (spx[i] + xlshift[i]) * (1. + xvshift[i] / (scipy.constants.c * 1.e-3))
        spy_plot = spy[i] * yfactor[i] + yshift[i]
        ax.step(spx_plot, spy_plot, where="mid", c=colors[i], label=labels[i], lw=linew[i],
                ls=lines[i])
    if xticks_val != "NA":
        ax.set_xticks(xticks_val)
        if xticks_label != "NA":
            ax.set_xticklabels(xticks_label)
    if yticks_val != "NA":
        ax.set_yticks(yticks_val)
        if yticks_label != "NA":
            ax.set_yticklabels(yticks_label)
    if xaxis_label != "NA":
        ax.set_xlabel(xaxis_label)
    if yaxis_label != "NA":
        ax.set_ylabel(yaxis_label)
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    if grid_flag:
        ax.grid()
    if legend_flag:
        if legend_loc != "out":
            ax.legend(loc=legend_loc)
        else:
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")


def plot_all_frames_norm(splist, outputf, apnum, fnum, fsr):
    """``warp/plotframes.py:plot_all_frames_norm``."""
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure(figsize=(25, 6))
    ax = plt.axes([0.04, 0.1, 0.88, 0.8])
    fnum = len(splist)
    pp = PdfPages(outputf)
    for i in range(len(apnum)):
        leftfnum = fnum
        repeatnum = 0
        while leftfnum > 0:
            spxlist, spylist, frameno = [], [], []
            for k in range(min(leftfnum, maxfnum)):
                spx, spy, _, _, _ = openspecfits(splist[repeatnum * maxfnum + k][i])
                spxlist.append(spx)
                spylist.append(spy)
                frameno.append("frame_NO%d" % (k + 1 + repeatnum * maxfnum))
            spectrum_plot(ax, spxlist, spylist, fsr[apnum[i]], [0, 2.], colors=colornames,
                          linew=[0.5], labels=frameno, xticks_label="Wavelength ($\\AA$)",
                          yticks_label="Normalized flux", legend_flag=True, legend_loc="out")
            plt.title("m=%d" % apnum[i])
            _savefig(plt, pp, format="pdf")
            plt.cla()
            repeatnum += 1
            leftfnum -= maxfnum
    pp.close()
    plt.close()


def plot_all_frames_flux(splist, outputf, apnum, fnum, fsr):
    """``warp/plotframes.py:plot_all_frames_flux``."""
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure(figsize=(25, 6))
    ax = plt.axes([0.04, 0.1, 0.88, 0.8])
    pp = PdfPages(outputf)
    for i in range(len(apnum)):
        leftfnum = fnum
        repeatnum = 0
        while leftfnum > 0:
            spxlist, spylist, frameno, spymed = [], [], [], []
            for k in range(min(leftfnum, maxfnum)):
                spx, spy, _, _, _ = openspecfits(splist[repeatnum * maxfnum + k][i])
                spxlist.append(spx)
                spylist.append(spy)
                spymed.append(np.median(spy))
                frameno.append("frame_NO%d" % (k + 1 + repeatnum * maxfnum))
            spectrum_plot(ax, spxlist, spylist, fsr[apnum[i]], [0, max(spymed) * 2.],
                          colors=colornames, linew=[0.5], labels=frameno,
                          xticks_label="Wavelength ($\\AA$)", yticks_label="Flux",
                          legend_flag=True, legend_loc="out")
            plt.title("m=%d" % apnum[i])
            _savefig(plt, pp, format="pdf")
            plt.cla()
            repeatnum += 1
            leftfnum -= maxfnum
    pp.close()
    plt.close()


def plot_all_frames_flux_BG(splist, bgspec, outputf, apnum, fnum, fsr):
    """``warp/plotframes.py:plot_all_frames_flux_BG`` (``Background
    Subtraction``): the background spectra (``_bgcw``) in a lower panel that
    is never cleared, the flux spectra above."""
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure(figsize=(25, 7))
    ax1 = plt.axes([0.04, 0.1, 0.88, 0.15])
    ax2 = plt.axes([0.04, 0.3, 0.88, 0.65])
    pp = PdfPages(outputf)
    for i in range(len(apnum)):
        leftfnum = fnum
        repeatnum = 0
        while leftfnum > 0:
            spxlist, spylist, frameno, spymed = [], [], [], []
            bgxlist, bgylist, bgymed = [], [], []
            for k in range(min(leftfnum, maxfnum)):
                spx, spy, _, _, _ = openspecfits(splist[repeatnum * maxfnum + k][i])
                bgx, bgy, _, _, _ = openspecfits(bgspec[repeatnum * maxfnum + k][i])
                spxlist.append(spx)
                spylist.append(spy)
                bgxlist.append(bgx)
                bgylist.append(bgy)
                spymed.append(np.median(spy))
                bgymed.append(np.median(bgy))
                frameno.append(f"frame_NO{k + 1 + repeatnum * maxfnum:d}")
            spectrum_plot(ax1, bgxlist, bgylist, fsr[apnum[i]],
                          [min(bgymed) - 5. * (max(bgymed) - min(bgymed)),
                           max(bgymed) + 5. * (max(bgymed) - min(bgymed))],
                          colors=colornames, linew=[0.5], labels=frameno,
                          xticks_label="Wavelength ($\\AA$)", yticks_label="Flux",
                          legend_flag=False)
            spectrum_plot(ax2, spxlist, spylist, fsr[apnum[i]], [0, max(spymed) * 2.],
                          colors=colornames, linew=[0.5], labels=frameno,
                          xticks_label="Wavelength ($\\AA$)", yticks_label="Flux",
                          legend_flag=True, legend_loc="out")
            plt.title(f"m={apnum[i]:d}")
            _savefig(plt, pp, format="pdf")
            plt.cla()
            repeatnum += 1
            leftfnum -= maxfnum
    pp.close()
    plt.close()


def plot_combined_norm(splist, outputf, apnum, fsr):
    """``warp/plotframes.py:plot_combined_norm``."""
    plt = _plt()
    plt.figure(figsize=(18, 4))
    for i in range(len(apnum)):
        plt.xlim(fsr[apnum[i]][0], fsr[apnum[i]][1])
        spx, spy, rcrval1, rcdelt1, rcrpix1 = openspecfits(splist[i])
        plt.step(spx, spy, color="k", lw=3., where="mid")
        plt.ylim(0, np.median(spy) * 1.5)
        plt.xlabel("Wavelength")
        plt.ylabel("Normalized flux")
        plt.title("m=%d" % apnum[i])
        _savefig_small(plt, outputf[i], format="png")
        plt.clf()
    plt.close()


def plot_2dimages_mask(maskname, outputfig, title):
    """``warp/plotframes.py:plot_2dimages_mask`` (``title``: WARP's file name;
    ``maskname`` the mask file or its pixel array)."""
    plt = _plt()
    import matplotlib.cm as cm
    pixdata = _image_data(maskname)
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.imshow(pixdata, cmap=cm.gray, vmin=0, vmax=1, origin='lower', interpolation="none")
    plt.title(title)
    _savefig_small(plt, outputfig, dpi=400)
    plt.clf()
    plt.close()


def plot_2dimages(inputimage, outputfig, title):
    """``warp/plotframes.py:plot_2dimages`` (``title``: WARP's file name;
    ``inputimage`` the FITS file or its pixel array)."""
    plt = _plt()
    import matplotlib.cm as cm
    pixdata = _image_data(inputimage)
    blarray = pixdata > 10.
    thres = np.median(pixdata[blarray])
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.imshow(pixdata, cmap=cm.gray, vmin=-3. * thres, vmax=3. * thres, origin='lower',
               interpolation="none")
    plt.title(title)
    _savefig_small(plt, outputfig)
    plt.clf()
    plt.close()


def snr_plots(lams_sn, snr_val, spfiles, aplength, outputpng):
    """``warp/plotframes.py:snr_plots``."""
    plt = _plt()
    from matplotlib.ticker import ScalarFormatter
    fig, ax1 = plt.subplots(figsize=(6, 4))
    twocolors_1 = ["orangered", "r"]
    twocolors_2 = ["dodgerblue", "b"]
    ax1.set_xlabel(r"Wavelength in air ($\AA$)")
    ax1.set_ylabel("Signal-to-noise ratio")
    ax1.set_ylim(0., np.max(snr_val) * 1.2)
    ax2 = ax1.twinx()
    spmax = []
    lammin = []
    lammax = []
    for j in range(aplength):
        spx, spy, a, b, c = openspecfits(spfiles[j])
        ax2.plot(spx, spy, color=twocolors_1[j % 2], alpha=0.5)
        spmax.append(np.median(spy))
        lammin.append(np.min(spx))
        lammax.append(np.max(spx))
    ax2.set_ylabel("Flux (Arbitrary Unit)")
    ax2.set_ylim(0., np.max(spmax) * 3.)
    ax2.set_xlim(np.min(lammin), np.max(lammax))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    for j in range(aplength):
        ax1.plot(lams_sn[j], snr_val[j], color=twocolors_2[j % 2], lw=3.)
        ax1.scatter(lams_sn[j], snr_val[j], color=twocolors_2[j % 2], s=30)
    _savefig(plt, outputpng)
    plt.close()


def peak_count_fwhm(splist, outputf, apnum, fnum, fwhm="INDEF", mode="WIDE"):
    """``warp/plotframes.py:peak_count_fwhm``."""
    plt = _plt()
    averageorder = {"WIDE": 45, "HIRES-J": 142, "HIRES-Y": 171}
    averagecount = []
    frameno = [i + 1 for i in range(fnum)]
    if averageorder[mode] in apnum:
        j = apnum.index(averageorder[mode])
    else:
        j = np.argmin(np.absolute(np.array(apnum) - averageorder[mode]))
    selectedorder = apnum[j]
    for i in range(fnum):
        spx, spy, _, _, _ = openspecfits(splist[i][j])
        averagecount.append(np.average(spy))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Frame number")
    ax1.set_ylabel("Average count at m=%d" % selectedorder)
    ax1.set_xticks(frameno)
    ax1.set_xticklabels(["No.%d" % frameno[i] for i in range(fnum)])
    ax1.set_ylim(0., np.max(averagecount) * 1.2)
    ax1.set_xlim(min(frameno) - 0.5, max(frameno) + 0.5)
    ax1.scatter(frameno, averagecount, color="r", s=30, label="Count")
    if not (isinstance(fwhm, str) and fwhm == "INDEF"):
        ax2 = ax1.twinx()
        ax2.scatter(frameno, fwhm, color="b", s=30, label="FWHM")
        ax2.set_ylabel("FWHM (pix)")
        if fwhm.size > 1:
            ylimfactor = max(np.nanmax(np.absolute(fwhm - np.nanmedian(fwhm))) * 1.2, 2.)
            ax2.set_ylim(np.nanmedian(fwhm) - ylimfactor, np.nanmedian(fwhm) + ylimfactor)
        else:
            ax2.set_ylim(fwhm[0] - 2., fwhm[0] + 2.)
        ax2.set_xlim(np.min(frameno) - 0.5, np.max(frameno) + 0.5)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    _savefig_small(plt, outputf)
    plt.clf()
    plt.close()


def aperture_plot(datfile, imgfile, lowlim, upplim, bgregion, flagskysub):
    """``warp/plotframes.py:aperture_plot``."""
    plt = _plt()
    rl = Path(datfile).read_text().splitlines(keepends=True)
    med_plt_x = np.array([float(rl[i].split()[0]) for i in range(len(rl))])
    med_plt_y = np.array([float(rl[i].split()[1]) for i in range(len(rl))])
    xshift_fix = (lowlim + upplim) / 2.
    plt.figure()
    plt.scatter(med_plt_x, med_plt_y)
    plt.plot([xshift_fix, xshift_fix], [-2, 2], "g", label="Center")
    plt.plot([lowlim, lowlim], [-2, 2], "g--", label="Aperture")
    plt.plot([upplim, upplim], [-2, 2], "g--")
    if flagskysub and bgregion != "INDEF":
        regions = bgregion.split(",")
        bg_pixs = []
        for j in range(len(regions)):
            plt.plot([float(regions[j].split(":")[0]), float(regions[j].split(":")[1])],
                     [0.5, 0.5], "r")
            bg_pixs.append(float(regions[j].split(":")[0]))
            bg_pixs.append(float(regions[j].split(":")[1]))
        avg_bg = np.average(bg_pixs)
        plt.text(avg_bg, 0.55, "background", color="r", va="bottom", ha="center")
    plt.ylim(-2.0, 2.0)
    plt.xlabel("Distance from defined aperture center (pix)")
    plt.ylabel("Normalized count")
    plt.legend()
    plt.grid()
    _savefig_small(plt, imgfile, format="png")
    plt.clf()
    plt.close()


def cosmicRay2dImages(crdata, crfig, title, apset, bpfdata, xlim1=-30, xlim2=30):
    """``warp/plotframes.py:cosmicRay2dImages`` (the CR mask and static mask
    arrays given; ``apset`` the multihole aperture file, its
    :func:`cr_plot_geometry`, or a :class:`decanter.calib.aperture.ApertureSet`)."""
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    if isinstance(apset, tuple):
        maskap, slitcoord, array_length, echelle_orders = apset
    elif isinstance(apset, (str, Path)):
        maskap, slitcoord, array_length, echelle_orders = cr_plot_geometry(apset, xlim1, xlim2)
    else:
        maskap = apset.apmask_array(low_lim=xlim1, upp_lim=xlim2)
        slitcoord = apset.slitcoord_array(low_lim=xlim1, upp_lim=xlim2)
        array_length, echelle_orders = apset.array_length, list(apset.echelle_orders)
    apy = np.arange(array_length) + 1.
    Xarray, Yarray = np.meshgrid(apy - 1, apy - 1)
    reqnotbpf = bpfdata == 0
    reqbpf = bpfdata != 0
    reqcr = (crdata != 0) & reqnotbpf
    pp = PdfPages(crfig)
    plt.figure(figsize=(10, 10))
    plt.scatter(Xarray[reqcr], Yarray[reqcr], s=5., marker=".", color="r",
                label="CR mask ({} pix)".format(np.sum(reqcr)))
    plt.scatter(Xarray[reqbpf], Yarray[reqbpf], s=5., marker=".", color="b",
                label="bad pixel mask ({} pix)".format(np.sum(reqbpf)))
    plt.legend()
    plt.title(title)
    plt.xlim(0., array_length)
    plt.ylim(0., array_length)
    _savefig(plt, pp, format="pdf")
    plt.clf()
    plt.figure()
    plt.hist(slitcoord[reqcr], bins=20, histtype="bar")
    plt.xlabel("Slit coordinate (pix)")
    plt.ylabel("Cosmic ray pixels")
    plt.xlim(xlim1, xlim2)
    plt.title(title)
    _savefig(plt, pp, format="pdf")
    plt.clf()
    plt.hist(maskap[reqcr], bins=20, histtype="bar")
    plt.xlabel("Echelle order m")
    plt.ylabel("Cosmic ray pixels")
    plt.xlim(min(echelle_orders) - 1, max(echelle_orders) + 1)
    plt.title(title)
    _savefig(plt, pp, format="pdf")
    plt.clf()
    pp.close()
    plt.close("all")


__all__ = [n for n in dir(sys.modules[__name__]) if not n.startswith("_")]
