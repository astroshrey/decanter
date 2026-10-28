"""WARP's post-FSR products, IRAF header editing and the fast exact kernels.

The data-dependent tests feed WARP's own products from
``reductions/TOI3486_decanterref`` (a WARP ``-s`` reduction) into the ports
and require WARP's bytes back; they skip when that reduction is absent.
"""
from __future__ import annotations

import re
import numpy as np
import pytest
from astropy.io import fits

from decanter._localpaths import WINERED_ROOT
from decanter.utils import fpemu
from decanter.utils.iraf_header import IrafHeader

REF = WINERED_ROOT / "reductions" / "TOI3486_decanterref"
needs_ref = pytest.mark.skipif(not (REF / "TOI-3486_NO1").is_dir(),
                               reason="TOI3486_decanterref not available")


# --- IRAF header editing ------------------------------------------------------

def test_iraf_header_edits_in_place_and_appends():
    h = IrafHeader.from_images([
        "WCSDIM  =                    1".ljust(80),
        "CRVAL1  =      12522.264514188".ljust(80),
        "BANDID1 = 'spectrum: background none, weights none, clean no'".ljust(80),
    ])
    h.put_num("CRVAL1", "1.")
    h.put_str("REFSPEC1", "comp.0140")
    h.delete("WCSDIM")
    assert h.keys() == ["CRVAL1", "BANDID1", "REFSPEC1"]
    assert h.text("CRVAL1") == "1."
    assert h["BANDID1"] == "spectrum: background none, weights none, clean no"
    assert h.cards[2].image == "REFSPEC1= 'comp.0140'".ljust(80)


def test_date_value_replaced_with_comment_kept():
    raw = "DATE    = '2026-06-24'         /  Creation UTC (CCCC-MM-DD) date of FITS header "
    h = IrafHeader.from_images([raw])
    h.set_string_keep_comment("DATE", "2026-10-20T06:11:29")
    # what WARP's files carry (fxf impstr): the slash moves right by one blank
    assert h.cards[0].image == ("DATE    = '2026-10-20T06:11:29' /  Creation UTC (CCCC-MM-DD) "
                                "date of FITS header")


# --- exact kernels --------------------------------------------------------------

def test_fma64_scalar_path_is_the_array_path():
    rng = np.random.default_rng(3)
    for _ in range(20000):
        a, b, c = (float(rng.standard_normal() * 10 ** rng.uniform(-6, 6)) for _ in range(3))
        x = fpemu.fma64(np.asarray(a), np.asarray(b), np.asarray(c))
        assert fpemu.fma64(a, b, c) == x


def test_scalar_fma32_and_chain_match_batched_kernels():
    from decanter.utils import icfit_exact as icx
    from decanter.utils import sfit_exact as sf
    rng = np.random.default_rng(4)
    a = rng.standard_normal(5000).astype(np.float32)
    b = rng.standard_normal(5000).astype(np.float32)
    c = rng.standard_normal(5000).astype(np.float32)
    ref = fpemu.fma32(a, b, c)
    got = np.array([sf.fma32_s(float(x), float(y), float(z)) for x, y, z in zip(a, b, c)],
                   dtype=np.float32)
    assert np.array_equal(ref.view(np.int32), got.view(np.int32))
    assert np.array_equal(sf._fma32_lean(a, b, c).view(np.int32), ref.view(np.int32))
    chain = icx.fma_chain(a[:, None], b[:, None])[0]
    assert sf.fma_chain_1(a, b) == chain


def test_banded_cholesky_single_system_matches_batched():
    from decanter.utils import icfit_exact as icx
    from decanter.utils import sfit_exact as sf
    rng = np.random.default_rng(5)
    for nr in (4, 9, 18):
        nb = 4
        full = rng.standard_normal((nr + 8, nr)).astype(np.float64)
        spd = np.einsum("ki,kj->ij", full, full)
        band = np.zeros((nb, nr), dtype=np.float32)
        for ii in range(nb):
            for r in range(nr - ii):
                band[ii, r] = spd[r, r + ii]
        vec = rng.standard_normal(nr).astype(np.float32)
        mf_b, _ = icx.chofac(band[:, :, None])
        c_b = icx.choslv(mf_b, vec[:, None])[:, 0]
        mf_1, _ = sf.chofac_1(band)
        c_1 = sf.choslv_1(mf_1, vec)
        assert np.array_equal(mf_b[:, :, 0].view(np.int32), mf_1.view(np.int32))
        assert np.array_equal(c_b.view(np.int32), c_1.view(np.int32))


# --- WARP products from WARP's inputs ----------------------------------------------

def _load(path):
    from decanter.wavelength.iraf_onedspec import Spectrum1D
    with fits.open(path) as h:
        return Spectrum1D(np.asarray(h[0].data, np.float32), h[0].header.copy())


_SKIP = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "EXTEND", "IRAF-TLM", "DATE"}


def _same(spec, warp_path):
    w = _load(warp_path)
    a = np.asarray(spec.data, np.float32)
    b = np.asarray(w.data, np.float32)
    ia = [c.image for c in spec.header.cards if c.keyword not in _SKIP and c.image.strip()]
    ib = [c.image for c in w.header.cards if c.keyword not in _SKIP and c.image.strip()]
    return a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32)) and ia == ib


@needs_ref
@pytest.mark.parametrize("m, cut", [(131, "1.05"), (147, "1.30"), (154, "1.05")])
def test_continuum_sarith_vac2air_reproduce_warp(m, cut):
    from decanter.warptree_products import spectrum_products
    d = REF / "TOI-3486_NO1" / "onedspec"
    vac = _load(d / f"VAC_flux/fsr{cut}/TOI-3486_NO1_m{m}_fsr{cut}_VAC.fits")
    prods = spectrum_products(vac, m)
    for key, sub in (("VAC_norm", "VAC_norm"), ("VAC_cont", "VAC_cont"), ("AIR", "AIR_flux"),
                     ("AIR_norm", "AIR_norm"), ("AIR_cont", "AIR_cont")):
        f = d / f"{sub}/fsr{cut}/TOI-3486_NO1_m{m}_fsr{cut}_{key}.fits"
        assert _same(prods[key], f), key


@needs_ref
def test_scombine_sum_and_norm_comb_reproduce_warp():
    from decanter.wavelength.iraf_scombine import scombine_average
    s = REF / "TOI-3486_sum"
    vacs = [_load(REF / f"TOI-3486_NO{i}/onedspec/VAC_flux/fsr1.05/TOI-3486_NO{i}_m140_fsr1.05_VAC.fits")
            for i in (1, 2)]
    assert _same(scombine_average(vacs), s / "VAC_flux/fsr1.05/TOI-3486_sum_m140_fsr1.05_VAC.fits")
    norms = sorted((s / "VAC_norm/fsr1.30").glob("TOI-3486_sum_m*_fsr1.30_VAC_norm.fits"),
                   key=lambda p: int(re.search(r"_m(\d+)_", p.name).group(1)))
    assert _same(scombine_average([_load(p) for p in norms]),
                 s / "VAC_norm/fsr1.30/TOI-3486_sum_fsr1.30_VAC_norm_comb.fits")


@needs_ref
def test_iraf_aperture_databases_reproduce_warp():
    from decanter.warptree_products import iraf_apdb_text
    db = REF / "calibration_data" / "database"
    stamp = re.compile(r"^# \w{3} \d\d:\d\d:\d\d \d\d-\w{3}-\d{4}$")

    def body(text):
        return [ln for ln in text.splitlines() if not stamp.match(ln)]
    src = (db / "apTOI-3486_NO2_sscfm_m140trans").read_text()
    assert body(iraf_apdb_text(src, "TOI-3486_skyNO2_fm_m140trans")) == body(
        (db / "apTOI-3486_skyNO2_fm_m140trans").read_text())
    assert body(iraf_apdb_text(src.replace("TOI-3486_NO2_sscfm_m140trans", "x"), "last")) != []


@needs_ref
def test_snr_estimate_reproduces_warp():
    from decanter.warptree_products import snr_estimate
    fs = [REF / f"TOI-3486_NO{i}/onedspec/VAC_flux/fsr1.05/TOI-3486_NO{i}_m140_fsr1.05_VAC.fits"
          for i in (1, 2)]
    txt, _, _ = snr_estimate([_load(f) for f in fs], [f.name[:-5] for f in fs])
    want = (REF / "SNR_dat/fsr1.05/SNratio_m140_fsr1.05.dat").read_text()
    drop = lambda t: [ln for ln in t.splitlines() if not ln.startswith("Time:")]  # noqa: E731
    assert drop(txt) == drop(want)


@needs_ref
def test_aperture_plot_is_warps_png(tmp_path):
    from decanter.warptree_plots import aperture_plot
    cs = np.load(REF / "reduction_log/centersearch_log.npz")
    xs, gs = cs["psfCenter"][0], cs["psfWidth"][0]
    lo = -1. * np.nanmedian(gs) + np.nanmedian(xs)
    hi = np.nanmedian(gs) + np.nanmedian(xs)
    sp = REF / "TOI-3486_NO1/images/spatial_profile"
    out = tmp_path / "a.png"
    aperture_plot(sp / "TOI-3486_NO1_sscfm_m140trans.dat", out, lo, hi, "INDEF", False)
    assert out.read_bytes() == (sp / "TOI-3486_NO1_sscfm_m140trans.png").read_bytes()


# --- figures ------------------------------------------------------------------------

def test_figures_from_arrays_and_single_render_are_warps_bytes(tmp_path):
    """plot_2dimages / plot_2dimages_mask drawn from the in-memory pixels give
    the bytes of the same figure drawn from the FITS file WARP reads, and the
    figures saved without pyplot's second render are pyplot.savefig's bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decanter.warptree_io import write_astropy_image, write_iraf_fits
    from decanter.warptree_plots import plot_2dimages, plot_2dimages_mask
    from decanter.utils.iraf_header import IrafHeader
    rng = np.random.default_rng(11)
    img = (rng.standard_normal((96, 96)) * 30 + 40).astype(np.float32)
    write_iraf_fits(tmp_path / "img.fits", img, IrafHeader())
    plot_2dimages(tmp_path / "img.fits", tmp_path / "a.png", "img.fits")
    plot_2dimages(img, tmp_path / "b.png", "img.fits")
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    mask = (rng.random((96, 96)) < 0.05).astype(np.int16)
    write_astropy_image(tmp_path / "mask.fits", mask)
    plot_2dimages_mask(tmp_path / "mask.fits", tmp_path / "c.png", "mask.fits")
    plot_2dimages_mask(mask, tmp_path / "d.png", "mask.fits")
    assert (tmp_path / "c.png").read_bytes() == (tmp_path / "d.png").read_bytes()

    def draw():
        plt.figure()
        plt.scatter(np.arange(10.), np.arange(10.) ** 0.5)
        plt.plot([1, 1], [-2, 2], "g--", label="Aperture")
        plt.legend()
        plt.grid()

    draw()
    plt.savefig(tmp_path / "e.png", format="png")
    plt.close()
    draw()
    plt.gcf().savefig(tmp_path / "f.png", format="png")
    plt.close()
    assert (tmp_path / "e.png").read_bytes() == (tmp_path / "f.png").read_bytes()


def test_quantised_png_from_the_rendered_buffer_is_savefig_small(tmp_path):
    """WARP's savefig + savefig_small (PNG written, re-read, quantised) and
    the buffer path give the same file, at the default dpi and at 400."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decanter.warptree_plots import _savefig_small, savefig_small
    rng = np.random.default_rng(12)
    img = rng.random((60, 60)).astype(np.float32)
    for k, kw in enumerate(({}, {"dpi": 400}, {"format": "png"})):
        for fast in (False, True):
            fig = plt.figure(facecolor="white", figsize=(3, 3))
            fig.add_subplot(111).imshow(img, cmap="gray", origin="lower", interpolation="none")
            plt.title("x.fits")
            out = tmp_path / f"{k}_{int(fast)}.png"
            if fast:
                _savefig_small(plt, out, **kw)
            else:
                plt.savefig(out, **kw)
                savefig_small(out)
            plt.close()
        assert (tmp_path / f"{k}_0.png").read_bytes() == (tmp_path / f"{k}_1.png").read_bytes()


# --- dopcor: DOPCORnn before the cards smw_saveim appends (t_dopcor.x) ------------

#: IRAF's header of a HIRES-J FSR cut (TOI3486_decanterref NO1 m134, cut range
#: 1.67, WARP's cut_1dspec) from its WCS on, and of its ``vac2air_spec``
#: output (pyraf, scripts/iraf_oracle/pyraf_vac2air_check.py): LTM1_1 > 1 puts
#: the last physical pixel before NAXIS1 = 3836, so smw_saveim appends NP2 --
#: after the DOPCOR01 card t_dopcor.x:232 wrote (its smw_saveim is at :268).
_VAC_TAIL = [
    'WCSDIM  =                    1',
    'LTM1_1  =     1.00003611698829',
    "WAT0_001= 'system=equispec'",
    "WAT1_001= 'wtype=linear label=Wavelength units=angstroms'",
    "DCLOG1  = 'Transform'",
    'DC-FLAG =                    0',
    "CTYPE1  = 'LINEAR  '",
    'CRVAL1  =      13083.775393644',
    'CRPIX1  =     1.00007223267219',
    'CDELT1  =   0.0417961105704308',
    'CD1_1   =   0.0417961105704308',
    "APNUM1  = '134 134 166.15 175.00'",
    'NCOMBINE=                    1',
    "DCLOG2  = 'REFSPEC1 = comp_HIRES-J100_20260625_fm_ecall.0134'",
    "AIRORVAC= 'vac     '",
    "BANDID1 = 'spectrum: background none, weights none, clean no'",
    'LTV1    =  3.61156838977195E-5',
]
_AIR_TAIL = _VAC_TAIL[:7] + [
    'CRVAL1  =      13080.198331777',
    'CRPIX1  =     1.00007223267219',
    'CDELT1  =   0.0417846836490101',
    'CD1_1   =   0.0417846836490101',
] + _VAC_TAIL[11:14] + [
    "AIRORVAC= 'air     '",
    "BANDID1 = 'spectrum: background none, weights none, clean no'",
    'LTV1    =  3.61156838977195E-5',
    "DOPCOR01= ' 81.9735 all'",
    'NP2     =                 3835',
    'VAC2AIR =             -81.9735',
]


def test_dopcor_card_precedes_the_np2_smw_saveim_appends():
    from decanter.wavelength.iraf_onedspec import Spectrum1D, vac2air
    h = IrafHeader.from_images([c.ljust(80) for c in _VAC_TAIL])
    rng = np.random.default_rng(0)
    vac = Spectrum1D(rng.normal(1000, 30, 3836).astype(np.float32), h)
    air = vac2air(vac)
    assert [c.rstrip() for c in air.header.images()] == _AIR_TAIL
    # the data: the SN = NP2 pixels, then the last one repeated (t_dopcor.x:189-203)
    assert np.array_equal(air.data[:3835], vac.data[:3835])
    assert air.data[3835] == vac.data[3834]
