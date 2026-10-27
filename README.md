# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

Fast, pure-Python reduction of WINERED near-infrared echelle spectra. A WARP
([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) clone whose output is
bit-identical to WARP's in all three modes (HIRES-Y, HIRES-J, and WIDE), plus an absolute
wavelength calibration that anchors each order to telluric absorption and OH
airglow.

## Install

```bash
pip install -e .
pip install -e '.[wavecal]'   # for the physical wavelength calibration
pip install -e '.[mlx]'       # optional: Apple-GPU rectification (same output)
```

## Use

```python
import decanter

calib = decanter.Calibration.from_dir("path/to/calibration_set")
r = decanter.reduce("obj.fits", calib, sky="sky.fits")   # nod-subtracted (A−B)

spec = r.obj[(1.30, 163)]        # order 163 at FSR cut 1.30
spec.wavelength, spec.flux       # vacuum-Å grid, flux
```

Omit `sky=` to reduce a single nod position on its own: `decanter.reduce("obj.fits", calib)`.
Without a sky frame there is no nod subtraction, so the background emission (OH
airglow lines), dark current, bias, and stray light are **retained** in the spectrum.
Pass `subtract_background=True` to estimate and remove that background from the
slit during extraction (suppressing the OH lines).

For a time series, `reduce_many` applies the WARP cross-frame wavelength shift.
Passing `wavecal_config` layers the telluric/OH calibration on top of it;
omitting it leaves the WARP-only behaviour unchanged. Pass `align=False` to
skip the relative shift and let the wavecal register the frames on its own —
see [Cross-frame alignment](#cross-frame-alignment).

```python
series = decanter.reduce_many(pairs, calib)                # WARP alignment only

series = decanter.reduce_many(
    pairs, calib,
    wavecal_config=decanter.WavecalConfig(),               # hybrid, mode="auto"
    workdir="output/corrected",
)

series.shifts                     # WARP relative shifts, Å
series.wavecal_solution.velocity  # physical shifts, km/s per (frame, order)
```

The calibration is applied by rescaling each order's `CRVAL1`/`CDELT1`, so the
written spectra are calibrated without the flux being resampled.

### Cross-frame alignment

`--alignment none` is the recommended setting when the physical wavecal is on:

```bash
python reduce.py \
  --frames TOI2109/ --listfile TOI2109.txt \
  --calib TOI2109/calibration_LCO25b_setting2_HIRES-Y100 \
  --out output/TOI2109 \
  --alignment none --wavecal auto --diagnostic-pdf
```

`--alignment warp` (the CLI default, for WARP-compatible output) and
`--alignment atmospheric` (a pooled telluric/OH pre-registration before the
physical solve) remain available. The atmospheric pass exists for cases where
the drift exceeds what one search window can cover; with the default
`shift_search_kms = 25` a single pass covers WINERED's usual range.

The four wavecal choices are `oh_static`, `oh_refit`, `hybrid_static`, and
`hybrid_refit`, with `auto` picking `hybrid_refit` for HIRES-Y/J and
`hybrid_static` for WIDE.

`python reduce.py --help` reduces a whole night from raw frames and calibration
directory in one command. `decanter-hrccs examples/hrccs/wasp69b.toml` runs the
downstream cross-correlation pipeline on a calibrated output directory.

## Modes

`decanter.reduce` (and `reduce_many`, `reduce_to_warp_tree`) take one switch,
`mode=`:

| mode | what runs |
|---|---|
| `"warp"` (default) | the WARP clone, vectorised: cosmic-ray, rectification, extraction and dispersion kernels restructured, their calibration-only work cached, only the part of each rectified order the extraction reads computed, and the rectification on the Apple GPU when [MLX](https://github.com/ml-explore/mlx) is installed |
| `"warp-reference"` | the scalar, stage-by-stage exact port of WARP the default is checked against (about 3-5x slower) |

Both WARP-clone implementations follow the same recipe (stages, apertures,
cross-frame shift, dispersion grid, FSR cuts) with the same arithmetic. Their
output is bit-identical: by construction on the CPU path
(`DECANTER_FAST_BACKEND=numpy`), and on the GPU path apart from float32
subnormals in rectified-strip pixels the extraction never reads (visible only
with `save_intermediates=True`). Both are bit-identical to WARP except WARP's
documented uninitialised-memory tail pixels. `save_intermediates=True`,
`extract="optimal"` and `subtract_background=True` work in the default mode
(they rectify whole strips). See [docs/VECTORISED.md](docs/VECTORISED.md) for the
implementation, what is verified and timings, and
[docs/BITEXACT.md](docs/BITEXACT.md) for parity with WARP.

## Reduce a night into a WARP tree

```python
tree = decanter.reduce_to_warp_tree("lists/TOI836.txt", "science_data/2025_02_15",
                                    "calibration_data/calibration_LCO25a_setting4_HIRES-Y100",
                                    "reductions/TOI836", paramfile="parameter_files/nominal_params.txt",
                                    workers=6)
```

writes every file `Warp_sci.py` writes (`--noreport`): per frame the
`onedspec/{VAC,AIR}_{flux,norm,cont}` and `sky_emission/{VAC,AIR}` spectra,
`images/` (spatial profiles, 2-D and mask figures), the intermediates WARP
always keeps; the `<OBJECT>_sum` spectra, `SNR_dat/`, `spectra_image/`,
`rawdata_image/`, `calibration_data/` (with the per-frame aperture databases)
and `reduction_log/` (logs, `status.txt`), and in the tree root every other
entry of the calibration set (every file and directory but dot files: WARP
copies them all into its work directory and moves only its own calibration
files into `calibration_data/`, e.g. the empty `sys` of
`calibration_LCO25a_setting5_HIRES-Y100`). It is a drop-in for a WARP tree.
What is identical, on every verification night and option run
(docs/BITEXACT.md "WARP tree products", "File bytes"):

- FITS files: the data bit for bit, every header card's text in order, and
  the file bytes -- WARP's IRAF files with IRAF's layout (blank cards before
  `END`, zero padding), its astropy files as astropy writes them -- except,
  intentionally: the time-stamp values (`IRAF-TLM`, `DATE`, the run time in
  `FIXPIX`) and decanter's provenance card `DECANTER = 'decanter <version>
  mode=<mode>'`, written where WARP's file has its first blank card (so
  sizes and offsets are WARP's; the 2-D intermediates, strips and raw copies
  have none, and a header with no blank card gets none);
- aperture databases, logs, `status.txt`, `.dat` files: the text, except
  the time-stamp values (the run's write and run times); `.npz` logs: every
  array bit for bit; PNG figures: byte for byte; PDF figures: byte for byte
  except the `/CreationDate` value. Byte for byte as well (line terminators
  and final newlines, the npz zip containers member by member), wherever
  checked with the strict `compare_full_tree.py --bytes` (2026-10-21: the
  three `-s` references and four option runs).

What differs: WARP's uninitialised s13 tail pixel (the last pixel of an
FSR cut whose REAL bounds give one pixel fewer than the double bounds; WARP
writes heap memory, decanter the previous value; which cuts, per
calibration set: docs/BITEXACT.md "Accepted s13 tail pixels"), and what
WARP's own Python derives from all of a spectrum's pixels: through
`np.median` the SNR estimate of such an (order, cut) in a 2..28-frame night
(`SNratio_fsr*.png`, possibly the `.dat`) and the per-page y-limits of
`spectra_image/<OBJ>_VACflux.pdf` when the case is in the first cut (the
pixel itself is not drawn), and through `np.average`
`reduction_log/count_and_fwhm.png` if a first-cut case falls in the order it
averages (171 / 142 / 45; no calibration set on disk has one); decanter adds
`reduction_log/decanter_provenance.json`; it does not write WARP's `-s` IRAF
task logs `reduction_log/{logfile,null}`, `slit_viewer/` (WARP `-v`) or the
PDF report (`--noreport`). A tree locked to a WARP reduction
(`apertures=<WARP root>`) takes WARP's logged aperture window from that root's
`reduction_log/aperture_log.npz` (without it, `aperture_log.npz` holds the
databases' `%.2f` values).
Options: `intermediates=False` skips the always-kept intermediates (six
2048x2048 images per frame in `4-OBJ_mask`), `save=True` is WARP's `-s` (every
intermediate), `rawdata=False` / `plots=False` skip the raw copies / figures.
WARP's figures are the only products that cost noticeable time (about a third
of a default tree's CPU time); with `plots=False` a tree with every other WARP
file takes no longer than the VAC-only trees of earlier versions
(docs/BITEXACT.md "WARP tree products", Speed).
`python -m decanter.compare_full_tree WARP_ROOT DECANTER_ROOT [--bytes]` compares two
trees file by file, each class apart (FITS data bit for bit in every HDU,
header card text; text and databases line for line, npz arrays, PNG, PDF;
with `--bytes` every file's raw bytes too, npz as zip containers member by
member, only fields that are time stamps of WARP's or IRAF's format masked,
in place).

The parameter file's extraction options are followed bit for bit as well:
`Background Subtraction: average|median|minimum|fit` (apall's background,
sampled in the listfile's `bg=` regions -- without one, WARP's `INDEF`: the
aperture window, and for `fit` every pixel at or right of the aperture
centre, as IRAF's `rg_xadd` reads it; the `_none`/`_bg*` spectra and the
background panel of `VACflux.pdf`), `CUTRANSFORM flux: yes` (the
rectification conserves flux) and `Extract 2d spectrum: yes` (`twodspec/` and
`8B-OBJ-2DSPEC`: apall strips, WARP's resampled strips, through the same
truncate/shift/dispcor/air chain). With `decanter.reduce` the same options
come from `config` (`Config.from_parameter_file`) and `background_region=`
(the `bg=` token); the 2-D spectra are `Reduction.obj_2d` /
`obj_2d_resample`. The option tokens are read as WARP's pyraf reads them
(`med` is `median`, `no` is a run without subtraction that still writes the
background products, `Yes` is `yes`; `m`, `avg` or `INDEF`, on which WARP
stops, raise `ValueError`), and every IRAF step reads its input as IRAF's
FITS kernel does (`-0`, subnormals, Inf/NaN become `+0`, which matters where
a background region reaches the strip's off-order pixels). See
docs/BITEXACT.md, "WARP's parameter-file options".

## RV stability check

`--serval-rv` runs SERVAL on the calibrated spectra and writes an RV-stability
plot, so a night can be checked without a separate analysis step. It needs an
event ephemeris, because in-transit or in-eclipse exposures are excluded before
SERVAL builds its template:

```bash
python reduce.py ... --serval-rv --serval-ephemeris examples/hrccs/wasp69b.toml
```

SERVAL itself is a separate program ([mzechmeister/serval](https://github.com/mzechmeister/serval)).
Point Decanter at a checkout with `--serval-dir`, or `$SERVAL`; `~/mzechmeister/serval`
and `~/serval` are tried otherwise.

`--serval-telluric-threshold` (default 0.90) masks fitted transmission below
that value.

The same check runs on an already-written reduction.

```python
import decanter

result = decanter.run_serval_rv_stability_directory(
    "output/corrected", ephemeris="examples/hrccs/wasp69b.toml",
)
print(result.exposure_rms_mps, result.figure_pdf)
```

## Transit and eclipse HRCCS

Set `observation_type = "transit"` or `"eclipse"` in the TOML `[system]`
table, along with `event_midpoint_bjd_tdb` and `event_duration_hours`. The
same fields describe both geometries.


`[injection].scale` multiplies only the synthetic planet used in the single
injection-recovery experiment.

```bash
decanter-hrccs examples/hrccs/bd143065b.toml
```

