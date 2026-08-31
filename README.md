# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

Fast, pure-Python reduction of WINERED near-infrared echelle spectra. A WARP
([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) near-clone validated
across all three modes (HIRES-Y, HIRES-J, and WIDE), plus an absolute
wavelength calibration that anchors each order to telluric absorption and OH
airglow.

## Install

```bash
pip install -e .
pip install -e '.[wavecal]'   # for the physical wavelength calibration
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

