# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

A fast, pure-Python reduction pipeline for **WINERED** near-infrared echelle
spectra. decanter runs WARP's reduction
([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) and, by default,
calibrates every frame's wavelengths on the OH airglow and telluric H2O lines,
so the spectra sit in the true telluric frame. A `warp-clone` mode reproduces
WARP bit for bit in all three instrument modes (HIRES-Y, HIRES-J, WIDE).

## Install

```bash
pip install -e .            # numpy, scipy, astropy, jax
pip install -e '.[mlx]'     # optional, Apple silicon: GPU rectification, same output
```

## Quickstart

```python
import decanter

# Point at any calibration-set directory, or a WARP reduction root.
calib = decanter.Calibration.from_dir("/path/to/calibration_set")

r = decanter.reduce("WINA00045837.fits", calib, sky="WINA00045838.fits")

spec = r.obj[(1.30, 163)]        # OrderSpectrum, keyed by (FSR cut, order)
spec.wavelength, spec.flux       # vacuum-Å grid + flux (counts)
```

A night's time series, every frame on its true telluric wavelength scale:

```python
series = decanter.reduce_many(frame_pairs, calib)
series.wavecal.z_star            # km/s per frame: the correction applied
```

A whole night written in WARP's layout (`reductions/<testname>/`), so anything
that reads WARP output works unchanged; `reduction_log/` also gets the
calibration (`decanter_wavecal.json`) and a before / after plot of a telluric
line in every frame (`decanter_wavecal_telluric.png`):

```python
decanter.reduce_to_warp_tree("TOI836.txt", "science_data/2025_02_15",
                             calib_dir, "reductions/TOI836",
                             paramfile="nominal_params.txt")
```

## Modes

- `mode="default"`: WARP's reduction without its cross-frame shift, then the
  OH + telluric wavelength calibration ([docs/WAVECAL.md](docs/WAVECAL.md))
  applied in place. Not bit-identical to WARP; validated on 59 WARP nights.
- `mode="warp-clone"`: WARP to the bit, including its cross-frame
  cross-correlation shift (s10/s11) as the parameter file sets it. Matches
  WARP except for WARP's own uninitialised last pixel in some FSR cuts;
  bit-identity is defined on Apple silicon (arm64), where the WARP references
  were made ([docs/BITEXACT.md](docs/BITEXACT.md)).

`waveshift=True/False` overrides WARP's cross-frame shift in either mode.
