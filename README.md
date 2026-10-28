# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

A fast, pure-Python reduction pipeline for **WINERED** near-infrared echelle
spectra. decanter is a clone of WARP
([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) whose output is
bit-identical to WARP's in all three instrument modes (HIRES-Y, HIRES-J, WIDE).

## Install

```bash
pip install -e .            # requires numpy, scipy, astropy
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

A time series:

```python
series = decanter.reduce_many(frame_pairs, calib)
```

WARP's cross-frame wavelength shift (s10/s11, which moves every frame onto a
reference frame's grid) is **off by default**, so each frame keeps its own
ThAr scale and the telluric lines stay in their true frame. Pass
`waveshift=True` to apply it as WARP does.

A whole night written exactly as WARP writes it (`reductions/<testname>/`), so
anything that reads WARP output works unchanged. Add `waveshift=True` (WARP's
default) for a tree byte-identical to WARP's:

```python
decanter.reduce_to_warp_tree("TOI836.txt", "science_data/2025_02_15",
                             calib_dir, "reductions/TOI836",
                             paramfile="nominal_params.txt", waveshift=True)
```

## Modes

- `mode="warp"` (default): the WARP clone, vectorised and calibration-cached,
  with the rectification on the Apple GPU when MLX is installed.
- `mode="warp-reference"`: the slower stage-by-stage port it is checked against.

Both match WARP bit for bit, except for WARP's own uninitialised last pixel in
some FSR cuts. Bit-identity is defined on Apple silicon (arm64), where the
WARP references were made. Details: [docs/BITEXACT.md](docs/BITEXACT.md) and
[docs/VECTORISED.md](docs/VECTORISED.md).
