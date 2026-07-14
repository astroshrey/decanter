# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

Fast, pure-Python reduction of WINERED near-infrared echelle spectra. Currently
a WARP ([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) near-clone
validated across all three modes (HIRES-Y, HIRES-J, and WIDE).

## Install

```bash
pip install -e .
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

For a transit, loop over your frames: `[decanter.reduce(o, calib, sky=s) for o, s in pairs]`.
