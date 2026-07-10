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
r = decanter.reduce("obj.fits", "sky.fits", calib)   # one (object, sky) pair

spec = r.obj[(1.30, 163)]        # order 163 at FSR cut 1.30
spec.wavelength, spec.flux       # vacuum-Å grid, flux
```

For a transit, loop over your frames: `[decanter.reduce(o, s, calib) for o, s in pairs]`.
