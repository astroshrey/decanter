# decanter

Fast, pure-Python reduction of **WINERED** near-infrared echelle spectra — a
drop-in [WARP][warp] clone, validated to float32 precision across all three
modes (HIRES-Y, HIRES-J, WIDE).

[warp]: https://ui.adsabs.harvard.edu/abs/2024PASJ...76..244H/abstract

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

MIT licensed.
