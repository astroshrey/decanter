# decanter

A fast, pure-Python reduction pipeline for **WINERED** near-infrared echelle
spectra. `decanter` reproduces [WARP][warp] — the reference IRAF-based WINERED
pipeline — to float32 precision across all three instrument modes (HIRES-Y,
HIRES-J, WIDE), while running ~6–10× faster and parallelizing trivially over
frames.

[warp]: https://ui.adsabs.harvard.edu/abs/2024PASJ...76..244H/abstract "Hamano et al. 2024"

## Install

```bash
pip install -e .        # requires numpy, scipy, astropy
```

## Use

The workflow is: **load a calibration set, then reduce your frames.** Each frame
is an independent `(object, sky)` pair in, wavelength-calibrated per-order
spectra out.

```python
import decanter

# 1. Load the calibration set for your night/mode. Point it at any path —
#    either a calibration-set directory or a WARP reduction root (see below).
calib = decanter.Calibration.from_dir("/path/to/calibration_set")

# 2. Reduce an (object, sky) frame pair.
r = decanter.reduce("WINA00045837.fits", "WINA00045838.fits", calib)

# 3. Pull out spectra, keyed by (FSR cut, echelle order).
spec = r.obj[(1.30, 163)]        # an OrderSpectrum
spec.wavelength, spec.flux       # vacuum-Å grid + flux (counts)
r.orders, r.fsr_cuts             # what this reduction produced
r.sky[(1.30, 163)]               # the sky-emission path, same keying
```

A full transit is just a loop — reductions are independent, so this parallelizes
directly (e.g. with `concurrent.futures`):

```python
calib = decanter.Calibration.from_dir(calib_dir)
spectra = [decanter.reduce(obj, sky, calib) for obj, sky in frame_pairs]
```

Or write everything to FITS in WARP's directory/suffix layout:

```python
r.write_to("out/", save_intermediates=True)
```

## Calibrations

`Calibration.from_dir(path)` accepts **any absolute path** to either:

- a **calibration-set directory** — one that directly holds `input_files.txt`
  (e.g. a set you downloaded), or
- a **WARP reduction root** — one that holds a `calibration_data/` subdirectory.

It reads the set's instrument configuration (mode / slit / grating setting) from
the comp-lamp header, and `reduce()` **refuses a calibration that doesn't match
your science frame**, raising `CalibrationMismatch` before doing any work — so
you can't silently reduce a night with the wrong setting. Bypass the check with
`reduce(..., check_calib=False)` if you're deliberately re-purposing a set.

## Reduction mode

`reduce(..., mode="warp")` — the default, and today the only recipe — is a
bit-for-bit WARP clone. The `mode` argument is a forward-looking hook: future
recipes (e.g. `mode="default"`, with decanter's own improved steps) will be a
drop-in opt-in without changing the call site.

## Validation

decanter is validated frame-by-frame against WARP across the full WINERED
archive — 50+ reductions, ~4,300 science frames, all three modes — at the
float32 noise floor (median relative difference per order ≈ 4×10⁻⁵). See the
per-stage notes in `HANDOFF.md` for the numbers.

Not yet implemented: cross-frame wavelength alignment and the S/N-weighted
multi-frame stack (`decanter.combine()` is a stub). `reduce()` is single-frame;
for a transit time series you loop it and combine downstream.

## Status

- **Phases 1–2 (landed):** the full WARP-clone reduction chain, all modes,
  validated to the noise floor.
- **Phase 3 (in progress):** JAX acceleration of the extraction hot loops; a
  telluric-anchored per-frame wavelength solution (replacing WARP's
  cross-correlation waveshift); optimal / spectroperfectionist extraction.

## Repository notes

`HANDOFF.md` (per-stage parity status), `PLAN.md` / `PLAN_FULL.md` (design),
`PLAN_PHASE3.md` (roadmap), and `EXECUTION_LOG.md` (development diary) are
working documents kept in-tree. `scripts/` holds the WARP-parity campaign
harness (see `scripts/README.md`).
