# The default WARP-clone implementation

```python
r = decanter.reduce(obj, calib, sky=sky)                       # mode="warp" (default)
series = decanter.reduce_many(pairs, calib)
tree = decanter.reduce_to_warp_tree(listfile, raw_dir, calib_dir, dest)

r_ref = decanter.reduce(obj, calib, sky=sky, mode="warp-reference")   # for verification
```

`mode="warp"`, the default of `reduce`, `reduce_many` and
`reduce_to_warp_tree`, runs the WARP clone through the vectorised,
calibration-cached implementation in `decanter.fast` described here. The
scalar, stage-by-stage exact port it was built from (and is checked against)
stays available as `mode="warp-reference"` (docs/BITEXACT.md).

The result has the same type, keys (`r.obj[(cut, order)]`), apertures, header
WCS text and on-disk layout as `mode="warp-reference"`, and the same values,
bit for bit (the one exception, MLX subnormals in unread strip pixels, is
below). Below, "reference" means `mode="warp-reference"` and "default"
`mode="warp"`; the measurements were taken when the two were called warp mode
and fast mode, on the same code.

**History of the switch.** Until 2026-10-20 `mode="warp"` ran the reference
port and `mode="fast"` this implementation; the outputs did not change when
the default switched, only the time they take. `reduce_to_warp_tree` records
what ran in `reduction_log/decanter_provenance.json` (`mode`,
`implementation`: `vectorised` | `reference`, `fast_backend`: `numpy` |
`mlx <version>` | null) and every final spectrum's `DECANTER` card names the
mode (`mode=warp` or `mode=warp-reference`).
One behavioural difference: `reduce_many(mode="warp")` keeps only the
1-D intermediates of each frame (no 2-D frames or strips), as fast mode did;
`mode="warp-reference"` keeps all of them.

**Options.** Every option of `reduce` works in the default mode, implemented
there rather than falling back to the reference port: `sky=None`,
`extract="optimal"` (with `optimal_profile`), `subtract_background=True`,
`aperture=(low, high)` / `Manual Aperture: yes`, `save_intermediates=True`
(including the full rectified strips), `workdir`, every `Config` switch
decanter supports. `save_intermediates=True`, `extract="optimal"` and
`subtract_background=True` make it rectify whole strips (slower, see Speed).

## What changes and what does not

The default runs the reference port's own orchestration (`decanter.api.reduce`): the same
stages in the same order, the same calibration inputs, the same per-frame
aperture (locked to WARP's databases, or solved by the WARP center search), the
same cross-frame shift search (s10), the same truncation / shift, dispcor grid
and FSR cuts. Four kernels are restructured for speed, keeping the reference port's
arithmetic:

| stage | `warp-reference` (formerly warp mode) | `warp`, the default (formerly fast mode) |
|---|---|---|
| s02 CR mask | scipy median filters, WARP's per-(order, window, slit bin) loops with float32 `np.std` / `np.average` | sorting-network medians (exact selections), cached tile maps, NumPy's float32 pairwise reductions emulated for all tiles at once (`fast/image2d.py:segment_std`) |
| s06 transform | IRAF `transform` per order and frame: float32 with emulated FMAs, every pixel | the same IRAF arithmetic, with the image-independent geometry cached per calibration, the spline solves batched over all orders and images, and only the extraction band evaluated (`fast/rectify.py`) |
| s08 apall | `apsum` per image | the same arithmetic on obj + sky together, aperture geometry cached (`fast/apsum.py`) |
| s12 dispcor map | stateful MWCS inverse with emulated FMAs | the same state machine in plain float64 (`fast/wavelength.py`) |

Everything else (s01, s03 apscatter, s04, s05, s07 center search, s10, s11,
s13) is the reference port's code unchanged.

**s06 arithmetic.** IRAF computes the `msifit(II_BISPLINE3)` coefficients and
the `ii_bispline3` evaluation in REAL (float32) with fused multiply-adds. The
NumPy backend does each fused step as a float64 sum of the exact float32
product, rounded once to float32; a sum that lands exactly on a float32
midpoint (where that double rounding could be wrong) is detected from its bits
and resolved with the exact two-sum error. So NumPy-backend strips are
bit-identical to the reference port's by construction. The MLX backend runs the same
operations as Metal kernels; the Metal compiler contracts them into the same
FMAs, and its band values have been bit-identical in every comparison so far.
That is an observation about the compiler (MLX 0.32.4 on an M4 Pro), not a
guarantee. One known exception, outside every extraction band: in whole strips
the GPU handles float32 subnormals differently (it flushes subnormal
intermediates), so near-zero pixels in strip regions off the illuminated order
differ: mostly subnormal or near-subnormal values, and, where a flushed
intermediate feeds a small normal value, a few pixels up to ~130 ULP apart at
|value| ~ 5e-31 or 2 ULP at 4.5e-29. On WARP's `-s` references (2026-10-20,
`scripts/bitexact_checks/end_to_end.py`, which checks every differing pixel
against the band): TOI2109 NO1 643k, TOI3486 NO1-2 576k + 577k, WASP69 NO1-2
289k + 288k of 107M / 94M / 36M obj+sky strip values, |WARP value| <= 4.5e-29,
|difference| <= 6.0e-36, none inside a band and none read by apall (every s08
1-D spectrum is bit-identical). `end_to_end.py` accepts such a difference
when it lies outside the extraction band; the parity report
(`scripts/parity_report.py`) when every differing value of a strip has
|WARP| < 1e-28 and |difference| < 1e-35 (KNOWN difference
`mlx-s06-subnormal`). Earlier, from TOI2109 alone, the bound was 3e-29 / 4e-36.

**The extraction band.** s06 evaluates each strip only where the extraction
reads it: the multihole reference trace +- the reference aperture +- 8 columns
(`BAND_MARGIN`), following the trace row by row (about 20% of the strip). WARP's
per-frame apertures reuse the reference trace and are clamped to the reference
window, and the center search reads only that window, so nothing outside is
used. Pixels outside the band are NaN; an order whose extraction comes out
non-finite is rectified in full and extracted again, and a locked aperture that
leaves the band is rectified in full from the start. `save_intermediates=True`,
`extract="optimal"` and `subtract_background=True` rectify whole strips.
So do WARP's `Background Subtraction` and `Extract 2d spectrum` options
(the `bg=` regions and `resample2Dspec`'s window, twice the aperture around
the trace, may lie outside the band), and for those two the rectification
runs on the NumPy backend whatever the active backend is, since off-band
pixels then reach the products. `CUTRANSFORM flux: yes` multiplies each band
pixel by the plan's flux factor (`OrderBand.flux_factor`, the values the
reference multiplies by). The background (`extract/apbackground.py`:
`sky_values_fast`, `apsum_background_fast` through `fast/apsum.apsum_multi(sky=,
buffer_cols=)`) and the 2-D strips (`strip_extract_fast`) are vectorised over
lines with the reference's arithmetic; a median/minimum line whose sample
holds a NaN, a zero, an infinity or overlapping regions is evaluated by the
scalar code (`asok` order decides there).

**Backends.** s06 runs on the Apple GPU through MLX (`pip install
decanter[mlx]`) when MLX imports and has a GPU, otherwise in NumPy.
`DECANTER_FAST_BACKEND=numpy|mlx|auto`, `decanter.fast.set_backend(...)` or
`with decanter.fast.backend("numpy"): ...` override the choice. Use
`numpy` when bit-identity must hold by construction rather than by observation.
`reduce_to_warp_tree` resolves the backend once in the parent process, pins it
in every worker and records it in the provenance.

**Why bit-identity matters for multi-frame runs.** WARP's cross-frame shift
search (s10, `warp/ccwaveshift.py`) takes the `argmin` of float32
squared-residual curves on 0.5 / 0.1 / 0.01 px grids. Near the minimum these
curves are flat at float32 resolution: on 13 AUMic frames, 106 of 936 search
decisions had a runner-up within 1e-6 (relative) of the winner, some of them
exactly tied. Any difference in the input spectra, even 1e-7, can flip such a
decision, and one flipped order can change which orders survive the sigma
clip: an earlier, non-identical version of fast mode moved one AUMic frame's
adopted shift by 3.9e-3 A (0.6 sigma at the object spectrum's line flanks).
The default therefore keeps s03 exact and makes s06 exact rather than
approximating them.

## Verified

`scripts/fast_parity.py` (raw frames -> `reduce(mode="warp-reference")` and
`reduce(mode="warp")`, WARP's adopted cross-frame shift applied), on the three
WARP `-s` references: TOI2109_decanterref (HIRES-Y, 1 frame pair),
TOI3486_decanterref (HIRES-J, 2), WASP69_decanterref (WIDE, 2); locked and
free path; every final `_fsr<cut>_VAC` spectrum (object and sky, every order,
both FSR cuts) plus the 1-D extractions s10 reads; 1368 spectra per backend:

| backend | spectra with any float32 bit different | header WCS + apertures |
|---|---|---|
| NumPy | 0 of 1368 | identical |
| MLX | 0 of 1368 | identical |

Also checked:

* **Cross-frame shift.** AUMic, WARP's reference frame NO161 plus NO1-NO12, free
  path: the s10 per-order shifts of both backends equal the reference port's (338 / 338)
  and WARP's own `waveshift_log`; adopted shifts identical; all 1-D extractions
  bit-identical (3,771,378 values).
* **WARP trees.** `reduce_to_warp_tree` of GJ3090_nominal_nows frames 1-20,
  default (MLX, and NumPy) against the reference: all 5721 FITS files have
  identical data and headers (apart from the `DECANTER` mode card and the run
  timestamps in `FIXPIX` and the aperture databases' first line);
  `waveshift_log.npz` identical.
* **Non-WARP options** (TOI2109 orders 163 and 170, free path, both backends):
  `sky=None`, `extract="optimal"`, `subtract_background=True` and
  `save_intermediates=True` (including the full rectified strips) are
  bit-identical to the reference port.
* **After the switch to the default (2026-10-20).** `end_to_end.py` on
  TOI2109 NO1, TOI3486 NO1-2, WASP69 NO1-2, locked and free: `warp-reference`
  and the default with NumPy are bit-identical to WARP at every stage (s01-s13)
  except WARP's uninitialised s13 tail pixel (TOI2109 m175 fsr1.30, 1 obj + 1
  sky); the default with MLX is the same apart from the s06 subnormals above.
  Every `reduce` option (`sky=None`, `extract="optimal"`,
  `subtract_background=True`, both together with `sky=None`, a manual
  aperture, `save_intermediates=True`) gives the same bits as
  `warp-reference` with both backends (TOI2109 orders 163, 170, free path).
  `reduce_to_warp_tree` (default, MLX) of TOI836 2025-02-15 NO1-NO3, with
  the adopted shifts of the existing 66-frame decanter tree
  `reductions/TOI836_dcbx` passed as listfile `ws=` tokens: all 858 FITS files
  (3.14M values) have identical data and headers to that tree (apart from
  `DECANTER`, `FIXPIX` and the checksum cards), and the 78 per-frame aperture
  databases are identical below their timestamp line; `warp-reference` the same.
* **Kernels.** CR mask, apall and the dispcor map are bit-identical (the last on
  all 19 calibration sets in `calibration_data/`); s06 NumPy band and full
  strips are bit-identical on TOI2109 (21.9M and 107M values).

## Speed

`scripts/fast_bench.py` (recipes `warp-reference`, `warp-numpy`, `warp-mlx`) on GJ3090_nominal_nows (HIRES-Y, 26 orders, obj + sky,
free path), Mac mini M4 Pro, each recipe in a fresh process, machine shared
with other jobs (load average ~5-9 on 14 cores):

| recipe | one pair, cold | one pair, warm | 10 pairs, `reduce_many` (total / per pair) |
|---|---|---|---|
| reference | 25.0 s | 16.3 s | 179.4 s / 17.9 s |
| default, NumPy | 11.5 s | 4.8 s | 59.8 s / 6.0 s |
| default, MLX | 8.5 s | 1.9 s | 31.2 s / 3.1 s |

"Cold" includes the calibration-only setup, which is cached for the rest of
the process: the IRAF transform plans (`rectify/iraf_transform.tr_setup_many`,
~4.5 s, shared with the reference port), the line schedules (~1 s), the band geometry,
the CR tile maps and the dispcor map. Warm, s06 takes ~0.1 s on MLX and ~3 s in
NumPy (reference: ~16 s); s03 (the reference port's exact apscatter) ~0.8 s; the exact 1-D
tail (s11-s13) ~0.5 s; apall ~0.25 s; the center search ~0.2 s; s02 ~0.25 s.
`reduce_many` re-runs the 1-D tail with each frame's shift, hence the higher
per-pair batch cost. Whole-strip rectification (`save_intermediates=True`,
optimal extraction, `subtract_background`) in NumPy costs about as much as the
reference port's s06.

`reduce_to_warp_tree(..., workers=N)` starts N processes, each paying the cold
setup. GJ3090_nominal_nows frames 1-20 as a WARP tree: reference,
`workers=4`: 114.6 s; default, MLX, `workers=1`: 87.5 s; NumPy,
`workers=2`: 100.8 s.

Peak memory of one `reduce` (TOI2109, 26 orders, obj + sky,
`/usr/bin/time -l`): default/MLX 1.78 GB, default/NumPy 1.87 GB (reference: 1.64 GB).

## Verifying

```bash
PYTHONPATH=. python -m pytest tests/unit/test_fast.py              # kernels + vs the reference port
PYTHONPATH=. python scripts/fast_parity.py --out /tmp/fastparity  # the bit-identity table
PYTHONPATH=. python scripts/fast_bench.py ~/Disk/winered/reductions/GJ3090_nominal_nows --pairs 10
```
