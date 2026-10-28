# decanter.wavecal: method and validation record

This is the development record of the global OH + telluric wavelength model
(`decanter/wavecal/model.py`). The numbers come from the WARP reductions on the
development machine; paths to `scratchpad` studies are not part of this repository.

Line list
---------
`decanter/data/oh_h2o_vacuum_lines.txt` (provenance in its header): HITRAN2020 OH positions
(with the Oliva+2015 / Bernath & Colin 2009 positions as a cross-check column, HITRAN -
Oliva = -21 m/s median), a two-temperature + per-v' population model for OH intensities,
and H2O / O2 anchor candidates with HITRAN S, gamma_air, delta_air (pressure shift applied at
run time, `P_eff` = 0.60 atm H2O, 0.38 atm O2) and modelled depth / blend metrics at
R = 60k and 25k.

Validation (WARP reductions, 2026-10-20; `tests/unit/test_wavecal.py` holds the
synthetic and regression checks)
------------------------------------------------------------------------------------------
* OH per-frame zero point vs the ThAr-free study (scratchpad `thar_free`): TOI-2109
  0.013 km/s rms, TOI-3486 0.038, WASP-69 0.16 (WIDE, 45 s frames); absolute median
  +0.084 / +0.533 / +10.55 km/s on the period ThAr scale (study +0.105 / +0.543 / +10.47;
  the absolute value moves by up to 0.05 km/s with the mode basis).  Per-frame scatter
  about the smooth drift + nod: 0.021 / 0.022 / 0.059 km/s.
* Star vs OH drift slope (OLS): 1.056 +- 0.007 (TOI-2109), 0.78 +- 0.03 (TOI-3486),
  1.074 +- 0.005 (WASP-69).  The study's 0.99 / 0.72 / 1.03 used fixed (non-tracking)
  centroid windows, which bias drift slopes low (reproduced: 0.98-0.99 with fixed windows,
  1.04-1.05 with tracking windows on the same TOI-2109 lines); WARP's own per-order s10
  xcorr shifts give 0.90-1.07 (median 0.98) on TOI-2109's telluric orders, but the held-out
  m163 residual does not correlate with the drift (slope +0.04 +- 0.03), i.e. ~1.05 holds.
* Held-out orders (excluded from the star term), per-frame residual velocity of the
  telluric cores after one Lanczos resample to the observatory frame, rms over the night,
  s10 as delivered -> default labels ("auto" = 3 x 3 modes on these full nights):
  TOI-2109 m160 / m163 / m181 0.92 / 0.91 / 0.90 -> 0.055 / 0.16 / 0.038 (m163 metric
  noise floor ~0.14 by split-half; 28/42 frames zeroed by s10); TOI-3486 m133 / m140 /
  m147 / m152 0.21 / 0.14 / 0.072 / 0.14 -> 0.11 / 0.13 / 0.080 / 0.097; WASP-69 m44 /
  m52 / m59 0.80 / 0.69 / 0.41 -> 0.092 / 0.23 / 0.094 (m52 with 2 x 2 modes: 2.84 -> 0.13
  on the four s10-zeroed frames).  2 x 2 modes: 1.20 km/s summed over the 10 orders vs 1.08.  He I
  window (+-6 A) telluric-core residual rms, TOI-2109: 0.024 -> 0.0125 (continuum noise
  0.013).
* Robustness: blank / aborted frames (all-NaN or 2 % flux) leave the other frames' labels
  within 0.5 sigma; the solution does not depend on the frame order; a missing FRAMEID
  falls back to per-frame time ids (identical result on TOI-2109); corrupt UT-END is
  replaced by EXPTIME (a corrupt EXPTIME alone is ignored).

Single images and sparse epochs (block C2b)
-------------------------------------------
:func:`anchor_night` needs a NIGHT: its time model (GP drift + nod, star tied to OH by
`beta`) and its `(x, m) x (drift, nod)` modes are fitted across frames.  One exposure (a
single RV spectrum, `decanter.reduce`), or a few exposures on different nights (RV epochs),
go through :func:`anchor_frames`.  :func:`anchor` chooses (:func:`choose_method`, overridable):
one frame -> `"single"`; one epoch with >= `series_min_frames` frames over
>= `series_min_span_h` h -> `"series"` (anchor_night, unchanged); else `"sparse"`.
:func:`anchor_reduction` takes `decanter.Reduction` output (one = single, a
`TransitSeries` night = series).

* Every frame gets its own zero points: `z_sky` from the OH lines of its sky spectrum,
  `z_star` from the telluric lines of its object spectrum (so the slit-illumination
  offset of that exposure is included), `z_inst` = `z_sky` of the paired sky spectrum
  (+ `drift_rate_kms_h x |dt|` in quadrature).  A frame with < `min_lines_static`
  telluric lines falls back to `z_inst` + the surface's slit prior.
* A zero point is the GLS mean of `d_k - mu_k` over the frame's lines
  (:func:`anchored_zero_points`): the per-line static offsets `mu_k` come from a
  :class:`StaticSurface` of the calibration set (every epoch on the surface's absolute
  scale), or without one from the epoch groups themselves (`"sparse"`, shared self prior)
  or 0 (`"single"`: line-list gauge; the static offsets then count as noise and, as in
  anchor_night, a `sky_shape` surface is fitted to the frame's own OH offsets).
* Errors: the centroid errors scaled by a fitted `lam` plus an additive excess per epoch
  group (per-line scatter is 2.5-3.7 x the formal error and correlates with it at 0.75-0.9),
  Huber sandwich inflation for heavy tails.
* Labels: `z` + the surface's static shape `S(x, m)` (sky: OH; star: tellurics) + its A/B
  pattern `nod x P(x, m)`.  The drift-dependent shape of anchor_night's modes is
  night-specific and NOT applied by default (`use_surface_drift`).  Frame-to-frame z
  values of two paths are only comparable through the labels (z alone absorbs the
  line-weighted mean of whatever pattern is applied).

The surface (:func:`build_static_surface`, :func:`surface_from_warp_reductions`;
:meth:`StaticSurface.save` / `load` / `find`) is built from one or more night solutions
of ONE calibration set (ThAr template fingerprint), quality-gated per night, and stored as a
small JSON file (:data:`SURFACE_FILENAME`, ~75 kB for 11 nights) inside the calibration set;
a surface whose prior makes a frame's residuals worse than the bare line list is rejected at
use time (warning); its A/B pattern is applied with it (`use_surface_modes`).

Follow-up (block C2c, 2026-10-21; `tests/unit/test_wavecal_follow.py`)
------------------------------------------------------------------------------
* Mode guard (:func:`_guard_modes`, `modes_min_lines_per_term` / `modes_max_leverage`): a
  detector-mode basis must have >= 2 lines per term and a maximum prediction leverage over the
  detector <= 200, else `(3, 3) -> (2, 2) -> (2, 1) -> none`.  TOI-270c's 13 telluric lines
  (leverage 1.2e8) had a 3 x 3 star pattern swinging 654 km/s across the detector: now (2, 1),
  0.26 km/s (labels sane everywhere; the unguarded labels made 25 of ~510 stellar lines
  unusable).  The guard does NOT remove TOI-270c's ~0.1 km/s series RV offset: with one RV
  template built from the guarded labels, the cross-epoch spread with TOI-270d/oct is
  0.123 +- 0.015 km/s guarded, 0.028 +- 0.041 unguarded, 0.129 without star modes, 0.110 with
  (2, 2), 0.027 for single frames (the earlier "0.395 -> 0.11" compared against a template
  built from the unguarded labels).  It triggers on 5 of 52 nights on disk (TOI-270c, pi Men,
  AU Mic, HD 136352, G 2938: 7-37 lines); the other 47 (incl. TOI-2109 / TOI-3486 / WASP-69,
  leverage <= 80) are bit-identical in every solution field and label.
* A-only frames (`night.sky is None`, e.g. `decanter.reduce(sky=None)`): OH from the object
  spectrum itself (`z_inst` at dt = 0), features with contaminating OH / telluric lines in
  their window excluded (`aonly_*`).  21 TOI-2109 A frames: `z_inst` - series OH model at
  the exposure -0.001 +- 0.026 km/s (pull sd 0.94; the A-B partner's OH: 0.079 rms), `z_star`
  - the A-B frame's +0.020 +- 0.004 km/s (sd 0.017: a 5-sigma offset, cause not identified;
  validated on an F star only - stellar lines in the OH windows are not excluded).  `decanter.reduce(sky=None)` extracts B-nod frames at
  the A slit position (sky only), so only A-nod frames carry the star.
* Epoch model (`epoch_model`, sparse path): `"nod"` (one zero point + A/B term per epoch) or
  `"nod_drift"` (+ linear drift; ABBA makes nod and drift orthogonal) instead of free
  per-frame zero points.  4-frame ABBA epochs on TOI-2109 / HD 3167 / GJ 3090 / TOI-3486 vs the
  series: `"nod"` is biased per frame by the 20-min drift (star pull sd 1.7-3.1), `"nod_drift"`
  matches `"frame"` (0.038-0.047 km/s rms) with smaller errors; epoch means agree.  Only
  `"frame"` and `"nod_drift"` remain in the code.
* One absolute scale (removed from the code: mixed results, see below) (`LineScale`, `build_line_scale`, `build_static_surface(
  scale=)`): per-line offsets `err_k` shared by every calibration set + a smooth term per set,
  fitted over all nights of a mode; files record `part["scale"]`.  Cross-epoch RV spread with
  files from other nights only (scale without the target's nights), per-set gauge -> one scale:
  TOI-969 (two sets) 0.042 -> 0.014 km/s (no file 0.010), TOI-270 0.030 -> 0.023, HD 3167 0.084
  -> 0.092, GJ 3090 0.020 -> 0.023 (TOI-270: the files are rejected on 7-13 of each night's
  frames).  An x-slope common to every set is not fixed by line positions (~0.01 km/s per
  600 px setting shift).
* Cross-epoch anchor (HD 3167, GJ 3090, TOI-969, TOI-270, LTT 9779): telluric zero points carry
  0.03-0.04 km/s per-epoch systematics (HD 3167: star - OH moves 0.13-0.17 km/s between two
  nights, depth dependent: weak lines 0.35, deep 0.12-0.14; not stellar blends (0.01 km/s with
  a K1V template divided out), not a resolved pressure shift (dP_eff = 0.14 +- 0.10 atm), not
  the drift pattern (< 0.02)).  OH zero points (+ a constant slit term) are consistent to
  ~0.01 km/s on 4 of 5 targets (chi2 0-4.4 for 1 dof) but fail on poor-sky nights (TOI-270c).
  For epoch RVs combine the two with floors (telluric 0.032, OH 0.010 km/s; leave-one-target-out
  floors give sum chi2 0.86 for 4 dof): expected per-pair sd 0.015-0.021 km/s on HD 3167 /
  GJ 3090 / TOI-969 / LTT 9779 (the observed 0.002-0.017 spreads are partly cancellation: on
  HD 3167 and LTT 9779 the telluric and OH epoch differences have opposite signs, -0.092 vs
  +0.043 and +0.050 vs -0.013 km/s).  Within a night tellurics stay the anchor.

Star-mode source (block C2d, 2026-10-22)
----------------------------------------
Two alternatives to the telluric-fitted starlight modes were tried (the OH instrument modes,
and a telluric fit with an extra airmass / time term kept telluric-only) and validated on
held-out stellar lines over 5 nights. Neither beat the telluric-fitted modes (pooled
blue-minus-red trend chi2 and per-frame stellar RV rms), so only the telluric-fitted modes
remain in the code.
