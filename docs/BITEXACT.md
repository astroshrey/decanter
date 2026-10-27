# Bit-exact WARP parity (the WARP-clone modes)

Goal: `decanter.reduce(...)` reproduces WARP's products **bit for bit** (0
differing float32 values) on this machine, at every stage, for both the object
and sky paths, in both the locked path (WARP's per-frame trans aperture
databases supplied) and the free/standalone path (decanter solves the aperture
itself, as WARP does).

Two implementations of the WARP clone exist:

- `mode="warp-reference"`: the scalar, stage-by-stage exact port this document
  describes (the modules named in the status table). It is the reference the
  other one is checked against, and kept for that.
- `mode="warp"` (the default): the same recipe and
  arithmetic with four kernels vectorised and calibration-cached
  (`decanter.fast`, docs/VECTORISED.md). Bit-identical to `warp-reference` by
  construction on the NumPy backend; on the MLX backend (the default when MLX
  is installed) bit-identical from s07 on and in every extraction band, with
  the one exception of float32 subnormals in s06 strip pixels outside every
  extraction band (only visible with `save_intermediates=True`).

Until 2026-10-20 `mode="warp"` named the reference port; the status below was
established on it and re-checked on both implementations.

This is the development record of the WARP clone. Paths under
`~/Disk/winered/` are the verification reductions on the development
machine, and the `scripts/` it names (`scripts/compare_full_tree.py` ships as
`python -m decanter.compare_full_tree`) are
development tools not included in this repository.

Bit-identity is defined against WARP as run on this Mac mini (iraf-community
2.18.1, Apple clang, arm64). IRAF built elsewhere (x86, gcc) rounds differently,
so another machine's WARP is only reproduced to the float32 floor.

## Why a straight numpy port is a few ULPs off

IRAF SPP code is translated to Fortran then C (`f2c -KRw8 -Nn802`: REAL stays
single precision) and compiled with `clang -O2`. Things that matter:

- **FMA contraction.** clang fuses `a*b + c` inside one C expression into a
  single-rounding fused multiply-add. Use `decanter.utils.fpemu.fma32` /
  `fma64`. Which operand pair gets fused in `a*b + c*d` is not obvious: decide
  it with the oracle (below), never by guessing.
- **Mixed precision.** `x ** i` (REAL ** INTEGER) is f2c `pow_ri`, computed in
  double; a REAL times that result is a double expression, rounded to REAL only
  on assignment. Integer-times-REAL constants (`n * 1e-4f`) are REAL products.
- **Text parsing.** Databases and header values are parsed with IRAF `ctod`
  (`digits * 10.0D0**e`, not correctly rounded): `fpemu.iraf_ctod`. Values read
  into REAL variables (`dtgetr`, aperture limits, curve coefficients) are then
  cast to float32. pyraf passes Python floats to IRAF as `str(value)`.
  IRAF writes header doubles with `%.15g`-style text, and downstream tasks read
  that text back, so a value that crosses a file boundary in WARP must cross it
  the same way in decanter.
- **The FITS kernel's input mapping.** Every IRAF task reads a REAL image
  through `ieevupkr` with mapping on (`fxf_opix`, fxfopix.x:61-68): `-0.0`
  and subnormals become `+0`, `+-Inf`/NaN become 0; output is not mapped. In
  WARP only the s06 strips hold such values (off the illuminated order), and
  only where an extraction reads off-order pixels does it change a result.
  `decanter.utils.iraf_imio.imio_read_real` (checked against the compiled
  routine) is applied wherever a port of an IRAF task reads an image, never
  where WARP's Python (astropy) does.
- **SPP pointers.** In the f2c C, `memr[p - 1]` is SPP `Memr[p]`; `Memr[buf]`
  is the first element of an allocated buffer. Off-by-one errors in ported
  pointer arithmetic show up as whole-pixel shifts.
- **Stateful algorithms.** e.g. the MWCS multispec inverse carries its Newton
  start point and derivative from one call to the next, falls back to a grid
  search after 10 steps, and quantises the result. Port the statefulness.
- **Uninitialised memory.** Rarely WARP reads past a buffer (s13: the last
  pixel of an (order, cut) whose REAL FSR bounds give one pixel fewer than the
  double bounds). These are not reproducible; document them, do not chase
  them. See "Accepted s13 tail pixels" below.
- **Text round trips in MWCS.** dispcor maps output-pixel edges through the
  OUTPUT multispec WCS, whose w1/dw are stored as `"%.14g"` text
  (`smw_swattrs`); `wf_msp_fwd` evaluates `crval + cdelt*(x-1)`. Using the
  unrounded grid flipped one Newton result sitting 8e-7 quanta from a 1e-4
  rounding boundary (LCO24b setting5 m178 edge 3052; fixed in
  `iraf_msp.dispcor_edges`).

## Tools

- `scripts/iraf_oracle/gen_f2c.sh file.x` regenerates the exact C IRAF compiled.
- `scripts/iraf_oracle/build.sh libA.a libB.a` + `oracle.py` call the compiled
  IRAF kernels through ctypes, for random-input bit-exactness tests of a port.
- WARP `-s` reductions with every intermediate saved:
  `~/Disk/winered/reductions/TOI2109_decanterref` (1 frame pair),
  `AUMic` (272 frames), `GJ3090_nominal_nows` (64), `GJ3090_zifan_nows` (80).
  HIRES-J and WIDE references (two A/B pairs each, NO1-NO2, every intermediate):
  `TOI3486_decanterref` (LCO26a setting3 HIRES-J100, orders 131-154) and
  `WASP69_decanterref` (LCO24b setting4 WIDE100, orders 42-61). Parity must
  hold in all three instrument modes.
  Every other reduction under `reductions/` has the final `onedspec` /
  `sky_emission` VAC products and per-frame trans aperture databases.

## WARP tree products (`reduce_to_warp_tree`)

`decanter.reduce_to_warp_tree` writes every file `Warp_sci.py --noreport`
writes (and, with `save=True`, every `-s` file), so a decanter tree is a
drop-in for a WARP tree. What it takes beyond the s01-s13 chain:

| product | WARP | port (all REAL arithmetic as compiled, FMAs read off the `.o` files) |
|---|---|---|
| `onedspec/VAC_norm` | `PyContinuum` -> `onedspec.continuum` (`t_sfit.x`, type=ratio, spline3, order 7 / 15, niterate 2 / 10, low 2, high 3, grow 1 A) | `utils/sfit_exact.py`: SPLINE3 curfit (`cv_bevalr.x`, `cv_fevalr.x`, `cvacptsr.x`, `cvrejectr.x`, banded `cvchomatr.x`) + `ic_rejectr`/`ic_deviantr` with region growing; x = the REAL `shdr_open` wavelengths of the `SN` points (fused MWCS `mw_c1tran`), the ratio over `SN` pixels and 0 beyond |
| `onedspec/VAC_cont` | `sarith VAC / VAC_norm` | `iraf_onedspec.sa_1d` (`t_sarith.x:sa_1d`, `advzr`, BANDID1 moved, `smw_saveim`) |
| `AIR`, `AIR_norm`, `AIR_cont`, `sky_emission/AIR` | `vac2air_spec`: `dopcor(redshift=-v, isveloc=yes)` + `hedit VAC2AIR, AIRORVAC` | `iraf_onedspec.dopcor` (`t_dopcor.x`: the `SN` pixels then the last one repeated; EQUISPEC `smw_saveim` with `w1/(1+z)`, `mw_saveim` card rules) |
| `<OBJ>_sum/*` | 2..28 frames: `scombine combine=average group=all`; else `scopy` of frame 1's six files; continuum/air as for a frame | `wavelength/iraf_scombine.py` (`scb_default`, MULTISPEC `%.14g` attribute -> EQUISPEC, `scb_rebin`/`shdr_rebin` poly5, `ic_gdatar`, `ic_averager`, NCOMBINE) |
| `<OBJ>_sum_fsr*_{VAC,AIR}_norm_comb` | `scombine` of every order's sum norm | the same (`shdr_rebin` onto the all-order grid, D_MIX masks) |
| `SNR_dat/fsr*/SNratio_m*.dat` | `warp/SNratio_estimate.py` (`sarith /` median, `scombine` of the first 10, `continuum order=10`, `sarith -`, `continuum type=difference`) | `warptree_products.snr_estimate` |
| `ap<OBJ>_skyNO<i>_fm_m*trans`, `ap<OBJ>_NO<i>_s`, `aplast` | IRAF `ap_dbwrite` (`apextract/apdb.x`) by `apall` (sky, reference = the frame DB), `apscatter`, the last `apall` | `warptree_products.iraf_apdb_text` (REAL `%g` of the parsed tokens) |
| 1-D intermediates' headers | apall (`apmw.x`), `truncate`, `PySpecshift` (`scopy` + `specshift` + `scombine w1 dw nw`), `dispcor_single`'s `hedit REFSPEC1` on its *input*, `dispcor`, FSR `scopy` | `warptree_chain.py`, card by card (`utils/iraf_header.py`: in place / appended / deleted as IRAF does; `mw_saveim`'s keep-the-old-text rule) |
| `4-OBJ_mask/{sscfm,mask,maskflat,mf1,mf2,noise}`, `-s` 2-D files | `imarith`/`apscatter`/`imarith /`/`fixpix`/`cutransform` headers; `badpixmask.py` astropy files | `warptree_chain.py`, `warptree_io.py` (FITS-kernel structural cards, ORIGIN/IRAF-TLM/OBJECT/DATE) |
| `images/spatial_profile/*.dat` | `centersearch_fortrans` / `make_slit_profile` | `utils/psf.py` (the profile the bit-exact center search bins) |
| logs, `status.txt` | `warp/logger.py`, `apscat_log`, `cutransform_log`, `warp/config.py` | `warptree_products.py` |
| figures (`*.png`, `*.pdf`) | `warp/plotframes.py` | `warptree_plots.py`, line-for-line ports fed the tree's own files |

The s08 extract `CRVAL1` is apall's own logical->world value
`fma(CD, 1, W - CD)` of the strip (`apmw_open`), not the strip's `CRVAL2`;
the s11 truncate now reads that text too (`api._extracted_wcs`).

**File bytes.** The IRAF FITS kernel writes a header as its cards, blank
cards up to the end of the last 2880-byte block, and `END` as that block's
last card (`fxf_write_header`, fxfupdhdr.x:927-971: the last record is
filled with blanks and `END` goes in its last 80 bytes), in the minimal
number of blocks, and the data unit big-endian, zero-padded to a block. All
103,785 FITS files of the 61 fresh WARP runs under
`/private/tmp/decanter_fullprod/warp` and the 4,001 of the three `-s`
references have this layout (no interior blank card, no extra block) except
the astropy-written cosmic-ray files, which end `END` + spaces as astropy
writes them. decanter writes the same layout (`warptree_io._cards_bytes`;
until 2026-10-21 it wrote `END` right after the last card and padded with
spaces, the same size and card text but different bytes). So every
IRAF-written FITS file of a decanter tree is WARP's file byte for byte
except, exactly:

- the time-stamp values: `IRAF-TLM` and `DATE` (card columns 12-30, the
  ISO write time `YYYY-MM-DDThh:mm:ss`) and IRAF's `cnvdate` run time at
  the start of `FIXPIX`'s value (columns 12-23: `%3.3s %2d %2d:%02d`, day
  and hour space-padded, on IRAF's clock; `%3.3s %2d %5d`, the year, only
  for a time not later than 334 days before the process's first `cnvdate`
  call, which a run-time stamp never is; sys/etc/cnvdate.x,
  t_fixpix.x:124-125; "Run-time stamps IRAF writes" below) -- the write
  times. A `DATE` card that is not the kernel's write time (the raw frames'
  `DATE = '2025-08-07'` in `rawdata_image/`) is not one of them;
- decanter's provenance card `DECANTER= 'decanter <version> mode=<mode>'`
  (per-frame 1-D and 2-D spectra; `'decanter <version>'` on `<OBJ>_sum`
  and `norm_comb`; the 2-D frame intermediates, strips and raw copies have
  none), written in the first blank card after the last card, where WARP's
  file has 80 blanks: same file size, same offsets, every other byte
  WARP's. A header whose cards and `END` fill its last block exactly has no
  blank card, and decanter then writes no `DECANTER` card rather than a
  longer header, so that file is WARP's byte for byte up to the time
  stamps. WARP products with such a header exist: TOI763 NO14 (LCO26a
  setting1) has 104, `onedspec/VAC_flux` and `onedspec/VAC_cont` of both
  cuts in all 26 orders (215 cards + `END` = exactly 6 blocks, no blank
  card; its other products and the other frames' have blank cards), which
  decanter writes without the card, byte-identical to WARP's (full-tree
  campaign 2026-10-21, `qa/decanter_parity_all/results_fulltree/probes/decanter_card_check.json`;
  every other card-less FITS file of the 58 trees is a raw copy, a
  calibration-set copy or a 2-D frame intermediate/strip). The HIRES-J
  calibration set's comp file has no blank card either; both programs
  copy it byte for byte.

The astropy-written files -- `4-OBJ_mask/{mask,mf1,mf2,noise}` (`badpixmask.py`)
and `resample2Dspec`'s strips (astropy re-writing the IRAF strip, which keeps
its blank cards, so the `DECANTER` card sits where the template had it) --
are written by the same astropy as WARP's and are byte-identical as well.
`scripts/compare_full_tree.py --bytes` checks all of this (`fits-bytes`).

**Run-time stamps IRAF writes.** Two stamps of a WARP tree come from IRAF
itself, and decanter makes them with ports of the same routines
(`decanter/utils/iraf_time.py`): `FIXPIX = '<cnvdate> Bad pixel file is
<mask>'` (t_fixpix.x:124-125; carried by every product made from the fixed
frames: s05 `4-OBJ_mask/*_sscfm`, sky `2-SKY_mask/*_fm` and everything after
them) and the first line `# <cnvtime>` of the databases IRAF writes
(`dtptime`, xtools/dttext.x:564-576: apall's sky databases, apscatter's
`ap<OBJ>_NO<i>_s`, `aplast`; ecidentify's `ec<comp>` in a calibration set).

- *IRAF's clock*: `clktime(0)` = `ZGTIME` = `gmt_to_lst (time(0))`
  (unix/os/zgtime.c, gmttolst.c), the Unix time minus `mktime` of
  1980-01-01 00:00:00 with `tm_isdst = 0`: local *standard* time with the
  zone's offset of January 1980. Daylight saving time is never applied, and
  a zone whose standard offset changed since 1980 keeps the old one. Here
  (America/Los_Angeles) IRAF's clock is UTC-8 all year, one hour behind the
  wall clock during PDT.
- *cnvdate* (sys/etc/cnvdate.x): `"%3.3s %2d "` then `"%2d:%02d"`, the day
  and the hour space-padded: `'Oct 22  2:52'`, `'Jul  3 10:33'`.
- *cnvtime* (sys/etc/cnvtime.x): `"%3.3s %02d:%02d:%02d %02d-%3.3s-%04d"`,
  zero-padded, IRAF's English names: `'Thu 02:52:42 22-Oct-2026'`.

What WARP's trees hold (`qa/decanter_parity_all/probes/fixpix_clock_census.py`,
the 123 WARP trees on disk: 59 reductions, 64 fresh WARP 3.8.14 runs, all
made 2026-06-26..2026-10-21, i.e. during PDT): of the 241,560 `FIXPIX` cards
of the per-frame VAC spectra, 86,220 have a space-padded single-digit hour
and 155,340 a two-digit one, none a zero-padded hour; 191,664 a
space-padded day, none a zero-padded one; none the year form. In every kept
s05 file (698, in 71 trees) `FIXPIX` is the same file's `IRAF-TLM` (UTC)
minus 8 h; frame NO1's `FIXPIX` is 56.8-61.0 min and apscatter's
`ap<OBJ>_NO1_s` stamp 56.0-60.0 min *before* `status.txt`'s `Starting time`
(WARP's `time.ctime()`, the PDT wall clock) in every tree. The compiled IRAF
run through pyraf (`proto.fixpix`; `apextract.apfind`, whose database goes
through `ap_dbwrite`/`dtptime`) under 39 and 8 time zones on 2026-10-22 --
every IRAF hour 0-23, zones with daylight saving in effect (PDT, AEDT,
Chile, Newfoundland, BST, Chatham), zones whose offset changed since 1980
(Kathmandu, Apia: another date, Caracas) -- wrote exactly decanter's text of
the same second (47 of 47; tests/unit/test_iraf_time.py keeps the cases).
The stamps WARP's own Python writes are not IRAF's and stay as WARP writes
them: `status.txt` (`time.ctime()`, wall clock), an SNratio `.dat`'s
`Time:` (`datetime.datetime.today()`), the per-frame trans databases
(`time.strftime(..., time.gmtime())`, UTC, aperture.py:154).

Until 2026-10-22 (f49c5d4 and before) decanter wrote `FIXPIX`'s time as
`time.strftime("%b %e %H:%M")` -- the hour ZERO-padded, on the wall clock --
and the IRAF databases' as `time.strftime("# %a %H:%M:%S %d-%b-%Y")` (IRAF's
form, wall clock). A run started between 00:00 and 09:59 local time thus
wrote `FIXPIX = 'Oct 22 00:03 ...'` where IRAF's form is `' 0:03'` (IRAF's
clock read `'Oct 21 23:03'` at that moment). `compare_full_tree.py --bytes`
masks the field only in cnvdate's form, so that card differed in every FITS
file carrying it: the six reductions the 2026-10-22 final campaign launched
after midnight (TOI406, TOI969_night1, GJ9827, TOI270_oct, TOI620,
TOI2109) failed on it and on nothing else. On the smallest case,
TOI2109_decanterref (1 frame, `-s`) reduced by f49c5d4 at 04:03 PDT, the
first differing files are s05's `4-OBJ_mask/TOI2109_NO1_sscfm.fits` and
`2-SKY_mask/TOI2109_skyNO1_fm.fits` (`FIXPIX = 'Oct 22 04:03 ...'`; IRAF
would write `'Oct 22  3:03'`), with every s01-s04 file identical and every
later FITS product differing in that card only (1016 files; its two s13
tail files then not accepted, the tail rule needing an identical header).

Names and places follow WARP's own rules:

- `Apscatter: no`: WARP sets `obj_ssc_list = obj_s_list` (Warp_sci.py:284-297),
  so the object chain is `_s` -> `_sf` -> `_sfm` and every later name follows:
  `3-OBJ_flat/*_sf`, `4-OBJ_mask/*_sfm`, `5-OBJ_cut`, `6-OBJ_transform`, the
  per-frame databases `ap<OBJ>_NO<i>_sfm_m*trans`, the spatial profiles,
  `2d_image/*_sfm.png` (its title too) and `cutransform_log.txt`
  (`warptree.obj_suffixes`). `apertures=<WARP root>` and
  `Calibration.from_dir` find `_sfm_` databases as well as `_sscfm_` ones.
- Exactly 28 frames (`frameNumberLimit`): WARP computes `snestimate`,
  `snr_plots` and the two `*_norm_comb` spectra for `1 < n <= 28`
  (Warp_sci.py:736-757, 800-802) but moves them only for `n < 28` (877-890),
  so `SNratio_m*_fsr*.dat`, `SNratio_fsr*.png` and
  `<OBJ>_sum_fsr*_{VAC,AIR}_norm_comb.fits` (its shortFile name) stay in the
  tree root, and there is no `SNR_dat/`.
- An OBJECT longer than 13 characters: the `snestimate` input list holds the
  shortFile work names (`OBJECT_NO1_m42_fsr1.05_VAC`), and with `-s` the
  sky extraction keeps its short name, `5-SKY_extract/OBJECT_skyNO<i>_fm_m*trans1d.*`
  (Warp_sci.py:574 builds it from the short string).

The kernels were checked against the compiled IRAF objects first
(`scripts/iraf_oracle/sfit_oracle.py`: `cvinit(SPLINE3)` + `cvfit`, `ic_rejectr`
with grow, `cvvector` + `advzr` -- 0 differing float32 bits on 400 random
spectra shaped like FSR cuts), then fed WARP's own `_VAC` files: every
`VAC_norm`/`VAC_cont`/`AIR*`/sky `AIR`, the sums, `norm_comb`, the SNR `.dat`
and the sky/`_s` databases of the three `-s` references came out identical
in data and header text before any of it was wired into the tree writer
(tests/unit/test_warptree_products.py keeps a sample).

**Whole trees** (`scripts/compare_full_tree.py WARP DECANTER [--bytes]`: every
WARP file at the same path, each kind of comparison a class of its own --
`fits-data` every HDU's data unit bit for bit (the raw words: `-0.0` vs
`+0.0` and NaN payloads count), `fits-header` every HDU's non-blank card
images in order (time stamps `IRAF-TLM`, `DATE`, FIXPIX's run time excluded,
decanter's `DECANTER` card skipped), with `--bytes` `fits-bytes` the whole
file byte for byte (layout, padding) up to exactly those time-stamp values,
`DECANTER` in WARP's first blank card and the s13 tail value (a per-frame
VAC cut whose only difference is its last value; since 2026-10-21 only on
an `n_r < n` cut where the WARP tree kept the frame's s12 spectrum to judge
it, otherwise listed as `tail_pixel_unverified`); `apdb`, `text`
line for line, `npz` arrays bit for bit, PNG byte for byte, PDF byte for
byte with the `/CreationDate` value masked in place; until 2026-10-21 FITS
data were compared by value and HDU 0 only). The tables below were made before
`--bytes` existed (data and header classes); see "File bytes" above and
the 2026-10-21 re-runs in EXECUTION_LOG.md for the byte checks. Default mode
`warp` (MLX), re-checked 2026-10-21 with commit c094b20 (the directory sets
are equal too):

| WARP tree (calibration set) | frames | WARP files | identical | accepted s13 tail-pixel files | WARP files not written |
|---|---|---|---|---|---|
| TOI2109_decanterref, `-s` (LCO25b setting2 HIRES-Y100; `save=True`, `DECANTER_WARP_MINPACK=c1153`) | 1 | 1489 | 1486 | 2 (m175 fsr1.30 obj + sky) | `reduction_log/logfile` |
| TOI3486_decanterref, `-s` (LCO26a setting3 HIRES-J100; `save=True`) | 2 | 2057 | 2055 | 0 | `reduction_log/{logfile,null}` |
| WASP69_decanterref, `-s` (LCO24b setting4 WIDE100; `save=True`) | 2 | 1729 | 1727 | 0 | `reduction_log/{logfile,null}` |
| TOI2109 2025-08-07, archived (LCO25b setting2 HIRES-Y100; c1153) | 42 | 22752 | 22668 | 84 (m175 fsr1.30, every frame, obj + sky) | none |
| TOI332 2024-09-20, archived (LCO24b setting6 HIRES-Y100; c1153; scombine of 20 frames, SNR_dat) | 20 | 11238 | 11238 | 0 | none |
| TOI3486 2026-06-24, archived (LCO26a setting3 HIRES-J100; c1153) | 56 | 27691 | 27691 | 0 | none |
| WASP69 2024-09-10, archived (LCO24b setting4 WIDE100; c1153) | 190 | 77543 | 77543 | 0 | none |

The archived trees hold no intermediates (they were harvested) and were
reduced with `intermediates=False`; the `-s` references check those. The
three `-s` trees reduced with `mode="warp-reference"` give the same counts.
`scripts/bitexact_checks/end_to_end.py <ref> NO<i> both warp` passes on all
five reference frames (locked and free).

**Strict `--bytes`** (since the commit after e1e11b8, 2026-10-21). Until
then it made only FITS files a byte comparison: text and database files
were compared line by line with every line that began with a time-stamp
label dropped (`Time:`, `Starting time`, `Termination time`, `Elapsed
time`, `# Mon 22:11:18 ...`, in any file), so line terminators, the final
newline and anything else on such a line were not compared; `.npz` files
array by array (not the member order, the `.npy` headers or the zip
container); FITS `IRAF-TLM` / `DATE` / `FIXPIX` cards masked by column
whatever their value. Now `--bytes` adds `apdb-bytes` and `text-bytes` (the
file byte for byte: terminators and the final newline count) and
`npz-bytes` (the zip container member by member: names and order, each
`.npy`'s bytes, every other container byte -- compression, CRCs,
attributes, local headers, central directory -- with only the zip time
stamps masked; numpy writes them as 1980-01-01 00:00 anyway), and a field
is masked only where it IS a time stamp of the format WARP or IRAF writes
there, in place, in both files, on the same card or line: FITS
`IRAF-TLM= '<ISO>'`, `DATE    = '<ISO>'` and `FIXPIX`'s `cnvdate`; `status.txt`'s
`Starting time: <ctime>`, `Termination time: <ctime>`, `Elapsed time:
<%dm%.1fs>`; an `SNratio_*.dat`'s first line `Time: <datetime>`;
`# <cnvtime>` lines in the run's own databases (per-frame `*_NO<i>_*`,
`*_skyNO<i>_*`, `aplast`) -- the calibration set's database copies must be
WARP's bytes. These are every time-like field the run writes in the text
files of 123 WARP trees (the 59 reductions and the 64 fresh WARP runs;
`qa/decanter_parity_all/probes/text_timestamp_census.py`, 399,527 files:
no `\r` anywhere; `status.txt`, `input_files.txt`, `centersearch_log.txt`
and `waveshift_log.txt` end without a newline wherever WARP wrote them);
`reduction_log/{logfile,null}`, IRAF's task logs, hold more, but decanter
does not write them. In the 2,096,086 FITS files of the same trees
(`probes/fits_timestamp_census.py`) every `IRAF-TLM` and `FIXPIX` card has
that format, and every `DATE` card but the 4,722 date-only ones of the raw
frames' copies in `rawdata_image/`, which are compared in full. The JSON
counts the masked fields whose two values differ (`masked_differences`).
Re-run on decanter trees made at e1e11b8 (`save=True`;
`qa/decanter_parity_all/strict_rerun.py`) of the three `-s`
references and, with `warp_options.py compare --bytes`, of four option
runs, old and strict tool give the same files and counts -- 0 new
differences (TOI2109_decanterref 1486 identical + 2 tails, TOI3486 2055,
WASP69 1727, `triage_wsband_Y5` 5041 + 16 tails + its two tail-derived SNR
files, `combo_J` 2727, `bgfitindef2d_W` 2287, the 28-frame default tree of
`WASP69_n28` 14785/14785; not written: the IRAF task logs): every text,
database and npz file is WARP's byte for byte up to the masked fields.

Fresh WARP 3.8.14 runs of other configurations (2026-10-20, installed scipy,
inputs copied to `/private/tmp/decanter_fullprod/warp/<name>/inputs`; nominal
parameters except as listed; `-s` unless noted), each reduced by decanter with
the same inputs (`save=True` for the `-s` runs):

| run | configuration | frames | WARP files | identical | tail | not written |
|---|---|---|---|---|---|---|
| WASP69_n28 (no `-s`) | the first 28 listfile lines (`frameNumberLimit`): SNR files and `*_norm_comb` in the root | 28 | 14785 | 14785 | 0 | none |
| cfgA_TOI2109 | Apscatter no, Cosmic Ray no, Sky Emission no | 1 | 1214 | 1212 | 1 | `logfile` |
| apscno_TOI2109 | Apscatter no | 1 | 1485 | 1482 | 2 | `logfile` |
| WASP69_longname | OBJECT `WASP-69_b_transit` (18 characters) | 2 | 1729 | 1727 | 0 | `logfile`, `null` |
| TOI2109_nocr | Cosmic Ray no | 1 | 1480 | 1477 | 2 | `logfile` |
| cfgB_WASP69 | Cosmic Ray no, Correct Shift no | 2 | 1713 | 1711 | 0 | `logfile`, `null` |
| cfgC_WASP69 | Cosmic Ray no, Manual Aperture (`ap=` tokens) | 2 | 1711 | 1709 | 0 | `logfile`, `null` |
| cfgF_WASP69 | Cosmic Ray no, Measure Shift no | 2 | 1713 | 1711 | 0 | `logfile`, `null` |
| cfgG_TOI2109 | Cosmic Ray no, cut range 1.00, 1.20, 1.50 | 1 | 1844 | 1843 | 0 | `logfile` |
| cfgH_TOI2109 | Cosmic Ray no, one order (163) | 1 | 380 | 379 | 0 | `logfile` |
| cfgL_WASP69 | Cosmic Ray no, Manual Shift (`ws=` tokens) | 2 | 1713 | 1711 | 0 | `logfile`, `null` |

Default (non-`save`) trees of cfgA and apscno against the same `-s` runs:
every file a run without `-s` keeps is identical (1156 + 1 tail and
1320 + 2 tail); only `-s` files are absent.

The accepted tail pixels are exactly the documented s13 cases (TOI2109's
LCO25b setting2 m175 fsr1.30, obj and sky, every frame); every product IRAF
derives from those spectra (norm, cont, AIR, sums) is identical, because
IRAF's `shdr_open` reads `SN = n - 1` points of them. WARP's own Python
reads all `n`, and three of its products can depend on the value: the SNR
estimate of a 2..28-frame night (`np.median`; `SNR_dat/`, `SNratio_*`),
`spectra_image/<OBJ>_VACflux.pdf` (a page's y-limit, `np.median`) and
`reduction_log/count_and_fwhm.png` (`np.average`, only for a first-cut case
in the order it averages) -- see "Where the tail pixel reaches WARP's
products" below. `save=True` with `mode="warp"`
rectifies on the NumPy backend: with MLX the whole `6-OBJ_transform` /
`4-SKY_transform` strips carry the accepted off-band subnormals (TOI3486:
1.15M values, |WARP| <= 2.6e-29, |diff| <= 6e-36), which the default
(non-`-s`) tree never writes.

Not written: `reduction_log/null` and `reduction_log/logfile` (`-s` only:
the `scombine`/`transform`/`sfit` task logs, with user, host, times and
temporary file names), `slit_viewer/` (needs the slit-viewer frames, WARP
`-v`), and WARP's PDF report (decanter clones `--noreport`).

**Calibration-set entries in the tree root** (corrected 2026-10-21 after a
verifier's report). `Warp_sci.py:202-209` copies every entry of the
calibration set (`glob(calibpath + "*")`, so no dot files) into its work
directory -- files with `shutil.copy` (content and mode bits),
directories with `shutil.copytree`, a name already there (the listfile
copy, `status.txt`, the raw frames) skipped -- and `Warp_sci.py:1099-1108`
moves only the flat, mask and comp files `input_files.txt` names (lines
2-4), `database/`, `input_files.txt`, `calibration_parameters.txt` and
`readme.txt` into `calibration_data/` (the flat and mask to the trash
without `-s`). Every other entry stays in the tree root.
`calibration_LCO25a_setting5_HIRES-Y100` holds an empty `sys` (mode 0770),
so the WARP trees of HD86226, LTT3780, LTT3780_b, TOI620 and TOI969_night2
have a 0-byte `sys` in their root (mode 0770, and in all five the mtime of
WARP's copies of the comp file and `input_files.txt`, to the second),
which decanter did not write; the full-tree campaign
(`qa/decanter_parity_all/REPORT_fulltree.md`) listed it as a stray
non-product and its driver exempted the name -- wrongly: it is a WARP
product of the generic copy. No other set under `calibration_data/` has an
entry beyond WARP's own files except `LCO23a_setting2`'s `.swp`, a dot file
neither program copies. `reduce_to_warp_tree` now copies these entries
the same way (`warptree.calib_set_strays`, listed in
`decanter_provenance.json` as `calibration_strays`) and refuses
(`ValueError`, before anything is reduced) an entry whose name WARP's own
run uses in the root (`calibration_data`, `reduction_log`, `rawdata_image`,
`spectra_image`, `SNR_dat`, `slit_viewer`, `Trash`, `null`, `logfile`,
`winered_logo.pdf`, `<OBJ>_NO<i>`, `<OBJ>_sum`, the 28-frame `SNratio_*`
/ `*_norm_comb` files), where WARP stops, mixes it with its product or
moves it (`null`, `logfile`, `winered_logo.pdf` go to `reduction_log/`,
Warp_sci.py:1125-1130). The campaign driver's exemption is removed.
Checked against fresh WARP 3.8.14 runs with copied inputs
(`warp_options.py`, `triage_wsband_Y5`: that set; `triage_wsbandw_W`:
LCO24b setting4 WIDE100; both with `notes.txt`, `old/flat_old.txt` and
`.hidden` added to the copy): WARP's roots hold `sys` (Y5), `notes.txt`
and `old/flat_old.txt`, not `.hidden`, and decanter's trees the same files,
identical (the runs' rows in "s11: scombine's 1 % coverage rule" below).

**Speed** (none of this changes a byte). One spawn pool runs every phase
(frames, s11-s13, sums, raw copies, figures). `<OBJ>_sum` is built from the
frames' spectra in memory. `shdr_rebin`'s aligned-pixel sums are vectorised
(`iminterp.resample_poly5`; `norm_comb` 4x faster). The continuum fits lay
their FMA accumulations out rank by rank with the float64 products formed
once and `cvacpts`' ranks computed without a sort, and in mode `warp` a
frame's fits (and an SNR estimate's) are batched: `sfit_exact.sfit_ratio_many`
runs `ic_rejectr` for all of them in lockstep with one `ordered_fma` and one
sigma-chain loop per iteration, bit-identical to `sfit_ratio` on 524 real and
random spectra (`warp-reference` keeps the one-at-a-time fits). Figures are
rendered once (pyplot's `savefig` renders a second time for `draw_idle`), PNGs
are quantised straight from the rendered RGBA buffer (the same file as
WARP's `savefig` + `savefig_small` re-read), the 2-D figures are drawn from
the in-memory arrays and the multihole geometry is computed once per process;
`4-OBJ_mask`'s `mf1`/`mf2`/`noise` are taken from the vectorised cosmic-ray
detector, which computes them anyway. Timing against f7095a9 (before the
tree products), both run at the same time, `workers=2`, mode `warp`,
`DECANTER_WARP_MINPACK=c1153`, 2026-10-21 (load average 7-13):

| night | f7095a9 tree | this tree | wall s (f7095a9 -> now) | CPU s |
|---|---|---|---|---|
| PDS70, 8 frames | default | default | 29.6 -> 42.3, 27.5 -> 42.0 | 56.3 -> 78.8, 52.9 -> 76.4 |
| PDS70 | default | `plots=False` | 28.9 -> 27.2 | 55.1 -> 51.0 |
| PDS70 | `intermediates=False` | `intermediates=rawdata=plots=False` | 24.0 -> 25.6 | 46.1 -> 49.2 |
| G2938, 2 frames | default | default | 13.5 -> 19.2 | 26.4 -> 35.7 |
| G2938 | `intermediates=False` | `intermediates=rawdata=plots=False` | 12.5 -> 13.4 | 24.7 -> 26.5 |

So everything a default tree adds beyond the spectra f7095a9 wrote, except
the figures, costs nothing net (the FITS writer is faster than f7095a9's); the
figures (per frame 26 spatial-profile PNGs, the 2-D image, the mask PNG at
400 dpi and its PDF; per night two multi-page PDFs, one PNG per order, the
SNR and count figures, a PNG per raw frame) are about a third of a default
tree's CPU time (+50 % over the same tree without them; `plots=False` skips
them). The parent process peaks at about 160 MB.

## WARP's parameter-file options (`Background Subtraction`, `CUTRANSFORM flux`, `Extract 2d spectrum`)

decanter no longer refuses any option of WARP's parameter file: the three
that change the extraction are implemented in both implementations and
selected the way WARP selects them, by the parameter file
(`Config.from_parameter_file`, `paramfile=`) and, for the background region,
by the listfile's `bg=` token (`reduce_to_warp_tree`;
`decanter.reduce(..., background_region=)`).

WARP hands the `Background Subtraction` and `CUTRANSFORM flux` tokens to pyraf
as typed (warp/config.py:346-355, Warp_sci.py:561, cutransform.py:137), so
decanter takes them as pyraf 2.2.4 does (`decanter.config`: `pyraf_choice`,
`apall_background`, `pyraf_bool`; probed in WARP's environment with
`setParam` of `echelle.apall` / `transform` and with `transform` runs, 36 +
26 tokens, `tests/unit/test_param_options.py`):

- `Background Subtraction`: every token but `none` turns `flag_skysub` on --
  WARP then writes `_none`, `_bg`, `_bgc`, `_bgcw` and the background panel of
  `VACflux.pdf` -- and is apall's `background`, minimum-matched and
  case-sensitive (`basicpar._StringMixin` + `minmatch`; quotes stripped):
  `a`..`average`, `me`..`median`, `mi`..`minimum`, `f`..`fit`, and `n`, `no`,
  `non` = `none`. So `no` is a run without subtraction that still writes the
  background products (all zero). Refused: `m` (median or minimum), `avg`,
  `averages`, `Median`, `FIT`, `None`, `INDEF` (pyraf raises; WARP stops,
  "ERROR in extraction").
- `CUTRANSFORM flux`: `yes`/`no` in any case, `y`/`n`/`Y`/`N`
  (`_BooleanMixin` compares through `_Boolean`, lower-cased). Refused:
  `INDEF` in any case (pyraf takes it, `transform` stops with IRAF error 603
  "Parameter not a legal boolean": "ERROR in 2d image reduction"), `true`,
  `1`, `on`, `ye`, quoted `'yes'`.
- A pyraf parameter indirection (`)task.param`) is not modelled:
  `NotImplementedError`.

Those raise in `check_supported`, as does a listfile `bg=` region WARP stops
on, before anything is reduced (`warptree.check_bg_tokens`: with a
background, `aperture_plot` reads every comma-separated region as `float(a)`,
`float(b)` of `a:b` -- "ERROR in PSF center search", plotframes.py:551-557 --
and `write_apdatabase` `float()`s every number of a sample without `INDEF` --
"ERROR in extraction", aperture.py:176-179). A one-number region therefore
never reaches apall's background through WARP; `INDEF` (no `bg=`) does.
`float()` also takes `inf`, `-inf`, `Infinity`, `1e309` (= inf) and `1e39`:
WARP writes them as the database's `xmin`/`xmax` (`%.2f` of the extremes of
the window and the region numbers), and apall's `dtgetr` (apdb.x:227-228,
`gargr` -> `ctod` -> REAL) either finds no number ("Error in database field
value") or overflows the REAL conversion and kills `x_apextract.e`: WARP
stops, "ERROR in extraction", whatever the background mode, `none`
included (fresh WARP runs `vfpO1r3_bginf_W` `bg=inf:5` and `vOpt3_bginf_Y`
`bg=-6:-2,2:inf`). `check_bg_tokens` refuses such a region (`ValueError`;
both runs now refused before anything is written). A `nan` is never an
extreme -- Python's `min`/`max` keep the finite window -- so it only reaches
the `sample` text, which IRAF parses as text: not refused. A region number
of magnitude 2^30 or more is refused as not modelled
(`NotImplementedError`): WARP does not stop, but IRAF's `nint` of such a
limit (apskyeval.x:143-145) leaves 32-bit integer range.

**Several tokens of a kind on one listfile line.** WARP's `inputDataList`
appends every `bg=`, `ws=` and `ap=` token of every line to one list per
kind -- `INDEF` / `0.` for a line without `bg=` / `ws=` -- and frame
`NO<i>` takes entry `i - 1` (warp/config.py:128-145); its length check
(config.py:147-152) never fires, because `inputDataList` runs before the
parameter file is read (Warp_sci.py:163 vs 231). So two tokens on line 1
give `NO2` line 1's second token and shift every later frame.
`read_warp_listfile` reproduces these flat lists (`warp_flat_tokens`) and
warns (`UserWarning`) for every such line; the waveshift log keeps WARP's
list lengths (`waveShiftAve`/`Std`/`Num` zeros and, with `Manual Shift`,
`waveShiftAdopted` sized by the `ws=` list; the text log prints `frameNum`
entries). Fresh WARP 3.8.14 `-s` runs: `vfpO1r3_bgdup_W` (median, `bg=-6:-2
bg=2:6` on line 1: WARP's databases NO1 `sample -6:-2`, NO2 `2:6`) 1633/1633
identical (decanter kept each line's last token before: 1127 differing
files), `wsdup_W` (`Manual Shift: yes`, `ws=0.1 ws=0.2` / `ws=0.3`: NO2
shifted by 0.2, `waveShiftAdopted` = `[0.1, 0.2, 0.3]`) 1727/1729 and
`apdup_W` (`Manual Aperture: yes`, median, `ap=-18:-9 ap=9:20 bg=-24:-20
bg=22:26` / `ap=-17:-8 bg=-6:-2,2:6`) 1885/1887, the two others WARP's IRAF
task logs; 0 differing files, directories equal.

| option | WARP | decanter |
|---|---|---|
| `CUTRANSFORM flux: yes` | `transform flux=yes` (cutransform.py:137) for the object and sky orders: every output pixel times the bilinear Jacobian `J(U,V)` of `tr_setup`'s subsampled grid (`zgrid = 1/abs(der[4] der[8] - der[5] der[7])`, trsetup.x:299, 319; with no U surface `= 1/abs(dV/dy)`, the derivative `tr_invert` left behind), times `du`, times `dv` (`amulr`/`amulkr`, t_transform.x:577-581) | `rectify/iraf_transform.py`: all three inversion walks (exact, inlined, lanes) record `der[8]`; `TransformPlan.jgrid`/`jcoeff`/`flux_factor`; `tr_transform(flux=True)`, `tr_transform_many(flux=True)`; `fast/rectify.py` multiplies each band pixel by `OrderBand.flux_factor()` (the same values) |
| `Background Subtraction: average\|median\|minimum\|fit` or a prefix pyraf matches (+ `bg=`) | `apall background=<mode>` (Warp_sci.py:561) with the background ICFIT of the per-frame database WARP wrote (`sample` = the `bg=` region or `INDEF`, `xmin`/`xmax` = the extremes of window and region, Chebyshev order 1, `naverage -3`, `niterate 0`); also `apall background=none` (`_m###_none.####`), `sarith none - subtracted` (`_bg`), `truncate` (`_bgc`), `dispcor_single` (`_bgcw`, never shifted), and `plot_all_frames_flux_BG` for `<OBJ>_VACflux.pdf` | `extract/apbackground.py`: `ap_extract`'s data buffer widened to the ICFIT's `xmin`/`xmax`; `ap_skyeval` (per line the window `[nint(xmin + s) - 1, nint(xmax + s) + 1]`; `average` = linear `asigrl` over each region / summed widths, `median` = `amedr` (`asok`) of the pixels `nint(a)..nint(b)`, `minimum` from 9.9e36, `fit` = the pixel grid centred on the aperture (`nint(1000 (a - c)) / 1000`, `asieval`), `rg_xranges`/`rg_order`/`rg_merge`, `rg_wtbin` medians of 3 with summed weights, REAL order-1 `cvfit`: `V * (1/M)`); `ap_sum`'s sky over the same pixels with the same edge weights (`fma(wt1, s, wt2 * s)` + interior `s`, the compiled form of the data sum; `apskyeval.o` has no fused multiply-adds). An `INDEF` sample (WARP's without `bg=`) is the window `[xmin, xmax]` for average/median/minimum (`ap_skyeval` finds no `a$b` pair) and, for fit, the range `[0, INDEFR]`: `rg_xadd` reads a token's second number as `rval2 = rval1; ctor (str, j, rval2)`, and `ctor` stores `dval` -- which `ctod` set to 0 before finding no number -- whatever it returns (ctor.x:20-31, ctod.x:76, 122-123; only `int(log10 |dval|) > 38` leaves `rval2` alone), so a token with one number `v` is `[min(v, 0), max(v, 0)]` and `INDEF` takes every grid point at or right of the aperture centre: the whole window of an aperture right of the centre (the B nod of an A-B/B-A pair), nothing ("No sample points for fit", sky 0) for one entirely left of it (`utils/iraf_ranges.py`: `ctod`/`ctor`/`gctol` and the `rg_xadd` token parse, as compiled). Scalar reference `sky_values`/`apsum_background`; vectorised `sky_values_fast`/`apsum_background_fast` (`fast/apsum.apsum_multi(sky=, buffer_cols=)`) |
| `Extract 2d spectrum: yes` | `apall format=strip` with the same background (Warp_sci.py:576), `resample2Dspec` (scipy cubic `interp1d`, 0.01-pixel averages, rescaled to the box sum), each `truncate`d (`scopy w1=1. w2=2048.` of an NDSPEC image + 3 `hedit`), shifted (`PySpecshift` or `scopy`), `dispcor_single`, `vac2air_spec`; `twodspec/{VAC,AIR}`, `8B-OBJ-2DSPEC/{1,2,3}` | `apbackground.strip_extract(_fast)` (`ap_strip`: per line a linear `asifit` of `[nint(low + s) - 1, nint(high + s) + 1]`, evaluated from `x = low + s - ix1 + 1` while `1 <= x <= nasi`, `int(high - low + 1)` rows, minus the sky); `extract/resample2d.py` (WARP's own Python: the same NumPy/SciPy calls on the big-endian strip and WARP's `apertureSet` trace); every row through the object's s11-s12 kernels (`_finalize_wavelength(twod=)`); headers `warptree_chain.ua_strip`, `ua_truncate_2d`, `ua_copy_2d`/`ua_shift_2d`, `ua_dispcor_2d`, `vac2air(multispec=True)`; the resampled strip written as WARP writes it (astropy, `warptree_io.write_astropy_replaced`) |

What decides bits and header text in the 2-D chain:

- An `apall` strip is an NDSPEC image: `scopy` turns each row into an
  EQUISPEC aperture, `APNUM<i> = 'i 0 i.00 i.00 1.00 1.00'` (`smw_gwattrs`
  ND); `scombine` and `dispcor` build their output as MULTISPEC, which keeps
  one limit pair (`'i 0 i.00 i.00'`); the strip header itself gets the title
  ` - Aperture <m>`, `APNUM1`, `DISPAXIS = 1` re-added and a `linear` 2-D WCS
  (dispersion on axis 1, `CRVAL2`/`CRPIX2` dropped, `CDELT2 = CD2_2 = 1.`).
- Changing the dispersion of one aperture of a multi-aperture EQUISPEC image
  converts its WCS to MULTISPEC (`smw_swattrs` -> `smw_esms`), so `W`, `CD`
  and `z` pass `%.14g` text: in `dopcor` (the 2-D `AIR` spectra: `CDELT1`
  differs in the last digit from the single-aperture formula in 22 of 26
  HIRES-Y orders) and in `specshift` with a non-zero shift (the shifted zero
  point; `shift_truncated(multispec=True)`).
- The 2-D data are the 1-D kernels row by row (truncate, shift, dispcor; the
  `AIR` rows are the `VAC` rows).

In mode `warp` the background and 2-D options rectify whole strips on the
NumPy backend (the regions and `resample2Dspec`'s window may lie off the
extraction band, where the MLX backend's values differ: subnormals, which
apall would read as `+0`, but also small normal values next to flushed
intermediates, and `resample2Dspec` reads the strip as written;
docs/VECTORISED.md); `flux` alone keeps the band and the active backend.

**Verified** (2026-10-21) against fresh WARP 3.8.14 `Warp_sci.py -s
--noreport` runs, inputs copied to `/private/tmp/decanter_fullprod/warp/<run>/inputs`
(`scripts/bitexact_checks/warp_options.py run-warp`): the nominal parameters
plus the option(s), on one frame pair per instrument mode -- `_Y` TOI-2109
WINA00053571/72 (LCO25b setting2 HIRES-Y100, one listfile line), `_J`
TOI-3486 WINA00060317/18 (LCO26a setting3 HIRES-J100, A-B and B-A lines, so
frame 2 is shifted), `_W` WASP-69 WINA00045837/38 (LCO24b setting4 WIDE100,
A-B and B-A) -- with `bg=-6:-2,2:6` (line 1) and `bg=-7:-3,1:5` (line 2)
where a background is on (`bgavgindef`: average without `bg=`, i.e.
`INDEF`; `bgfitindef2d`: fit + 2-D without `bg=`; `vfp_bgfitindef_J`: fit
without `bg=`, a run a verifier made the same way, its raw frames and
calibration set checked byte-identical to `science_data`/`calibration_data`;
`combo`: fit + flux + 2-D; `combo_Wlong`: the `_W` frames with
OBJECT `WASP-69 b transit`, longer than WARP's 13-character `shortFile`
limit; four more runs a verifier made the same way, 2026-10-21:
`vfo0r2_farmed2d_W` median + 2-D with `bg=-60:-40,40:60` / `bg=-1000:1000`
and `vfpO2_farfit_Y` fit + 2-D with `bg=-130:-100,100:130`, regions in the
strips' off-order pixels; `vfp_bgmedabbr_W` `Background Subtraction: med`
(the A-B line only) and `vfpO0_fluxYes_bgno_Y` `Background Subtraction: no` +
`CUTRANSFORM flux: Yes`, tokens pyraf abbreviates / coerces; see "IRAF reads
its input through the FITS kernel's IEEE mapping" and "Option tokens" below).
decanter trees with `save=True` from
the same copied inputs (`warp_options.py compare <mode>`, `workers=2`),
every WARP file compared by `scripts/compare_full_tree.py`; 0 differing
files in every run, the directory sets equal, the only files not written
WARP's IRAF task logs `reduction_log/logfile` (and `null`), the only
non-identical ones the documented s13 tail pixels (TOI-2109 m175 fsr1.30, obj
and sky). Seconds are decanter's tree time on the loaded machine (load
average 8-11; WARP took 114-272 s per run):

| run | WARP files | identical, `warp` | identical, `warp-reference` | s13 tail | not written | directories |
|---|---|---|---|---|---|---|
| `bgavg_Y` | 1593 | 1590 (32 s) | 1590 (103 s) | 2 | 1 (logfile) | 77 equal |
| `bgavg_J` | 2249 | 2247 (61 s) | 2247 (119 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgavg_W` | 1889 | 1887 (30 s) | 1887 (52 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgmed_Y` | 1593 | 1590 (32 s) | - | 2 | 1 (logfile) | 77 equal |
| `bgmed_J` | 2249 | 2247 (60 s) | - | 0 | 2 (logfile, null) | 133 equal |
| `bgmed_W` | 1889 | 1887 (30 s) | 1887 (48 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgmin_Y` | 1593 | 1590 (33 s) | - | 2 | 1 (logfile) | 77 equal |
| `bgmin_J` | 2249 | 2247 (61 s) | - | 0 | 2 (logfile, null) | 133 equal |
| `bgmin_W` | 1889 | 1887 (30 s) | 1887 (47 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgfit_Y` | 1593 | 1590 (33 s) | - | 2 | 1 (logfile) | 77 equal |
| `bgfit_J` | 2249 | 2247 (62 s) | 2247 (152 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgfit_W` | 1889 | 1887 (31 s) | - | 0 | 2 (logfile, null) | 133 equal |
| `bgavgindef_Y` | 1593 | 1590 (33 s) | 1590 (108 s) | 2 | 1 (logfile) | 77 equal |
| `vfp_bgfitindef_J` | 2249 | 2247 (64 s) | 2247 (127 s) | 0 | 2 (logfile, null) | 133 equal |
| `bgfitindef2d_W` | 2289 | 2287 (39 s) | 2287 (75 s) | 0 | 2 (logfile, null) | 147 equal |
| `flux_Y` | 1489 | 1486 (32 s) | 1486 (38 s) | 2 | 1 (logfile) | 77 equal |
| `flux_J` | 2057 | 2055 (60 s) | 2055 (65 s) | 0 | 2 (logfile, null) | 133 equal |
| `flux_W` | 1729 | 1727 (30 s) | 1727 (32 s) | 0 | 2 (logfile, null) | 133 equal |
| `x2d_Y` | 1749 | 1746 (65 s) | 1746 (105 s) | 2 | 1 (logfile) | 84 equal |
| `x2d_J` | 2537 | 2535 (79 s) | 2535 (107 s) | 0 | 2 (logfile, null) | 147 equal |
| `x2d_W` | 2129 | 2127 (39 s) | 2127 (50 s) | 0 | 2 (logfile, null) | 147 equal |
| `combo_Y` | 1853 | 1850 (69 s) | 1850 (218 s) | 2 | 1 (logfile) | 84 equal |
| `combo_J` | 2729 | 2727 (82 s) | 2727 (198 s) | 0 | 2 (logfile, null) | 147 equal |
| `combo_W` | 2289 | 2287 (40 s) | 2287 (85 s) | 0 | 2 (logfile, null) | 147 equal |
| `combo_Wlong` | 2289 | 2287 (40 s) | 2287 (83 s) | 0 | 2 (logfile, null) | 147 equal |
| `vfo0r2_farmed2d_W` | 2289 | 2287 (41 s) | 2287 (68 s) | 0 | 2 (logfile, null) | 147 equal |
| `vfpO2_farfit_Y` | 1853 | 1850 (65 s) | 1850 (538 s) | 2 | 1 (logfile) | 84 equal |
| `vfp_bgmedabbr_W` | 1126 | 1125 (27 s) | 1125 (44 s) | 0 | 1 (logfile) | 77 equal |
| `vfpO0_fluxYes_bgno_Y` | 1593 | 1590 (31 s) | 1590 (37 s) | 2 | 1 (logfile) | 77 equal |

Without `-s` products, and locked (no WARP file missing that decanter was
asked to write, no directory WARP does not have):

| run | tree (`mode="warp"`) | WARP files | identical | s13 tail | not written (`-s` only / intermediates) |
|---|---|---|---|---|---|
| `flux_Y` | default (rectified on MLX) | 1489 | 1322 | 2 | 165 |
| `flux_J` | default (MLX) | 2057 | 1753 | 0 | 304 |
| `flux_W` | default (MLX) | 1729 | 1473 | 0 | 256 |
| `x2d_Y` | default | 1749 | 1478 | 2 | 269 |
| `combo_J` | default | 2729 | 2233 | 0 | 496 |
| `bgmed_W` | default | 1889 | 1633 | 0 | 256 |
| `bgfitindef2d_W` | default | 2289 | 1873 | 0 | 416 |
| `vfp_bgfitindef_J` | default | 2249 | 1945 | 0 | 304 |
| `bgavg_W` | `intermediates=False` (`VACflux.pdf` from the scratch `_bgcw`) | 1889 | 1261 | 0 | 628 |
| `combo_Y` | `intermediates=False` | 1853 | 1290 | 2 | 561 |
| `combo_J` | `-s`, `apertures=<WARP tree>` (locked) | 2729 | 2727 | 0 | 2 (task logs) |
| `bgfit_W` | `-s`, locked | 1889 | 1887 | 0 | 2 (task logs) |

0 differing files. Locked trees (`apertures=<WARP root>`) used to differ in
one file, `reduction_log/aperture_log.npz` (48/48 and 40/40 window limits):
WARP logs its center search's floats (`lowlim`/`highlim`, Warp_sci.py:
547-557, 596), a locked run only has the per-frame databases' `%.2f` tokens
(WARP's values are up to 0.0046 px from them). Since 2026-10-21 a locked
run takes WARP's logged window from `<WARP root>/reduction_log/aperture_log.npz`
wherever that log has the same orders and frames and its value's `%.2f`
text is the database token decanter extracted with
(`warptree.warp_logged_window`; `decanter_provenance.json` `aperture_log`
says which): combo_J and bgfit_W above are now identical in every file,
data, header text and bytes (`compare_full_tree.py --bytes`). Without that
log (a WARP root whose `reduction_log` was not kept, or a different
listfile) the npz holds the database values; `aperture_log.txt` is the
same either way (`%.2f`).

Per stage, each step fed WARP's own input from these runs, 0 differing
values or header cards (time stamps aside) everywhere:

- background-subtracted extraction (`6-OBJ_transform` strip + WARP's
  per-frame database -> `_m###.0###`): the vectorised code on all 16
  background runs (every order and frame), the scalar reference (sky values
  and sums, also against the vectorised ones) on the five `_Y` background
  runs (average, median, minimum, fit, `INDEF` average; 26 orders each);
  both implementations on the two fit + `INDEF` runs (`vfp_bgfitindef_J` 48
  strips, `bgfitindef2d_W` 40 strips and their 40 `format=strip` outputs);
- `apall format=strip` (scalar and vectorised) and `resample2Dspec`:
  `x2d`/`combo` `_Y`, `_J`, `_W`, every order and frame (26, 48, 40 strips
  per run);
- the 2-D rows through truncate / shift / dispcor (`2dcut`, `2dcuts`,
  `twodspec/VAC`): orders 163/175 (`_Y`), 131/146 (`_J`), 42/52 (`_W`), both
  frames, strip and resampled, including the shifted frames (-0.1156 and
  -0.0347/-0.0457 px);
- every 2-D header step (strip, truncate, copy/shift, dispcor, air; strip
  and resampled): all orders and frames of `x2d`/`combo` `_Y`, `_J`, `_W`;
- `transform flux=yes` (`5-OBJ_cut`/`3-SKY_cut` -> `_m###trans`, obj and
  sky, all orders): `flux_Y` NO1 with each inversion walk (exact, inlined,
  lanes; 52 strips each), `flux_J`/`flux_W` NO1-2, `combo_J` NO2, `combo_W`
  NO1, `tr_transform` and `tr_transform_many` alike.

**`fit` with `sample INDEF`** (corrected 2026-10-21 after a verifier's
report): decanter used to read a sample token with one number -- `INDEF`,
what WARP writes for every listfile line without `bg=` -- as `[v, v]`, so a
B-nod aperture's fit found no points and its sky was 0. On
`vfp_bgfitindef_J` that made 827 of 2249 files differ in both modes, every
NO2 product: NO2 m146 `_m146.0146` 5516/5528 values, median 2205.8 (WARP's
unsubtracted `_none` sum) instead of WARP's 92.70. With
`utils/iraf_ranges.py` (the compiled `ctod`/`ctor`/`gctol` and `rg_xadd`'s
token parse; `skyeval_regions` follows `ap_skyeval`'s scan position too, and
`image2d/apscatter._parse_sample` uses the same parse -- WARP's `10:2000`
reads as before):

- whole trees: the `vfp_bgfitindef_J` and `bgfitindef2d_W` rows above, 0
  differing files in both modes; re-run unchanged afterwards in mode
  `warp`: `bgfit_J`, `bgmed_W`, `combo_Y`, `bgavgindef_Y`, `bgmin_J`,
  `bgavg_W`, and in `warp-reference`: `bgfit_J`, `combo_W`, `bgavgindef_Y`
  (0 differing, the identical counts of the table);
- per stage (above): 0 differing values; NO2 m146 now has sky median 245.69
  and extracted median 92.70, W NO2 m42's strip 27324/27324 identical;
- the IRAF routines themselves, through ctypes on this Mac's compiled
  objects (`scripts/iraf_oracle/build_ranges.sh`, `ranges_oracle.py`):
  `ctod`/`ctor` 755,750 (string, position) cases, 0 differences -- 48,866
  more enter ctod's sexagesimal branch, which the port refuses
  (`NotImplementedError`; WARP's `float()` stops on every such `bg=` token
  first) -- and `rg_xranges` + `rg_order` + `rg_merge` 120,084 (sample,
  grid) cases, 0 differences (the old parse: 35,824 differences or
  exceptions);
- IRAF `apall` itself through pyraf, WARP's call on WARP's strips
  (`scripts/iraf_oracle/pyraf_apall_background_check.py`): the session
  first reproduces WARP's own 1-D and strip files of the frames (6/6 bit
  for bit), then 120 cases -- the A and B nods of W m42 and J m146 and a
  -28..28 window; fit with `INDEF`, `5`, `-5`, `0:3e38`, `-6:-2,2:6`,
  `5abc:7`, `*`, `abc`, `:5`, `5:1e39`, `12`, `-12:INDEF`; average, median
  and minimum with `INDEF`, `-6:-2,2:6`, `5`, `5abc:7`; onedspec and strip
  -- give 600 comparisons (vectorised and scalar sum and strip, scalar ==
  vectorised sky), 0 differences.

**IRAF reads its input through the FITS kernel's IEEE mapping** (corrected
2026-10-21 after a verifier's report). `fxf_opix` opens an image's pixels with
`ieesnanr (0.0); ieemapr (YES, NO)` (sys/imio/iki/fxf/fxfopix.x:61-68, for
READ_ONLY, READ_WRITE and WRITE_ONLY), and every buffer of REAL pixels is
unpacked by `ieevupkr` (fxfupk.x:61; sys/osb/ieeer.x, the portable version, in
`libvops.a`): a value whose exponent field is 0 -- `-0.0` and every
subnormal -- becomes `+0.0`, one whose exponent field is all ones -- `+-Inf`,
NaN -- becomes the NaN value 0.0; output is not mapped. So a task writes the
subnormals and `-0.0` its arithmetic makes, and the next task reads `+0`.
decanter passed its in-memory arrays to the ports instead, and apall's port
read WARP's s06 strips with their subnormals.

Where it matters: a scan of 56 WARP trees (the three `-s` references and the
53 runs under `/private/tmp/decanter_fullprod/warp`: 97,459 FITS files,
14.9e9 values) found 30.0M subnormals and 0.40M negative zeros, all of them in
the rectified strips `transform` writes (`6-OBJ_transform`,
`4-SKY_transform`, off the illuminated order), and no `+-Inf`/NaN anywhere; no
other file WARP writes (2-D frames, extractions, `format=strip` outputs, 1-D
products, calibration sets, raw frames) holds a value the mapping changes. A
strip value off the order only reaches a product through apall's background
regions (or a 2-D strip line whose aperture is off the order), so it showed
only with regions far from the aperture, in both implementations (the
verifier's counts at d60b72e; `apall_stage.py --nomap` below reproduces the
same files at the stage level):

- `vfo0r2_farmed2d_W` (a verifier's WARP run: median, 2-D, `bg=-60:-40,40:60`
  / `bg=-1000:1000`): 1 file of 2289 differed, the NO1 m42 `format=strip`
  output (13/27324 values: decanter -1.05e-41 .. -2.79e-40, WARP +0.0; the
  median sky picked a subnormal);
- `vfpO2_farfit_Y` (fit, 2-D, `bg=-130:-100,100:130`): 38 m159 files of 1853
  (the 2 others are the s13 tail pixels): a subnormal running sum `V` where
  IRAF has exactly 0 breaks a float32 rounding tie in the next `fma(3, median,
  V)`, so the sky is 1-2 ULP off (extraction 15/5551 values; strip, 2-D cuts,
  `VAC_norm` and the `twodspec` products after it).

`decanter.utils.iraf_imio.imio_read_real` is `ieevupkr` with mapping on
(`scripts/iraf_oracle/build_ieee.sh` + `ieee_oracle.py`: the compiled routine
on every exponent and sign with edge and random mantissas plus 20M random
words, 20,035,840 words, 0 mismatches). Every port of an IRAF task applies it to
the image it reads, and nothing that ports WARP's Python does:

| IRAF read | decanter |
|---|---|
| apall (object, sky, `format=strip`, every background mode) | the data buffer of `apsum`, `fast/apsum.apsum_multi`, `sky_values(_fast)`, `apsum_background(_fast)`, `strip_extract(_fast)` |
| `imarith -`, `apscatter`, `imarith /`, `imcopy` + `fixpix`, `imcopy` + `transform` (object and sky) | their inputs in `decanter.reduce` |
| `scopy` truncate, `PySpecshift` (`scopy` + `specshift` + `scombine`), `dispcor`, the FSR `scopy`, `sarith`, `continuum`, `dopcor`, `scombine` | `scopy_wavelength_truncate`, `shift_truncated`, `dispcor_one_order`, `truncate_spectrum_iraf`, `iraf_onedspec.sa_1d` / `continuum(_many)` / `dopcor`, `iraf_scombine`; `sarith none - subtracted` in `reduce` and the tree writer |
| not mapped (WARP reads with astropy) | the cosmic-ray mask, the PSF center search and `make_slit_profile`, `resample2Dspec`, the s10 shift search, the figures |

The vectorised `apsum_multi` keeps NaN (`keep_nan=True`): in a band-evaluated
strip NaN marks a pixel the rectification did not compute, and an extraction
that reads one must come out non-finite so the order is rectified in full
(`fast/rectify.orders_to_redo`); no strip `transform` writes holds a NaN (its
input was read mapped).

Checked (2026-10-21; both implementations unless noted):

- the two runs above: 0 differing files in `warp` and `warp-reference` (the
  table). Per stage, WARP's own strips and databases through every apall
  port (`scripts/bitexact_checks/apall_stage.py`: object sum, `_none`,
  `format=strip`, sky sum, scalar and vectorised; the strips hold 459,403 /
  508,263 values the mapping changes): 360 + 234 comparisons, 0 differing;
  with apall reading the strips as written (`--nomap`), exactly the
  reported ones differ: `farmed2d_W` NO1 m42 strip, `farfit_Y` NO1 m159
  sum and strip;
- IRAF apall itself (`scripts/iraf_oracle/pyraf_apall_background_check.py`,
  WARP's call on WARP's strips, now also on `vfpO2_farfit_Y` m159,
  `vfo0r2_farmed2d_W` NO1 m42 and `bgfit_J` NO1 m131 with the samples
  `-130:-100,100:130`, `-90:-60,60:90`, `-60:-40,40:60`, `-1000:1000`):
  the session first reproduces WARP's own files of all 8 strips (7 1-D and
  4 `format=strip` files, bit for bit), then 256 cases give 1280
  comparisons with decanter (scalar and vectorised sum and strip, scalar ==
  vectorised sky), 0 differing; with `--unmapped` (apall reading the strips
  as written) 40 comparisons in 15 cases differ, every one a region far from
  the aperture (`-130:-100,100:130`, `-90:-60,60:90`, `-60:-40,40:60`,
  `-1000:1000`);
- unchanged elsewhere (every identical count as before, 0 differing files,
  directory sets equal): in mode `warp` with `save=True` the 25 other runs of
  the table, the 10 runs of "WARP tree products" (cfgA-cfgL, apscno,
  longname, nocr), 14 more verifier runs (`vfo0r2_*`: odd and
  `INDEF`/`none` `bg=` tokens, many frames, selected orders, manual and no
  shift, fit/median + `INDEF` + flux + 2-D, a B-A pair; `vfp_manualbg_W`,
  `vfp_noscat_W`, `vfpO2_farfit2d_W`, `vfpO2_farmed_J`) and the three `-s`
  references (1486 / 2055 / 1727 identical); default trees of
  `vfo0r2_bg8_W` (8 frames, 5185/5185) and `WASP69_n28` (14785/14785), of
  the four runs above (MLX for `vfpO0_fluxYes_bgno_Y`) and `intermediates=
  False` trees of the two token runs; in `warp-reference` 13 of the runs
  (`bgfit_J`, `combo_W`, `bgavgindef_Y`, `x2d_J`, `flux_W`, `bgmed_W`,
  `vfp_bgfitindef_J`, `bgfitindef2d_W`, `vfpO2_farfit2d_W`,
  `vfpO2_farmed_J`, `vfo0r2_oddbg_fit2d_W`, `cfgA_TOI2109`, `cfgC_WASP69`)
  and the three references: 84 whole-tree comparisons in all.

**Option tokens** (corrected 2026-10-21 after a verifier's report).
`check_supported` used to accept only apall's full choice words and
`yes`/`YES`/`y`/`Y`/`no`/`NO`/`n`/`N`, and gated the background products on
the mode instead of `flag_skysub`. WARP runs `Background Subtraction: med`
(`vfp_bgmedabbr_W`: its NO1 extractions equal `bgmed_W`'s) and `Background
Subtraction: no` with `CUTRANSFORM flux: Yes` (`vfpO0_fluxYes_bgno_Y`: the
strips equal `flux_Y`'s, `_none` equals the extraction, `_bg*` are zero);
decanter refused both (`ValueError`), and the verifier's stand-in
`Config(flag_skysub=True, skysub_mode="none", fluxinput="yes")` stopped on a
missing `_bgcw` after writing 1486 files. Now the tokens are read as
described at the top of this section, and both runs give 0 differing files
in both implementations (table above): `status.txt` keeps the tokens as typed
(`skysub_mode: no`, `fluxinput: Yes`), `cutransform_log` `flux=Yes`, `BANDID1`
the matched word (`background none` / `background median`).

**Apertures WARP never writes** (2026-10-21, after a verifier's report).
WARP's per-frame databases start the dispersion window at line 1 (`low = 1 -
centre`) and keep the aperture on the strip. Two other geometries, which a
hand-edited database can give, were wrong in decanter:

- no line with a usable window (an aperture off the strip on every line):
  IRAF writes zeros, so did the scalar ports, but `sky_values_fast`,
  `strip_extract_fast` and `apsum_background_fast` raised (`np.zeros((nb,
  wmax))` with `wmax <= 1`); they now return the same zeros;
- a window starting after line 1 (`bmin > 1`): `ap_extract` passes `l1 = ys =
  1` to `ap_skyeval`, `ap_sum` and (through `ap_output`) `ap_strip`
  (apextract.x:478, 498, 536, 1347), so the trace
  is evaluated at buffer lines while the buffer origin `astart` comes from
  image lines (apextract.x:436-443). With a curved trace IRAF's pointers then
  leave the line: `ap_sum` reads `Memr[data + ix]` and the sky
  `Memr[sbuf + (iy-1) nx - 1 + ix]` in other lines, `ap_strip` the sky at
  `sbuf + (iy-1) nx + nint(low + s) - xs + c1 - 2 + i`, `ap_edge` the edge
  interpolant's coefficients below `c[1]`; and `ap_asifit` may get fewer than
  2 samples, where `asifit` errors and apall writes nothing (any format,
  since `ap_sum` runs first). The scalar `apsum_background` raised
  `IndexError` (an empty sample), and every port used the line's own sky and
  clipped indices. Now `apsum_exact._ap_sum_line` follows IRAF's pointer
  arithmetic in the whole buffers; reads before a malloc'ed REAL array return
  memio's header cells (sys/memio/msvfwa.x:46-50, 64-bit IRAF, little-endian:
  `c[0]` / sky `-1` = 0.0, the high half of the lower sentinel 20030125
  (minit.x:7), `-2` its low half, `-4` the element count, `-6` TY_REAL = 6;
  `IrafRealBuffer`, `_LinearAsi.with_header`); `asifit`'s error is a
  `ValueError` ("ASIFIT: too few points for LINEAR", also from
  `strip_extract(_fast)` via `apbackground.ap_sum_asifit_check`); a read of addresses or
  past an array's end (malloc padding) is refused (`NotImplementedError`).
  The vectorised ports (`apsum`, `apsum_multi`, `apsum_background_fast`,
  `strip_extract_fast`) find such lines from the geometry
  (`apsum_exact.irregular_lines`) and evaluate them with the scalar line; no
  line of a WARP database is one (`bmin = 1`: the two traces agree), so WARP's
  products are unchanged (`apall_stage.py` on `vfo0r2_farmed2d_W`,
  `vfpO2_farfit_Y`, `bgfit_J`, `bgmin_W`, `combo_Y`: 1444 comparisons with
  WARP's files, 0 differing; no timing change).

  Checked against IRAF itself (`scripts/iraf_oracle/pyraf_apall_bmin_check.py`,
  WARP's apall call, both formats, all five background modes): WARP's Y m159
  and W m42 strips with windows from lines 1, 2, 5, 50, 1000, centre + 217
  and 30 before the end (ending 0 or 100 lines early), a synthetic 30 x 40
  strip with four off-strip apertures and a synthetic 300 x 120 strip with
  steep traces (5-80 px) and windows from lines 20-200: 260 cases, every
  output of both implementations identical to IRAF's (194 x 4) or both stop
  (asifit; 49 1-D, 54 strip), except 12 x 4 that decanter refuses (sky
  reads past the buffer's end: the -40 px trace with a background) and 5
  1-D refused where IRAF stops as well. Before: the vectorised `apsum` /
  `apsum_multi` differed from IRAF already at `bmin = 50` (in mode `none`,
  the regular extraction), the scalar raised `IndexError` at `bmin = 1000`
  where IRAF stops, and the vectorised background ports raised at
  `bmin = 2992` where IRAF writes zeros. Unit tests:
  `tests/unit/test_apall_odd_apertures.py`.

Unit tests: `tests/unit/test_warp_options.py` (IRAF primitives, scalar ==
vectorised for every mode and sample and for apertures on either side of the
centre, `INDEF` fits, the `bg=` tokens WARP stops on, the transform's
Jacobian from every walk, the 2-D `APNUM` helpers),
`tests/unit/test_iraf_ranges.py` (`ctod`/`ctor` against values of the
compiled routines, `rg_xadd`'s second number), `tests/unit/test_param_options.py`
(the pyraf probe's accepted and refused tokens, `no` keeps `flag_skysub`),
`tests/unit/test_iraf_imio.py` (`ieevupkr`'s words, every apall port and the
1-D ports reading a strip/spectrum with subnormals, `-0.0`, `+-Inf`, NaN as
its mapped copy, scalar == vectorised, the band NaN still poisoning the
vectorised sum; 14 of its 16 tests fail with the mapping removed).

## Status (all stages bit-identical; checked 2026-10-20 on both implementations)

Checked with `scripts/bitexact_checks/end_to_end.py <root> <NOi> both <mode>`
(raw frames -> `decanter.reduce`, locked and free path, obj + sky, every saved
WARP product) on TOI2109_decanterref NO1 (HIRES-Y, with
`DECANTER_WARP_MINPACK=c1153`), TOI3486_decanterref NO1-2 (HIRES-J) and
WASP69_decanterref NO1-2 (WIDE), re-run 2026-10-20 after the MINPACK default
change and again after `warp` became the vectorised implementation (the
script fails on any undocumented difference or missing product):

| mode / backend | result |
|---|---|
| `warp-reference` | every stage 0 differing values, except the WARP uninitialised last pixel in TOI2109 s13 fsr1.30 m175 (1 obj + 1 sky value) |
| `warp` (default), NumPy | the same |
| `warp` (default), MLX | the same, plus near-zero s06 whole-strip pixels off the illuminated orders outside every extraction band (accepted MLX subnormals, docs/VECTORISED.md: 0.29-0.64M of 36-107M strip values per frame pair, abs(WARP) <= 4.5e-29, abs(diff) <= 6.0e-36); s08-s13 bit-identical |

`warp-reference` and the default give the same bits with every `reduce`
option (docs/VECTORISED.md, Verified).

All-transit campaign follow-up (2026-10-20, `qa/decanter_parity_all/REPORT.md`):
the s02 factor map, the inverted-aperture swap and `APNUM1` were fixed in both
implementations. `end_to_end.py` on fresh WARP `-s` reruns of Qatar6 NO9/NO10
and HD207496 NO35/NO36 (pair trees, renumbered NO1/NO2): every stage including
`mask_*_s` bit-identical, locked and free, in `warp-reference` and `warp`
(NumPy and MLX; MLX only with the accepted off-band s06 subnormals). TOI270c
full night (52 frames, default mode): 5408/5408 VAC files, every header
(`APNUM1` included), waveshift_log, aperture_log and per-frame DB tokens
identical to WARP. The center-search MINPACK flags (below) mark exactly the
three frames whose aperture_log differs (piMen NO259, NO338, L9859_b NO38;
NO1 of both reductions: none, identical).

| stage | WARP product | status |
|---|---|---|
| s01 obj-sky | `_s` | bit-identical |
| s02 CR mask | `mask_*_s` | bit-identical (`utils/cosmic_ray.py`: WARP's factor-map loop line for line -- 28 slit bins (r_s, r_s+2], r_s = -28..26, float32 `np.std` clipped at strict `|v| < 5 std` with an `==` early stop, builtin `max` NaN semantics, float32 noise model; default mode emulates the float32 reductions per tile, `fast/image2d.py`) |
| s03 apscatter | `scatter_*`, `_ssc` | bit-identical (`image2d/apscatter.py` + batched REAL curfit/icfit `utils/icfit_exact.py`; kernels vs `iraf_oracle/apscatter_oracle.py`, full frames vs fresh IRAF via `iraf_oracle/pyraf_apscatter_check.py`, WARP via `scripts/bitexact_s03.py`) |
| s04 flat divide | `_sscf` | bit-identical |
| s05 fixpix | `_sscfm`, sky `_fm` | bit-identical (`image2d/fixpix.py`) |
| s06 transform | `_m###trans` | bit-identical (`rectify/iraf_transform.py`; default mode `fast/rectify.py`, MLX backend up to the accepted subnormals outside the extraction bands) |
| s07 aperture (free path) | `ap*_NO#_sscfm_m###trans` | bit-identical, DB text token for token (`extract/aperture_solve.py`); manual aperture (`aperture=(low, high)` / listfile `ap=`) checked on TOI2109 NO1 |
| s08 apall box sum | `_m###.0###` | bit-identical (`extract/apsum_exact.py`; an inverted aperture, low > high after WARP's clamp to the reference window, is swapped as `ap_extract` does, `apextract.x:139-149`; `APNUM1` is apall's `cen + low`, `low + (high - low)` from the database tokens, `apextract.x:1195`) |
| s10 cross-frame shift | `waveshift_log.npz` | bit-identical (`waveshift/measure.py:warp_waveshift_log`) |
| s11 scopy truncate + shift | `_m###c`, `_m###cs` | bit-identical (`waveshift/apply.shift_truncated`, with `scombine`'s 1 % coverage blank since 2026-10-21, below) |
| s12 dispcor | `_m###csw` + header WCS | bit-identical (`wavelength/iraf_msp.py`, `iraf_wcs.py`; edges from the output WCS's `%.14g` text) |
| s13 FSR cut | `_fsr*_VAC` + header WCS | bit-identical except WARP's uninitialised last pixel when `n_r < n_out` (see accepted s13 tail pixels) |

**s11: scombine's 1 % coverage rule** (corrected 2026-10-21 after a
verifier's report). WARP shifts each truncated spectrum with
`PySpecshift`: `scopy`, `specshift` (the shift added to the zero point),
`scombine(w1=1., dw=CDELT1, nw=NAXIS1)` back onto the unshifted grid.
`scombine`'s `scb_rebin` (t_scombine.x:381-402) maps the input's pixel
edges 0.5 and SN + 0.5 to output pixels (`c`, `b`, REAL) and keeps only
`nint(min(b,c) + 0.01) .. nint(max(b,c) - 0.01)`: an output pixel that
gets less than 1 % of an input pixel is missing data, and with one input
spectrum `ic_averager` writes `blank` (0.) there. decanter resampled
without that window, so in a frame whose shift lies within 1 % of a whole
number of truncate pixels it kept the partial-coverage integral of the
edge pixel (`reductions/LTT3780_b`, a fresh WARP 3.8.14 tree that kept its
1-D intermediates: NO9 m163 `_cs[0]` WARP 0.0, decanter 5007.33; the `_cs`
and `_csw` of NO9 (+0.49617), NO14 (+0.49668) and NO127 (-0.49983), all 26
orders, differed). `shift_truncated` -- the shift of both implementations,
for the 1-D spectra and every row of the 2-D strips -- now applies the
window (`waveshift.apply.scb_coverage` with both linear MWCS transforms;
the 2-D rows too, which `specshift` makes MULTISPEC and `smw_saveim`
writes back as EQUISPEC, smwsaveim.x:59-78;
`wavelength.iraf_scombine.scb_rebin_window` is the overlap test
`<OBJ>_sum`'s `scombine` already used). The thresholds, as the 12-digit
text pyraf passes (WARP's own `PySpecshift` through pyraf on both sides of
each, `scripts/iraf_oracle/pyraf_specshift_check.py`):

| truncate grid | first pixel blank from | last pixel blank from | two pixels from |
|---|---|---|---|
| CDELT1 0.5, 4095 px (HIRES-Y, HIRES-J) | +0.494999974967 | -0.495056152344 | +0.994999945164 / -0.995056152344 |
| CDELT1 1.0, 2048 px (WIDE) | +0.989999949933 | -0.990051269532 | +1.98999989033 / -1.99005126954 |

(the two sides differ by the REAL rounding of `c` near 1.5 and of `b` near
4094.5 / 2047.5). A pixel with no overlap at all (beyond a whole pixel of
shift) was 0 before too.

What the edge pixel reached: `_cs`, through `dispcor` the `_csw` pixels
1 to 6-9 and 4091-4092 to 4095 (HIRES, by set) or 1-5 and 2045-2048
(WIDE), every 2-D
`_2dcuts` row and with them `twodspec/{VAC,AIR}` (strip and resampled,
uncut). Of the 1-D FSR cuts at WARP's 1.05 / 1.30, only WIDE's: decanter's
s12 and s13 ports on all 19 calibration sets under `calibration_data/`
(every order; shifts of +0.99233, -0.99967, +1.9985, -1.9995 px) change no
HIRES-Y or HIRES-J cut, but on LCO24b setting4 WIDE100 the last value of
m42 fsr1.30 (its cut ends at `_csw` pixel 2043.6) for a positive shift --
in `triage_wsbandw_W` decanter d2dc0a6 wrote -368.99118 where WARP has
-368.99078; IRAF's continuum and dopcor read SN = n - 1 points of that cut,
so only `VAC_flux` itself. `compare_full_tree` counted that file as an s13
tail (a last value only); it now takes such a file as a tail only on an
`n_r < n` cut, judged from the WARP tree's own s12 spectrum where it kept
one (m42 fsr1.30 is 1864 / 1864), and lists the tails it cannot confirm
(`tail_pixel_unverified`).

Frames on disk in the band (`waveshift_log.npz` of every WARP tree,
`scb_coverage`, a blanked pixel the old resample left non-zero): HD136352
NO7, 68, 69, 335, 348, 352, 360, 548, 647; HD15337 NO15; LTT3780 NO55;
LTT3780_b NO9, 14, 127; TOI1231 NO2, NO16; TOI421 NO33; WASP69 NO174, NO180
(WIDE, negative shifts). Only LTT3780_b kept its `_cs`/`_csw`; the VAC
spectra of the others were identical in the full-tree campaign (none is a
WIDE frame with a positive in-band shift), and re-run with the fix WASP69,
TOI1231, HD15337 and TOI421 give the campaign's counts again (below;
HD136352 was not re-run).

Checked (2026-10-21, ae9a07d unless noted):

- WARP's own `PySpecshift` (pyraf) on WARP's `_c` files of LTT3780_b NO9
  m163, TOI2109 NO1 m175, TOI3486 NO1 m146, WASP69 NO1 m42 and the 2-D
  `_2dcut` strips (strip and resampled) of x2d_Y m163 and x2d_W m42: 2042
  (spectrum, shift) cases -- the threshold steps above, WARP's LTT3780_b
  shifts, random shifts up to 3 px -- 0 differing values (d2dc0a6: 284
  cases differed); the oracle first reproduces WARP's own LTT3780_b NO9
  m163 `_cs` bit for bit;
- WARP's own `_c` / `_2dcut` files and WARP's applied shifts of 69 WARP
  trees (the seven trees under `reductions/` that kept them and every
  successful run under `/private/tmp/decanter_fullprod/warp`): 17,329 1-D
  spectra and 1,828 2-D strips (21,952 rows), 0 differing (d2dc0a6: 78,
  exactly LTT3780_b NO9, NO14, NO127 x 26 orders);
- fresh WARP 3.8.14 `-s` runs with `Manual Shift: yes` and `Extract 2d
  spectrum: yes` (`warp_options.py`): `triage_wsband_Y5` (LTT3780 NO1-4,
  `ws=` +0.4961666666666666, -0.4998333333333333, +0.494999974967,
  -0.495056152343: WARP blanks the first, the last, the first and no pixel)
  and `triage_wsbandw_W` (WASP-69 NO1-2, +0.989999949933 and
  -0.990051269532: first and last), decanter `save=True`, every WARP file
  compared (`compare_full_tree --bytes`), directory sets equal:

| run | WARP files | `warp` identical | `warp-reference` identical | s13 tail | other | d2dc0a6 (`warp`) |
|---|---|---|---|---|---|---|
| `triage_wsband_Y5` | 5061 | 5041 (126 s) | 5041 (181 s) | 16 (m178/m179 fsr1.05, obj + sky) | `SNratio_m178_fsr1.05.dat`, `SNratio_fsr1.05.png` | 4414: 624 differing (`_cs`, `_csw`, the strip and resampled `_2dcuts`, `twodspec` VAC and AIR of both, each 78 = NO1-3 x 26) + the two SNR files; `sys`, `notes.txt`, `old/` missing |
| `triage_wsbandw_W` | 2131 | 2129 (44 s) | 2129 (49 s) | 0 | - | 1806: 320 differing (the same eight kinds, each 40 = NO1-2 x 20) + NO1 m42 fsr1.30 `VAC_flux` (a tail to the old `compare_full_tree`, a difference to 1b2652d's); `notes.txt`, `old/` missing |

  (not written in either: the two `-s` IRAF task logs). The Y5 SNR files
  are the documented tail-pixel class (a 4-frame night on a set with
  `n_r < n_out` cuts: decanter's `snr_estimate` fed WARP's own VAC files
  gives WARP's `.dat` and both PNGs exactly; m178 fsr1.05 S/N total 101.9
  in WARP's file, 102.0 in decanter's);
- `qa/decanter_parity_all/driver_full.py` (frozen worktree ae9a07d, mode
  `warp`, `workers=2`, the options each WARP tree implies; `sys` no longer
  exempted; LTT3780_b, made after the campaign's `inventory.json`, from
  `inventory_extra.json`, installed scipy, its s13 tails predicted from
  WARP's own s12 spectra): every one PASS_KNOWN_ONLY, 0 UNEXPLAINED, 0 WARP
  files missing, directory sets equal --

| reduction | WARP files | identical | s13 tail (KNOWN) | tail-derived `VACflux.pdf` (KNOWN) | campaign (d2dc0a6) |
|---|---|---|---|---|---|
| LTT3780_b (136 frames, intermediates) | 90693 | 90148 | 544 | 1 | recheck: 89991 identical, 157 differing, `sys` missing |
| HD86226 | 52209 | 51816 | 392 | 1 | 51815, `sys` exempted |
| LTT3780 | 34325 | 34068 | 256 | 1 | 34067, `sys` exempted |
| TOI620 | 23805 | 23628 | 176 | 1 | 23627, `sys` exempted |
| TOI969_night2 | 21701 | 21540 | 160 | 1 | 21539, `sys` exempted |

  and four reductions with in-band frames whose WARP trees keep no s11/s12
  files (the same driver): WASP69 PASS 77543/77543, HD15337 PASS
  58520/58520, TOI421 PASS 22752/22752, TOI1231 PASS_KNOWN_ONLY 49915/50104
  + 188 tails + the tail-derived `VACflux.pdf` -- each the campaign's
  result.

**Center-search MINPACK.** WARP's s07 center search calls
`scipy.optimize.leastsq`, and scipy <= 1.14 (Fortran MINPACK) and 1.15 (C
MINPACK) stop about 1e-9 apart. WARP runs in the same venv as decanter, so by
default decanter calls the installed `scipy.optimize.leastsq` (scipy 1.14.1 on
this machine since 2026-10-19): that reproduces a WARP run made today. For
reductions made under scipy 1.15.3 (every archived run through 2026-10-19,
including TOI2109_decanterref) set `DECANTER_WARP_MINPACK=c1153`, a bit-exact
port of that build's compiled C MINPACK (`utils/minpack_exact.py`). Checked on
WARP's own s06 strips (2026-10-20): the fitted `psfCenter`/`psfWidth` floats
in `centersearch_log.npz` match exactly with `c1153` on TOI2109 NO1 (26/26
orders) and with the installed scipy on TOI3486 NO1-2 (24/24) and WASP69 NO1-2
(20/20), and differ in every order with the other flavour. The per-frame
aperture DBs (`%.2f` tokens) are identical with either flavour on all three
references, so the extracted spectra do not depend on the choice there; only
float logs of the fit (WARP's `centersearch_log`, `aperture_log.npz`) do.
`decanter_provenance.json` records the flavour used (`centersearch_minpack`).

WARP's parameter-file options `Background Subtraction`, `CUTRANSFORM flux`
and `Extract 2d spectrum` are bit-identical as well (section "WARP's
parameter-file options" above).

s10-s13 were verified on WARP's own saved inputs for every `-s` reduction
(`scripts/verify_waveshift.py`, `scripts/verify_tail_parity.py`): the shift
matrices and adopted shifts of AUMic, GJ3090_nominal(_nows) and
GJ3090_zifan(_nows) are equal element for element; 44.4M `_csw` and 46.6M
`_fsr*_VAC` values match with 273 uninitialised-memory pixels (one per frame,
m175 fsr1.30 when the REAL bounds give one pixel fewer than the double
bounds: WARP's value there is heap garbage, sometimes 0.0). The header WCS
(CRVAL1, CRPIX1, CDELT1, CD1_1, LTV1, LTM1_1) matches WARP's text on every
reduction on disk (`scripts/verify_header_wcs.py`: 5968/5968 per keyword).

**Accepted s13 tail pixels.** IRAF's `sa_sextract` sizes the FSR cut from the
double bounds (`n_out`, `ScopyWCS.n`); `shdr_extract` resamples from the same
bounds as REAL (`n_r`). Only `n_r < n_out` gives an unreproducible pixel: WARP
copies one pixel past the resampled buffer, so the last output pixel is heap
garbage (sometimes 0.0), and decanter repeats the previous value
(`wavelength/fsr.py`). `n_r > n_out` is a deterministic truncation and matches
WARP bit for bit. Whether a case occurs depends on the order's FSR bounds and
the `_csw` WCS (CRVAL1/CDELT1) of the reduction. The FSR table is not per
calibration set: WARP uses its global `warp/FSR/FSR_winered_20190610.txt`
(`Spec1Dtools.py`), and decanter's bundled `decanter/data/FSR_winered_20190610.txt`
is byte-identical. Only the WCS varies, so check a new reduction with
`scripts/fsr_tail_check.py <WARP-layout -s tree>` (it reads the tree's s12
products).

The `_csw` grid is the same in every frame of a calibration set -- the s11
shift resamples back onto the truncated grid, and dispcor's linear grid
comes from the set's id solutions and that grid alone -- so the cases are a
property of the set. Every set under `calibration_data/`, at WARP's cuts
1.05 and 1.30, `n_out / n_r` (`qa/decanter_parity_all/probes/s13_cases_by_set.py`,
2026-10-21: decanter's s12 arithmetic on the set's id files -- 4095 input
pixels HIRES, 2048 WIDE -- then `scopy_wcs`; its `_csw` CRVAL1/CDELT1 text
and length equal WARP's in all 34,900 WARP s12 files on disk, 5 sets in 59
WARP trees, and its `n_r < n_out` cases are exactly the (order, cut) cases
of the tail-pixel files the full-tree campaign found, in 67 of 67 reduction
results, the r1 re-runs and LTT3780_b included):

| calibration set | reductions on disk (frames) | `n_r < n_out`: WARP's last pixel is heap memory (accepted) | `n_r > n_out`: truncation, bit-identical |
|---|---|---|---|
| LCO23a_setting2_HIRES-Y100 | G2938 (2), TOI829 (68) | m160 fsr1.30 2559 / 2558 | - |
| LCO23a_setting4_HIRES-Y100 | LTT9779_n1 (64) | - | m167 fsr1.05 1957 / 1958, m165 fsr1.30 2470 / 2471 |
| LCO23a_setting7_HIRES-Y100 | LTT9779_n2 (72) | m178 fsr1.05 1856 / 1855, m167 fsr1.30 2434 / 2433 | m174 fsr1.05 1869 / 1870 |
| LCO23b_setting1_HIRES-Y100 | DSTucA (126) | - | m167 fsr1.05 1961 / 1962 |
| LCO24a_setting7_HIRES-Y100 | PDS70 (8), Qatar6 (52), TOI672 (74) | - | - |
| LCO24b_setting4_WIDE100 | WASP69 (190), WASP69_decanterref (2) | - | - |
| LCO24b_setting5_HIRES-Y100 | TOI270c (52), TOI544 (40) | - | m182 fsr1.05 1803 / 1804, m161 fsr1.30 2508 / 2509 |
| LCO24b_setting6_HIRES-Y100 | GJ3090_nominal(_nows) (64), GJ3090_zifan(_nows) (80), HD207496 (52), HD3167_0919 (64), HD3167_0921 (56), K2141 (48), LHS1140 (70), TOI1201 (54), TOI244 (32), TOI332 (20), TOI712 (60) | - | m174 fsr1.05 1872 / 1873 |
| LCO25a_setting2_HIRES-Y100 | TOI421 (42), TOI969_night1 (46) | - | m165 fsr1.05 2005 / 2006 |
| LCO25a_setting4_HIRES-Y100 | L9859_b (78), L9859_d (40), TOI1231 (94), TOI270d (48), TOI836 (66), WASP193 (55) | m161 fsr1.05 2028 / 2027 | m180 fsr1.30 2275 / 2276 |
| LCO25a_setting5_HIRES-Y100 | HD86226 (98), LTT3780 (64), LTT3780_b (136), TOI620 (44), TOI969_night2 (40) | m178 fsr1.05 1860 / 1859, m179 fsr1.05 1839 / 1838 | - |
| LCO25b_setting2_HIRES-Y100 | AUMic (272), TOI2109 (42), TOI2109_decanterref (1) | m175 fsr1.30 2339 / 2338 | - |
| LCO25b_setting4_HIRES-Y100 | GJ9827 (44), L1689 (36), TOI2109_take2 (42), TOI238 (64) | - | - |
| LCO25c_setting1_HIRES-Y100 | LHS1140_2025b (52) | - | - |
| LCO25c_setting4_HIRES-Y100 | TOI125 (8), TOI260 (60), TOI270_oct (44), TOI5800 (64), piMen (382) | - | - |
| LCO25c_setting5_HIRES-Y100 | HD15337 (110), TOI406 (46), TOI431 (70) | - | - |
| LCO26a_setting1_HIRES-Y100 | HD136352 (674), TOI763 (64) | **m163 fsr1.05 2012 / 2011** (He I order), m159 fsr1.30 2505 / 2504, m166 fsr1.30 2435 / 2434 | m166 fsr1.05 1966 / 1967, m180 fsr1.05 1824 / 1825 |
| LCO26a_setting3_HIRES-J100 | TOI3486 (56), TOI3486_decanterref (2) | - | m150 fsr1.05 2193 / 2194 |
| LCO26a_setting4_HIRES-Y100 | TOI1064 (100) | m168 fsr1.30 2430 / 2429 | - |

Every frame of a night on such a set carries the tail pixel at each listed
(order, cut), obj and sky (e.g. AUMic, TOI2109 and TOI2109_decanterref
m175 fsr1.30: 272 + 42 + 1 frames). WARP's values there (the full-tree
campaign's 7432 tail-pixel files) range from -1.45e33 to 2.40e38, 1547 are
exactly 0.0, none is NaN (per reduction: `qa/decanter_parity_all/REPORT_fulltree.md`,
"s13 tail pixel"; TOI836 m161 fsr1.05: -9.78e20 to 33289.5, 43 of 132
exactly 0.0, never equal to the previous pixel; LTT3780_b, a WARP run of
2026-10-20: 0 to 9943.6, 272 of 544 exactly 0.0). The truncation cases
were bit-identical in every tree of the campaign. No set has a case at the
order `count_and_fwhm.png` averages (m171 / m142 / m45 fsr1.05: `n_out =
n_r` on all 19 sets; next section).

**Where the tail pixel reaches WARP's products.** IRAF reads `SN = n - 1`
points of a tail-pixel spectrum (`shdr_open`), so nothing IRAF derives from
it sees the last pixel: `VAC_norm`, `VAC_cont`, `AIR*`, the sky `AIR`,
`<OBJ>_sum` (`scombine`, or `scopy` of frame 1) and `*_norm_comb` are
identical in every tree. WARP's own Python reads all `n` pixels of the
per-frame VAC spectra (`openspecfits`, astropy) in three places, so these
are every product the value can reach:

1. the spectrum itself, `onedspec/VAC_flux/fsr<c>/` and
   `sky_emission/VAC/fsr<c>/` (WARP's heap value, decanter's previous value);
2. the SNR estimate of a 2..28-frame night (`snestimate`,
   warp/SNratio_estimate.py:73-86, Warp_sci.py:736-757), through
   `np.median`: each frame's VAC spectrum of an (order, cut) is divided by
   `np.median(tmpdata)` over all `n` pixels, then IRAF takes over (`sarith /`,
   `scombine`, `continuum`, the difference spectra) and Python's clipped
   standard deviations follow. Where the heap value moves a frame's median,
   `SNR_dat/fsr<c>/SNratio_m<m>_fsr<c>.dat` (S/N to one decimal: identical
   unless a value sits at a rounding edge) and `SNR_dat/fsr<c>/SNratio_fsr<c>.png`
   (the unrounded values; its flux axis is the sums') differ -- in the tree
   root for exactly 28 frames. In the fresh WARP run `vOpt3_a2dN_Y2` (two
   TOI-2109 frames, LCO25b setting2: m175 fsr1.30) WARP's last pixel is
   10238.865 in both frames, NO2's median -15.60524 with it and -15.62343
   with decanter's (NO1's median is not moved), and the S/N totals of the
   three sections are 3.33810 / 3.73351 / 2.39149 from WARP's VAC files and
   3.34197 / 3.73786 / 2.39428 from decanter's (`warptree_products.snr_estimate`,
   both): the `.dat` is identical, the PNG differs (re-run 2026-10-21 in
   mode `warp`, `save=True`: 3091 of 3098 WARP files identical in data,
   header text and bytes, the 4 tail-pixel spectra, this PNG, and WARP's
   two task logs not written); in `triage_wsband_Y5` (four LTT3780 frames,
   m178 fsr1.05: S/N total 101.9 in WARP's file, 102.0 in decanter's) both
   differ. The one night on disk with 2..28 frames on a set with a case,
   G2938 (2 frames, m160 fsr1.30), is not affected: WARP's values 2978.59
   and -1577.74 move neither median, and its SNR files are identical;
3. `spectra_image/<OBJ>_VACflux.pdf` (`plot_all_frames_flux`,
   plotframes.py:173-207, and the flux panel of `plot_all_frames_flux_BG`,
   :210-257; Warp_sci.py:774-787), through `np.median`: every frame's
   spectra of the first cut, 6 frames per page, each page's y-range `[0,
   2 max(np.median(spy))]` over all `n` pixels -- so the limit moves where
   the heap value moves the largest median on the page. The pixel itself is
   not drawn: matplotlib (3.10, `Line2D._subslice`) draws a sorted line of
   more than 1000 points only between the x-limits plus one point on either
   side, the x-limits are the order's FSR range (WARP's FSR table), and the
   first cut's last pixel lies 45-50 pixels beyond its upper end. From
   WARP's own files, last value vs the previous one
   (`qa/decanter_parity_all/probes/vacflux_ylim_probe.py`), the pages whose
   limit moves among those of the tail orders: HD136352 38 of 113, HD86226
   11 of 34, L9859_b 0 of 13, L9859_d 1 of 7, LTT3780 11 of 22, LTT3780_b
   16 of 46, LTT9779_n2 4 of 12, TOI1231 5 of 16, TOI270d 1 of 8, TOI620 6
   of 16, TOI763 6 of 11, TOI836 1 of 11, TOI969_night2 6 of 14, WASP193 3
   of 10; the PDF differs in exactly the 13 nights where a page moves, and
   decanter's port fed WARP's own files gives WARP's PDF in each of them
   (the campaign; LTT3780_b in its r1 re-run);
4. `reduction_log/count_and_fwhm.png` (`peak_count_fwhm`,
   plotframes.py:489-532, Warp_sci.py:806-813), through `np.average`: the
   average count of every frame's first-cut spectrum of order 171
   (HIRES-Y), 142 (HIRES-J) or 45 (WIDE) -- or of the reduced order nearest
   to it (plotframes.py:494-497) -- sets the points and the y-limit (1.2 x
   the largest), so a first-cut tail case in that order moves the frames'
   points. No set on disk has one at m171 / m142 / m45, so no tree on disk
   is affected; a reduction of selected orders can be (order 163 alone on
   LCO26a setting1, whose m163 fsr1.05 is a case).

Not reached: `VACnorm.pdf` (the norm spectra, y-range fixed at `[0, 2]`),
`*_AIRnorm_combined_m*.png` (the sums), the s10 reference frame (the
`np.median` of the `_c` spectra, Warp_sci.py:603-610). The products in 2-4
are accepted with the tail pixel.

The 273 uninitialised pixels in the `verify_tail_parity.py` count above are
exactly the TOI2109 + AUMic m175 rows; the GJ3090 reductions contribute none.
A NO1 sweep of 59 reductions found the `n_r < n_out` class in 19 of them, on 7
calibration sets (e.g. LCO25a setting4 m161 fsr1.05, LCO26a setting1 m159/m166
fsr1.30); every differing pixel was predicted by `n_r < n_out`, and every such
spectrum differed. The all-transit campaign (58 reductions,
`qa/decanter_parity_all/REPORT.md`) found it in the He I order on one
calibration set: LCO26a setting1 HIRES-Y100, m163 fsr1.05 (HD136352, TOI763),
so the last pixel of every He I fsr1.05 spectrum of those nights is WARP's
uninitialised memory. That pixel is the red end of the cut, outside the He I
band: with HD136352 NO1's WCS (`CRVAL1 = 10755.0898914473`, `CDELT1 =
0.0343730561435223`, `CRPIX1 = -1051.13197261683`, `NAXIS1 = 2012`; the same
in every frame of HD136352 and TOI763, obj and sky) pixel 2012 is at
`CRVAL1 + CDELT1 (2012 - CRPIX1)` (WARP's own `openspecfits` formula) =
10860.38 A vacuum: 27 A redward of the He I triplet (10832.06-10833.31 A) and
1.68 A (49 pixels) beyond the order's FSR upper bound, 10858.70 A (the fsr1.05
cut ends at 10860.345 A). (Until 2026-10-21 this said 10824.21 A, 8 A
blueward of the triplet: `CRVAL1 + CDELT1 x 2011`, i.e. `CRPIX1` taken as 1.)
Through wpipe, the TOI836 and WASP69 He I timeseries and transmission spectra
from the decanter and WARP trees are bit-identical.

**Accepted s07 center-search floats (MINPACK undefined read).** scipy 1.15.3's
C translation of MINPACK recomputes a `qrfac` column norm one element past the
column (`enorm(m - j, &a[j+1 + k*lda])`); for the last column that read leaves
the Jacobian buffer, and unless it happens at the second-to-last Householder
step the value read (heap memory) steers the later pivoting. WARP's fit is
then not reproducible, not even by WARP: two runs can stop at different
points. `utils/minpack_exact.py` (`DECANTER_WARP_MINPACK=c1153`) counts these
fits in `STATS["undefined"]` (and substitutes 0.0 for the unreadable value).
Only WARP's float logs of the fit are affected -- `aperture_log.npz`
(the frame window), `centersearch_log` -- while the per-frame aperture
databases (`%.2f` tokens) and therefore every spectrum and the waveshift log
stay bit-identical. Seen in the campaign: piMen NO259 and NO338
(aperture_log differs by 3.8e-9 and 2.4e-8 px) and L9859_b NO38 (9.3e-8 px).
The full-tree campaign (2026-10-21, `qa/decanter_parity_all/REPORT_fulltree.md`)
flagged 43 fits in 22 frames of 5 reductions (L9859_b NO38, TOI270_oct
NO35, TOI270c NO33, WASP193 NO29 and NO47, 17 piMen frames); with
decanter's stand-in 0.0, 28 of them differ from WARP's logged floats, and
the magnitudes actually observed are:

- `reduction_log/aperture_log.npz`, the frame window (the medians over
  orders): piMen NO259 3.8e-9 px, NO338 2.4e-8 px, L9859_b NO38 9.3e-8 px
  (`aperture_log.txt`, `%.2f`, identical);
- `reduction_log/centersearch_log.npz`, `psfCenter` / `psfWidth` of 28 fits
  (L9859_b NO38, TOI270c NO33, WASP193 NO47, 25 fits in 14 piMen frames):
  centers 5.5e-10 to 1.9e-7 px and widths 1.2e-14 to 1.9e-6 px, except piMen
  NO176 m183, whose center differs by 2.2e-5 px and width by 6.2e-4 px
  (WARP 3.777732, decanter 3.778353) -- the one fit visible in
  `centersearch_log.txt` (`%.4f`: -17.8920 vs -17.8919, 3.7777 vs 3.7784);
  that frame's databases, its aperture_log row and every spectrum are
  identical.

The value read is heap memory, not a constant: re-reducing every flagged
frame pair with the stand-in (`minpack_exact.PAST_END_VALUE`) set to 0.0,
1.0, 0.1, -1.0 and 0.001 (`qa/decanter_parity_all/minpack_probe_all_fulltree.py`,
`results_fulltree/probes/minpack_probe_all.json`) reproduces each of the 43
fits bit for bit with at least one of them, but no single value reproduces
all (0.0: 15, 1.0: 33, 0.1: 30, -1.0: 33, 0.001: 21 of 43); fits of one
frame can need different values (WASP193 NO47: m170 0.0, m183 any of the
others). Each differing aperture_log window is `nanmedian(psfCenter) -/+
nanmedian(psfWidth)` of WARP's own logged fits of that frame, bit for bit.
decanter records the affected (frame, order) fits: `Reduction.centersearch_undefined`
(`{order: n}`) and, in a tree, `reduction_log/decanter_provenance.json`
`centersearch_minpack_undefined` (`{"NO<i>": {"<order>": n}}`); a comparison
classifies aperture_log / centersearch_log differences of exactly those frames
as KNOWN. With the installed scipy (Fortran MINPACK, scipy <= 1.14) the count
is always empty: that code has no such read.

The former "MWCS ULP" in WARP's `_csw` CRVAL1/CDELT1 was four IRAF text round
trips, none of them in MWCS arithmetic:

- `dispcor` writes the linear solution into a MULTISPEC attribute with
  `%.14g` and `smw_saveim` parses it back (`smwswattrs.x`, `smwsaveim.x`), so
  the header holds `ctod(%.14g(dw))`.
- `dc_gms` re-encodes the id database coefficients with `%.15g` before
  `wf_msp_coeff` parses them; `ctod("1.000000")` is 1.0000000000000007 but the
  re-encoded "1" is exactly 1.0, which moves the function centre/scale and so
  w1/dw by an ULP (`iraf_msp._msp_coeff`).
- pyraf hands real task parameters to IRAF as `clFloat.__str__`, i.e.
  `'{:.12}'.format(value)` (12 significant digits), not `str(value)`
  (`iraf_wcs.pyraf_real_text`); this sets the FSR bounds and the shift.
- IRAF's `ctod` fuses `value * 10 + digit` (an `fmadd` in ctod.o), which
  matters for 16-17 digit strings (`fpemu.iraf_ctod`), and IRAF prints
  doubles with `dtoc3`, which is not correctly rounded (`utils/iraf_fmt.py`,
  checked against the compiled routine on 120k values).
