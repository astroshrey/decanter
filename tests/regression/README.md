# Regression tests — decanter vs. WARP parity

This directory holds the per-suffix FITS-diff tests defined in
`PLAN_FULL.md` §Validation. They:

- read each stage's output from the reference WARP `-s` reduction
  (`~/Disk/winered/reductions/TOI2109_decanterref/` — see
  `tests/conftest.py:WARP_REFERENCE_PATH`; override with
  `$DECANTER_WARP_REF`)
- read decanter's corresponding output from the test workdir
- diff under the stage-specific tolerance from the table in
  `PLAN_FULL.md`

Tests skip automatically if the reference reduction is not present, so
they're safe to run on CI without the (multi-GB) dataset. Paths are
`$HOME`-relative (`decanter/_localpaths.py`) so the same checkout works
on both the laptop (`/Users/shrey`) and the Mac mini
(`/Users/shreyasvissapragada`).

**Status:** the parity tests themselves have not been written yet —
this directory is currently the spec. The reference reduction must be
a WARP `-s` run (all intermediates saved) or the stage-1-through-5
diffs will find nothing to compare against; regenerate with the
command in `CLAUDE.md` §Commands if needed.

Mark each regression test with `@pytest.mark.regression` so the slow,
data-heavy tests can be selected separately:

```bash
pytest -m regression       # only parity tests
pytest -m "not regression" # only fast unit tests
```
