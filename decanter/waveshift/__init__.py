"""Per-frame wavelength-shift correction (inherently multi-frame).

:mod:`decanter.waveshift.measure` computes per-frame shifts via the
three-stage cross-correlation search WARP runs in
``ccwaveshift.waveshift_oneorder``.

:mod:`decanter.waveshift.apply` applies a given shift to one order's
1D spectrum via the IRAF ``scopy`` → ``specshift`` → ``scombine``
combination.

Not used by :func:`decanter.reduce` (single-frame reductions have no
cross-frame shift). Reserved for the eventual :func:`decanter.combine`
multi-frame stack.
"""

from decanter.waveshift.apply import apply_waveshift_one_order
from decanter.waveshift.measure import waveshift_clip, waveshift_one_order

__all__ = ["apply_waveshift_one_order", "waveshift_one_order", "waveshift_clip"]
