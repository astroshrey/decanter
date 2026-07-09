"""Unit tests for :mod:`decanter.calib.aperture`.

Includes a byte-for-byte parity test against WARP's ``apertureSet`` —
when the WARP source is importable, we load the same database file with
both implementations and assert the resulting ``apmask_array`` and
``slitcoord_array`` are numerically identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from decanter.calib.aperture import Aperture, ApertureSet
from decanter.io.apdb import ApertureEntry, FUNCTION_TYPE_CHEBYSHEV, FUNCTION_TYPE_LEGENDRE
from decanter._localpaths import WARP_ROOT


def _make_entry(**overrides: object) -> ApertureEntry:
    """Helper: synthetic ApertureEntry with sensible defaults."""
    defaults = dict(
        order=159,
        center_x=200.0,
        center_y=1024.0,
        low=-32.0,
        high=30.0,
        function_type=FUNCTION_TYPE_CHEBYSHEV,
        poly_order=2,
        y_min=1.0,
        y_max=2048.0,
        coefficients=(0.0, 5.0),  # linear in y_norm
        background_sample="INDEF",
    )
    defaults.update(overrides)
    return ApertureEntry(**defaults)  # type: ignore[arg-type]


def test_aperture_trace_is_constant_for_zero_coeffs() -> None:
    """If all coefficients are zero, trace is just center_x at every y."""
    ap = Aperture(entry=_make_entry(coefficients=(0.0, 0.0)), array_length=2048)
    assert np.allclose(ap.trace_x, 200.0)


def test_aperture_trace_linear_chebyshev() -> None:
    """A length-2 Chebyshev with c=[0, 5] gives x = center + 5 * y_norm.

    y_norm = (2y - y_min - y_max) / (y_max - y_min) ∈ [-1, 1].
    At y_min y_norm = -1 -> x = center - 5. At y_max y_norm = +1 -> x = center + 5.
    """
    ap = Aperture(
        entry=_make_entry(y_min=1.0, y_max=2048.0, coefficients=(0.0, 5.0)),
        array_length=2048,
    )
    assert ap.trace_x[0] == pytest.approx(200.0 - 5.0, abs=1e-2)
    assert ap.trace_x[-1] == pytest.approx(200.0 + 5.0, abs=1e-2)


def test_aperture_legendre_function_type() -> None:
    """Function-type 2 uses Legendre; for orders 0 and 1, T_n == P_n so result matches."""
    cheb = Aperture(entry=_make_entry(function_type=FUNCTION_TYPE_CHEBYSHEV))
    leg = Aperture(entry=_make_entry(function_type=FUNCTION_TYPE_LEGENDRE))
    # P_0 = T_0 = 1; P_1 = T_1 = y_norm. So both should agree.
    assert np.allclose(cheb.trace_x, leg.trace_x)


def test_unknown_function_type_raises() -> None:
    ap = Aperture(entry=_make_entry(function_type=99))
    with pytest.raises(ValueError, match="unknown function type"):
        _ = ap.trace_x


def test_apmask_array_shape_and_label_values() -> None:
    """Two non-overlapping apertures produce two distinct order labels."""
    entries = {
        159: _make_entry(order=159, center_x=200.0, coefficients=(0.0, 0.0)),
        160: _make_entry(order=160, center_x=300.0, coefficients=(0.0, 0.0)),
    }
    apset = ApertureSet(
        apertures={m: Aperture(entry=e) for m, e in entries.items()},
        array_length=2048,
    )
    mask = apset.apmask_array(low_lim=-30, upp_lim=30, margin=10)
    assert mask.shape == (2048, 2048)
    assert mask.dtype == np.int32
    # Inside order 159 aperture (around center_x=200), pixels should be labelled 159.
    assert mask[1024, 200] == 159
    assert mask[1024, 300] == 160
    # Outside any aperture (e.g., between them at x=250), pixel is 0.
    assert mask[1024, 250] == 0
    # Margin zero-out: matches WARP's asymmetric convention — margin=10
    # zeros 1-indexed pixels {1..9} at the lower edge (0-indexed {0..8})
    # and 1-indexed {L-9..L} at the upper edge (0-indexed {L-10..L-1}).
    assert np.all(mask[:9, :] == 0)
    assert np.all(mask[-10:, :] == 0)
    assert np.all(mask[:, :9] == 0)
    assert np.all(mask[:, -10:] == 0)


def test_slitcoord_array_signed_residue() -> None:
    """Inside the aperture, slitcoord is signed; outside, sentinel."""
    apset = ApertureSet(
        apertures={159: Aperture(entry=_make_entry(center_x=200.0, coefficients=(0.0, 0.0)))},
        array_length=2048,
    )
    sc = apset.slitcoord_array(low_lim=-30, upp_lim=30, inter_order_value=-10000.0)
    # At pixel x=200 (1-indexed: column 199, value 200 in WARP convention),
    # slitcoord should be ~ x_pixel - 200. Using 1-indexed coords from the
    # implementation: x[199] = 200 in 1-indexed terms, so residue = 0.
    # Outside any aperture (e.g., x=400), sentinel.
    assert sc[1024, 199] == pytest.approx(0.0, abs=0.5)
    assert sc[1024, 199 - 10] == pytest.approx(-10.0, abs=0.5)
    assert sc[1024, 199 + 10] == pytest.approx(10.0, abs=0.5)
    assert sc[1024, 400] == -10000.0  # sentinel outside aperture


def test_load_real_warp_hiresy() -> None:
    """Loading WARP's HIRES-Y flat database yields ~26 echelle orders."""
    real = WARP_ROOT / "reference/HIRES-Y/database/apflat_HIRESY_20170727_m"
    if not real.exists():
        pytest.skip(f"WARP reference data not present at {real}")
    apset = ApertureSet.load(real)
    assert len(apset.echelle_orders) >= 26
    # Trace must be the expected length and inside the detector.
    for m, ap in apset.apertures.items():
        assert ap.trace_x.shape == (2048,)
        assert ap.trace_x.min() > -100  # generous bounds — some apertures sit near the edge
        assert ap.trace_x.max() < 2148


@pytest.mark.parametrize(
    "low_lim,upp_lim",
    [
        (-30, 30),  # the values cosmicRayMask uses
        (None, None),  # use per-aperture bounds from the database
    ],
)
def test_parity_with_warp_apertureset(low_lim, upp_lim) -> None:
    """Byte-for-byte parity: decanter ApertureSet vs WARP apertureSet on the same DB.

    Skips if WARP isn't importable.
    """
    real = WARP_ROOT / "reference/HIRES-Y/database/apflat_HIRESY_20170727_m"
    if not real.exists():
        pytest.skip(f"WARP reference data not present at {real}")
    warp_root = WARP_ROOT
    if str(warp_root) not in sys.path:
        sys.path.insert(0, str(warp_root))
    try:
        from warp.aperture import apertureSet as WarpApertureSet
    except ImportError:
        pytest.skip("WARP source not importable")

    # WARP's apertureSet expects a relative path inside ./database/ — we have to
    # cd into the parent of the database dir for it to find the file.
    import os
    old_cwd = os.getcwd()
    os.chdir(real.parent.parent)  # one level up from database/
    try:
        # WARP's constructor reads "./database/ap<refname>"; refname is
        # everything after "ap" in the filename.
        refname = real.name[2:]  # strip leading "ap"
        warp_set = WarpApertureSet(refname)
    finally:
        os.chdir(old_cwd)

    decanter_set = ApertureSet.load(real)

    # Map WARP's lowlim/upplim kwargs (which use "INDEF" sentinel).
    warp_lowlim = "INDEF" if low_lim is None else low_lim
    warp_upplim = "INDEF" if upp_lim is None else upp_lim

    warp_mask = warp_set.apmaskArray(lowlim=warp_lowlim, upplim=warp_upplim)
    decanter_mask = decanter_set.apmask_array(low_lim=low_lim, upp_lim=upp_lim)

    # Allow int-vs-float dtype difference but require value equality.
    assert np.array_equal(warp_mask.astype(np.int32), decanter_mask), (
        f"apmask parity failed: {np.sum(warp_mask != decanter_mask)} pixels differ"
    )

    warp_sc = warp_set.slitcoordArray(lowlim=warp_lowlim, upplim=warp_upplim)
    decanter_sc = decanter_set.slitcoord_array(low_lim=low_lim, upp_lim=upp_lim)
    assert np.allclose(warp_sc, decanter_sc, atol=1e-6), (
        f"slitcoord parity failed: max diff {np.max(np.abs(warp_sc - decanter_sc))}"
    )
