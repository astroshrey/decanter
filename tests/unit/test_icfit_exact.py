"""Unit tests for :mod:`decanter.utils.icfit_exact` (bit-exact REAL icfit)."""

from __future__ import annotations

import numpy as np

from decanter.utils import icfit_exact as icx
from decanter.utils.fpemu import fma32

R = np.float32


def _same_bits(a, b) -> bool:
    a = np.asarray(a, R)
    b = np.asarray(b, R)
    return a.shape == b.shape and bool(np.all(a.view(np.int32) == b.view(np.int32)))


def _adversarial(rng, n):
    """Operands whose sums often sit on float32 rounding midpoints."""
    a = rng.choice([1, -1, 0.5, 2, 3], n).astype(R)
    b = (rng.integers(-2**20, 2**20, n) * R(2.0 ** -30)).astype(R)
    c = (R(1.0) + rng.integers(0, 2**10, n) * R(2.0 ** -23)).astype(R)
    return a, b, c


def _double_rounding_traps(rng, n):
    """Triples where RN32(RN64(a*b + c)) != RN32(a*b + c).

    ``a*b = 2**e * (1 - k**2 * 2**-46)`` lands just below the float32
    midpoint ``c + 2**e`` (``c`` has an odd last bit), so float64 rounds onto
    the midpoint and ties-to-even then rounds the wrong way.
    """
    k = rng.integers(1, 64, n)
    m = rng.integers(0, 2**22, n) * 2 + 1                  # odd mantissa
    sc = (2.0 ** rng.integers(-20, 20, n)).astype(R)
    sg = rng.choice([-1, 1], n).astype(R)
    a = (R(1) + k * R(2.0 ** -23)).astype(R)
    b = ((R(1) - k * R(2.0 ** -23)) * R(2.0 ** -24) * sc * sg).astype(R)
    c = ((R(1) + m * R(2.0 ** -23)) * sc * sg).astype(R)
    return a, b, c


def test_fma32f_matches_error_free_fma() -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    a = (rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)).astype(R)
    b = (rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)).astype(R)
    c = (rng.standard_normal(n) * 10.0 ** rng.integers(-3, 8, n)).astype(R)
    assert _same_bits(icx.fma32f(a, b, c), fma32(a, b, c))
    a, b, c = _adversarial(rng, n)
    assert _same_bits(icx.fma32f(a, b, c), fma32(a, b, c))
    a, b, c = _double_rounding_traps(rng, 10_000)
    naive = (np.multiply(a, b, dtype=np.float64) + c).astype(R)
    exact = fma32(a, b, c)
    assert np.all(naive != exact)          # the traps really trap
    assert _same_bits(icx.fma32f(a, b, c), exact)
    # 0-d and broadcasting
    assert _same_bits(icx.fma32f(R(3), R(5), R(7)), fma32(R(3), R(5), R(7)).reshape(()))
    assert icx.fma32f(a[:, None], b[None, :5], R(1)).shape == (a.size, 5)


def _naive_chain(a, b, valid):
    acc = np.zeros(a.shape[1:], dtype=R)
    for p in range(a.shape[0]):
        acc = np.where(valid[p], fma32(a[p], b[p], acc), acc)
    return acc


def test_fma_chain_matches_sequential_fma() -> None:
    rng = np.random.default_rng(1)
    for kind in ("random", "adversarial"):
        npts, nfit = 300, 64
        if kind == "random":
            a = rng.standard_normal((npts, nfit)).astype(R)
            b = (rng.standard_normal((npts, nfit)) * 100).astype(R)
        else:
            a, b, c = _adversarial(rng, npts * nfit)
            a = a.reshape(npts, nfit)
            b = (b + c).reshape(npts, nfit)
        valid = rng.random((npts, nfit)) > 0.2
        assert _same_bits(icx.fma_chain(a, b, valid), _naive_chain(a, b, valid))
    # Plant a double-rounding trap at a random step of each chain.
    npts, nfit = 300, 64
    a = rng.standard_normal((npts, nfit)).astype(R) * R(1e-12)
    b = rng.standard_normal((npts, nfit)).astype(R)
    ta, tb, tc = _double_rounding_traps(rng, nfit)
    at = rng.integers(1, npts - 1, nfit)
    for f in range(nfit):
        a[:at[f], f] = 0
        a[at[f] - 1, f], b[at[f] - 1, f] = tc[f], 1     # acc becomes c
        a[at[f], f], b[at[f], f] = ta[f], tb[f]         # then the trap
    valid = np.ones((npts, nfit), dtype=bool)
    assert _same_bits(icx.fma_chain(a, b, valid), _naive_chain(a, b, valid))
    assert _same_bits(icx.fma_chain(a, b, None), _naive_chain(a, b, valid))


def test_cvfit_recovers_polynomial() -> None:
    x = np.arange(10, 2001, dtype=R)[:, None]
    t = (x - 1005) / 995
    y = (3 + 2 * t - t * t).astype(R)
    valid = np.ones_like(x, dtype=bool)
    cv, ier = icx.cvfit(x, y, np.ones_like(y), valid, 3, x[0, 0], x[-1, 0])
    assert int(ier[0]) == icx.OK
    fit = icx.cvvector(cv.coeff[:, None, :], x, cv.k1[None, :], cv.k2[None, :])
    assert np.max(np.abs(fit - y)) < 1e-4
