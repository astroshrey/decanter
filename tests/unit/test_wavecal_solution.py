"""The wavecal solution container, and the exactness of how it is applied."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decanter._reduction import OrderSpectrum, Reduction
from decanter.wavecal import C_KMS, WavecalSolution
from decanter.wavecal.solve import assemble_hybrid_ladder


def _spectrum(order: int = 163, n: int = 64) -> OrderSpectrum:
    return OrderSpectrum(
        order=order, fsr_cut=1.30, flux=np.arange(n, dtype=np.float32),
        crval1=10000.0, cdelt1=0.03, crpix1=1.0,
    )


def _reduction(frame_id: str = "WINA00000001", orders=(163, 164)) -> Reduction:
    return Reduction(
        obj_name="TARGET", obj_path=None, sky_path=None,
        obj={(1.30, m): _spectrum(m) for m in orders},
        sky={(1.30, m): _spectrum(m) for m in orders},
        meta={"OBJFRAME": frame_id},
    )


def _solution(velocity_kms: float, orders=(163, 164)) -> WavecalSolution:
    velocity = np.full((1, len(orders)), velocity_kms)
    return WavecalSolution(
        frame_ids=("WINA00000001",),
        orders=tuple(orders),
        velocity=velocity,
        source=np.full(velocity.shape, "telluric", dtype="U16"),
        bracketed=np.ones(velocity.shape, dtype=bool),
    )


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="expected"):
        WavecalSolution(
            frame_ids=("a", "b"), orders=(159,),
            velocity=np.zeros((1, 1)),
            source=np.full((1, 1), "telluric", dtype="U16"),
            bracketed=np.zeros((1, 1), dtype=bool),
        )


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        WavecalSolution.empty(["a", "a"], [159])


def test_applying_a_velocity_rescales_the_wcs_exactly() -> None:
    """A pure Doppler rescaling maps a linear grid onto a linear grid.

    The corrected wavelength array must equal lambda/(1 + v/c) to machine
    precision at every pixel -- not just at the reference pixel, which is what
    a constant angstrom offset would give.
    """
    velocity = 2.0
    reduction = _reduction()
    corrected = _solution(velocity).apply(reduction)

    for key, spec in reduction.obj.items():
        expected = spec.wavelength / (1.0 + velocity / C_KMS)
        got = corrected.obj[key].wavelength
        assert np.allclose(got, expected, rtol=0, atol=1e-9)


def test_a_constant_angstrom_shift_would_not_be_good_enough() -> None:
    """Guard the reason for rescaling instead of offsetting.

    Over one order the wavelength changes by ~1%, so a rescaling and a constant
    offset differ by ~1% of the correction -- tens of m/s for a km/s-sized
    correction, which is above the precision this module targets.
    """
    velocity = 2.0
    spec = _spectrum(n=2000)
    exact = spec.wavelength / (1.0 + velocity / C_KMS)
    constant_offset = spec.wavelength - spec.wavelength[0] * velocity / C_KMS
    difference_kms = (exact - constant_offset) / spec.wavelength * C_KMS
    assert np.max(np.abs(difference_kms)) * 1e3 > 5.0   # m/s


def test_flux_is_untouched_and_shared() -> None:
    reduction = _reduction()
    corrected = _solution(1.0).apply(reduction)
    for key, spec in reduction.obj.items():
        assert corrected.obj[key].flux is spec.flux


def test_sky_is_corrected_with_the_same_velocity() -> None:
    reduction = _reduction()
    corrected = _solution(3.0).apply(reduction)
    for key in reduction.sky:
        assert corrected.sky[key].crval1 == pytest.approx(corrected.obj[key].crval1)


def test_zero_velocity_is_a_no_op() -> None:
    reduction = _reduction()
    corrected = _solution(0.0).apply(reduction)
    for key, spec in reduction.obj.items():
        assert corrected.obj[key].crval1 == pytest.approx(spec.crval1)
        assert corrected.obj[key].cdelt1 == pytest.approx(spec.cdelt1)


def test_nan_velocity_passes_the_order_through_and_records_it() -> None:
    solution = _solution(1.5)
    solution.velocity[0, 1] = np.nan
    reduction = _reduction()
    corrected = solution.apply(reduction)

    assert corrected.obj[(1.30, 164)].crval1 == pytest.approx(10000.0)
    assert corrected.obj[(1.30, 163)].crval1 != pytest.approx(10000.0)
    assert corrected.meta["WAVECBAD"] == "164"


def test_strict_raises_on_a_missing_order() -> None:
    solution = _solution(1.5)
    solution.velocity[0, 1] = np.nan
    with pytest.raises(ValueError, match="no wavecal solution"):
        solution.apply(_reduction(), strict=True)


def test_unknown_frame_raises() -> None:
    with pytest.raises(KeyError, match="not in this solution"):
        _solution(1.0).apply(_reduction(frame_id="WINA99999999"))


def test_provenance_lands_in_the_output_metadata() -> None:
    corrected = _solution(1.25).apply(_reduction())
    assert corrected.meta["WAVECAL"] == "hybrid_refit"
    assert corrected.meta["WAVECZP"] == "absolute"
    assert corrected.meta["WAVECMED"] == pytest.approx(1.25)
    assert corrected.meta["OBJFRAME"] == "WINA00000001"


def test_roundtrip_through_npz(tmp_path: Path) -> None:
    solution = _solution(0.75)
    solution.meta["species"] = ["H2O", "CH4", "O2"]
    path = tmp_path / "solution.npz"
    solution.save_npz(path)
    loaded = WavecalSolution.load_npz(path)

    assert loaded.frame_ids == solution.frame_ids
    assert loaded.orders == solution.orders
    assert np.array_equal(loaded.velocity, solution.velocity)
    assert loaded.mode == solution.mode
    assert loaded.meta["species"] == ["H2O", "CH4", "O2"]


def test_coverage_reports_the_source_mix() -> None:
    solution = _solution(1.0)
    solution.source[0, 1] = "interpolated"
    coverage = solution.coverage()
    assert coverage["telluric"] == pytest.approx(0.5)
    assert coverage["interpolated"] == pytest.approx(0.5)


def test_hybrid_ladder_matches_notebook_priority_exactly() -> None:
    """Telluric > direct strong OH > interpolation, with direct-only anchors."""
    orders = np.arange(159, 164)
    telluric_velocity = np.array([[10.0, np.nan, np.nan, np.nan, 50.0]])
    telluric_accepted = np.array([[True, False, False, False, True]])
    # The first OH value conflicts with the telluric value: it must neither
    # win nor enter the smooth anchor fit.
    oh_velocity = np.array([[999.0, np.nan, 30.0, np.nan, np.nan]])
    oh_accepted = np.array([[True, False, True, False, False]])
    weight = np.ones_like(telluric_velocity)

    velocity, source, bracketed = assemble_hybrid_ladder(
        orders,
        telluric_velocity,
        telluric_accepted,
        weight,
        oh_velocity,
        oh_accepted,
        weight,
    )

    np.testing.assert_allclose(velocity[0], [10.0, 20.0, 30.0, 40.0, 50.0], atol=1e-9)
    assert source[0].tolist() == [
        "telluric", "interpolated", "OH", "interpolated", "telluric"
    ]
    assert bracketed[0].all()
