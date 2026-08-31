"""The four public wavecal modes must dispatch to distinct reference paths."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from decanter.wavecal.config import WavecalConfig
from decanter.wavecal.solve import _to_object_epoch, solve


class _RichModel(SimpleNamespace):
    def rich_orders(self, threshold):
        del threshold
        return np.asarray(self._rich, dtype=bool)


def _series():
    n_frames, n_pixels, n_orders = 2, 101, 3
    wave = np.column_stack([
        np.linspace(10_000.0 + 100 * j, 10_010.0 + 100 * j, n_pixels)
        for j in range(n_orders)
    ])
    return SimpleNamespace(
        n_frames=n_frames,
        n_pixels=n_pixels,
        n_orders=n_orders,
        frame_ids=("frame1", "frame2"),
        orders=(10, 11, 12),
        wave=wave,
        dv_pix_kms=np.ones(n_orders),
        obj=np.ones((n_frames, n_pixels, n_orders)),
        sky=np.ones((n_frames, n_pixels, n_orders)),
        time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([1.0, 2.0]),
        instmode="HIRES-Y",
        fsr_cut=1.3,
    )


def _oh_model():
    return _RichModel(
        _rich=[True, False, True],
        line_count=np.asarray([5, 0, 5]),
        rotational_temperature_k=200.0,
    )


@pytest.mark.parametrize(
    "mode, uses_telluric, refits",
    [
        ("oh_static", False, False),
        ("oh_refit", False, True),
        ("hybrid_static", True, False),
        ("hybrid_refit", True, True),
    ],
)
def test_mode_properties(mode, uses_telluric, refits) -> None:
    config = WavecalConfig(mode=mode)
    assert config.uses_telluric is uses_telluric
    assert config.per_exposure_refit is refits


def test_oh_only_skips_tellurics_and_uses_only_oh_anchors(monkeypatch) -> None:
    solve_module = importlib.import_module("decanter.wavecal.solve")

    def forbidden(*args, **kwargs):
        raise AssertionError("OH-only mode must not build or fit tellurics")

    monkeypatch.setattr(solve_module, "series_tau", forbidden)
    monkeypatch.setattr(solve_module, "fit_templates", forbidden)
    monkeypatch.setattr(
        solve_module,
        "measure_oh",
        lambda series, config, verbose: (
            _oh_model(),
            np.asarray([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]]),
            np.asarray([[0.9, np.nan, 0.9], [0.9, np.nan, 0.9]]),
        ),
    )

    solution = solve(_series(), WavecalConfig(mode="oh_static"), verbose=False)

    assert not np.any(solution.source == "telluric")
    np.testing.assert_array_equal(
        solution.source,
        np.asarray([
            ["OH", "interpolated", "OH"],
            ["OH", "interpolated", "OH"],
        ]),
    )
    assert np.isnan(solution.oh_tie_kms)


def test_hybrid_mode_prioritizes_telluric_then_oh(monkeypatch) -> None:
    solve_module = importlib.import_module("decanter.wavecal.solve")

    telluric_model = _RichModel(
        _rich=[True, False, False],
        native_template=np.ones((101, 3)),
        template_rms=np.asarray([0.1, 0.0, 0.0]),
        lsf_fwhm_kms=np.ones(3),
        parameters=np.zeros((3, 10)),
        family=np.asarray(["gaussian"] * 3),
    )
    monkeypatch.setattr(
        solve_module, "series_tau",
        lambda *args, **kwargs: (np.zeros((3, 101, 3)), {}),
    )
    monkeypatch.setattr(
        solve_module, "fit_templates",
        lambda series, tau, config, verbose: telluric_model,
    )
    monkeypatch.setattr(
        solve_module, "measure_series_shifts",
        lambda *args, **kwargs: (
            np.asarray([[1.0, np.nan, np.nan], [2.0, np.nan, np.nan]]),
            np.asarray([[0.9, np.nan, np.nan], [0.9, np.nan, np.nan]]),
        ),
    )
    monkeypatch.setattr(
        solve_module, "measure_oh",
        lambda series, config, verbose: (
            _oh_model(),
            np.asarray([[8.0, np.nan, 3.0], [9.0, np.nan, 4.0]]),
            np.asarray([[0.9, np.nan, 0.9], [0.9, np.nan, 0.9]]),
        ),
    )

    solution = solve(
        _series(),
        WavecalConfig(mode="hybrid_static", oh_tie="none"),
        verbose=False,
    )

    np.testing.assert_array_equal(
        solution.source,
        np.asarray([
            ["telluric", "interpolated", "OH"],
            ["telluric", "interpolated", "OH"],
        ]),
    )
    np.testing.assert_allclose(solution.velocity[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(solution.velocity[:, 2], [3.0, 4.0])


def test_automatic_mode_defaults_follow_instrument_band() -> None:
    config = WavecalConfig()
    assert config.mode == "auto"
    assert config.resolved_for("HIRES-Y").mode == "hybrid_refit"
    assert config.resolved_for("HIRES-J").mode == "hybrid_refit"
    assert config.resolved_for("WIDE").mode == "hybrid_static"


def test_automatic_mode_requires_a_known_instrument_band() -> None:
    with pytest.raises(ValueError, match="cannot choose an automatic"):
        WavecalConfig().resolved_for("UNKNOWN")


def test_oh_object_epoch_uses_paired_sky_warp_shift() -> None:
    series = SimpleNamespace(
        n_frames=2,
        frame_ids=("object-a", "sky-a"),
        dv_pix_kms=np.asarray([0.8, 1.2]),
        time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([2.0, 1.0]),
        meta=[
            {"OBJFRAME": "object-a", "SKYFRAME": "sky-a", "WAVSHIFT": 0.5},
            {"OBJFRAME": "sky-a", "SKYFRAME": "object-a", "WAVSHIFT": -0.25},
        ],
    )
    raw = np.asarray([[10.0, 20.0], [30.0, 40.0]])

    corrected = _to_object_epoch(raw, series, np.asarray([100.0, -100.0]))

    np.testing.assert_allclose(
        corrected,
        np.asarray([
            [10.2, 20.3],
            [29.6, 39.4],
        ]),
    )


def test_oh_object_epoch_falls_back_without_warp_provenance() -> None:
    series = SimpleNamespace(
        n_frames=2,
        frame_ids=("one", "two"),
        dv_pix_kms=np.asarray([1.0]),
        time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([2.0, 1.0]),
    )
    raw = np.asarray([[10.0], [20.0]])

    corrected = _to_object_epoch(raw, series, np.asarray([1.0, 3.0]))

    np.testing.assert_allclose(corrected, np.asarray([[8.0], [22.0]]))
