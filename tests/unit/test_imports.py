"""Smoke tests — every module must import without error.

This catches broken imports the moment they're introduced, before any
real test runs. Useful while the package is still mostly stubs.
"""

from __future__ import annotations

import importlib
import pkgutil

import decanter


def test_top_level_import() -> None:
    """``import decanter`` exposes a version string."""
    assert isinstance(decanter.__version__, str)


def test_every_submodule_imports() -> None:
    """Walk the package and import every submodule."""
    failures: list[tuple[str, Exception]] = []
    for module_info in pkgutil.walk_packages(decanter.__path__, prefix="decanter."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 — we surface every failure
            failures.append((module_info.name, exc))
    assert not failures, "submodules failed to import: " + ", ".join(
        f"{name} ({type(exc).__name__}: {exc})" for name, exc in failures
    )


def test_pipeline_module_lists_active_stages() -> None:
    """:data:`decanter.pipeline.STAGES` lists the 13 active stages (s01..s13).

    s00 (calib loader) is replaced by :class:`Calibration.from_dir`;
    s14..s19 (downstream stubs) were deleted in the Phase-2 cleanup
    because they're handled downstream by the user with different
    techniques. New code should prefer :func:`decanter.reduce` over
    iterating STAGES directly.
    """
    from decanter.pipeline import STAGES
    assert len(STAGES) == 13, f"expected 13 active stages, got {len(STAGES)}"
