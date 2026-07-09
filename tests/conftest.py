"""Shared pytest fixtures.

Anything that wants to short-circuit when WARP reference data is not
available should consume :func:`warp_reference_dir` and skip if it
returns ``None``. That lets unit tests run anywhere while regression
tests automatically skip on machines without the reference reduction.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from decanter._localpaths import WINERED_ROOT

# The WARP `-s` reference reduction (all intermediates saved) — the
# parity target described in tests/regression/README.md. Override with
# $DECANTER_WARP_REF if it lives somewhere non-standard.
WARP_REFERENCE_PATH = Path(
    os.environ.get(
        "DECANTER_WARP_REF",
        str(WINERED_ROOT / "reductions" / "TOI2109_decanterref"),
    )
)


@pytest.fixture(scope="session")
def warp_reference_dir() -> Path:
    """Return the path to the reference TOI2109 WARP reduction, or skip."""
    if not WARP_REFERENCE_PATH.is_dir():
        pytest.skip(f"WARP reference reduction not found at {WARP_REFERENCE_PATH}")
    return WARP_REFERENCE_PATH
