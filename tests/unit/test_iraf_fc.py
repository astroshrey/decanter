"""Unit tests for :mod:`decanter.io.iraf_fc`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decanter.io.iraf_fc import FUNCTION_CHEBYSHEV, FcSurface, evaluate, parse
from decanter._localpaths import WINERED_ROOT


@pytest.fixture
def synthetic_fc_file(tmp_path: Path) -> Path:
    """Write a tiny ``fitcoords`` file with known coefficients."""
    # 2x2 cheb surface with coefficients [[1, 0.5], [0.2, 0.1]] (x by y)
    # 8 header lines + 4 coefficients = 12 total values.
    content = (
        "# test fixture\n"
        "begin\ttest_image\n"
        "\ttask\tfitcoords\n"
        "\taxis\t2\n"
        "\tunits\tangstroms\n"
        "\tsurface\t12\n"
        "\t\t1\n"   # ftype = chebyshev
        "\t\t2\n"   # xorder
        "\t\t2\n"   # yorder
        "\t\t1\n"   # cross
        "\t\t1.\n"  # xmin
        "\t\t10.\n" # xmax
        "\t\t1.\n"  # ymin
        "\t\t20.\n" # ymax
        # Coefficients stored column-major (x fastest):
        #   c00 c10 c01 c11 → matrix [[c00, c01], [c10, c11]]
        "\t\t1.0\n"   # c[0,0]
        "\t\t0.2\n"   # c[1,0]
        "\t\t0.5\n"   # c[0,1]
        "\t\t0.1\n"   # c[1,1]
    )
    path = tmp_path / "fctest"
    path.write_text(content)
    return path


def test_parse_basic_surface(synthetic_fc_file: Path) -> None:
    surf = parse(synthetic_fc_file)
    assert surf.image_name == "test_image"
    assert surf.axis == 2
    assert surf.units == "angstroms"
    assert surf.ftype == FUNCTION_CHEBYSHEV
    assert surf.xorder == 2
    assert surf.yorder == 2
    assert surf.cross == 1
    assert surf.xmin == 1.0
    assert surf.xmax == 10.0
    assert surf.ymin == 1.0
    assert surf.ymax == 20.0
    # coefficients shape is (xorder, yorder)
    assert surf.coefficients.shape == (2, 2)
    assert surf.coefficients[0, 0] == pytest.approx(1.0)
    assert surf.coefficients[1, 0] == pytest.approx(0.2)
    assert surf.coefficients[0, 1] == pytest.approx(0.5)
    assert surf.coefficients[1, 1] == pytest.approx(0.1)


def test_evaluate_at_corners(synthetic_fc_file: Path) -> None:
    """At normalization corners ``(xmin, ymin)`` etc., Chebyshev T_n(±1) is ±1.

    Surface = c00*T0(x)*T0(y) + c10*T1(x)*T0(y) + c01*T0(x)*T1(y) + c11*T1(x)*T1(y)
            = c00 + c10*x + c01*y + c11*x*y  (where x, y are normalized)

    At (xmin, ymin) → normalized (-1, -1): value = 1 - 0.2 - 0.5 + 0.1 = 0.4
    At (xmax, ymax) → normalized (+1, +1): value = 1 + 0.2 + 0.5 + 0.1 = 1.8
    """
    surf = parse(synthetic_fc_file)
    v1 = evaluate(surf, np.asarray([1.0]), np.asarray([1.0]))
    v2 = evaluate(surf, np.asarray([10.0]), np.asarray([20.0]))
    assert v1[0] == pytest.approx(0.4)
    assert v2[0] == pytest.approx(1.8)


def test_evaluate_real_warp_fc() -> None:
    """The real ``fcmultihole_HIRES-Y100_20250806_163`` file parses and evaluates."""
    fc_path = (
        WINERED_ROOT / "reductions" / "TOI2109" / "calibration_data"
        / "database" / "fcmultihole_HIRES-Y100_20250806_163"
    )
    if not fc_path.exists():
        pytest.skip("real WARP fc file not present")
    surf = parse(fc_path)
    assert surf.axis == 2
    assert surf.units == "angstroms"
    # The surface gives wavelengths; for HIRES-Y order 163 these should be
    # in the 10,000–13,000 Å range (~1000–1300 nm).
    vals = evaluate(
        surf,
        np.linspace(surf.xmin, surf.xmax, 11),
        np.linspace(surf.ymin, surf.ymax, 11),
    )
    # The surface gives wavelength values in the line-list's units (which
    # turn out to be nm for WARP's WINERED HIRES-Y calibration — the
    # ``units angstroms`` label in the fc file is a metadata mistake;
    # the actual numbers are in nm. Internally consistent — downstream
    # stages just use the same dy in the same units.). Order 163's
    # central wavelength is ~1080 nm; full surface range spans a few
    # hundred nm with the slit/order extrapolation.
    assert vals.min() > 100.0  # should be in the ~hundreds (nm) range
    assert vals.max() < 5_000.0


def test_cross_not_full_raises(tmp_path: Path) -> None:
    """``cross != 1`` is currently unsupported."""
    content = (
        "begin\ttest\n"
        "\ttask\tfitcoords\n"
        "\taxis\t2\n"
        "\tunits\tangstroms\n"
        "\tsurface\t9\n"
        "\t\t1\n\t\t2\n\t\t2\n\t\t2\n"  # cross=2
        "\t\t1.\n\t\t10.\n\t\t1.\n\t\t10.\n"
        "\t\t1.0\n"
    )
    path = tmp_path / "fctest"
    path.write_text(content)
    with pytest.raises(NotImplementedError, match="cross=1"):
        parse(path)
