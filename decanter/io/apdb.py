"""Parse IRAF ``database/ap*`` aperture files.

These files encode the per-order aperture trace polynomials (Chebyshev
or Legendre), aperture extents (``low``, ``high``), and background
sample regions (e.g. ``"-22:-12,8:18"``). decanter Phase 1 loads them
directly rather than re-deriving traces from data — see PLAN_FULL.md
§Validation binding constraint.

WARP equivalent: ``warp/aperture.py:apertureSet.__init__`` (lines 13-77).

Format reference: WARP's ``reference/HIRES-Y/database/apflat_HIRESY_20170727_m``
and neighbors. The ``ap*`` files are plain text, one or more aperture
blocks per file, each delimited by ``begin\\taperture`` lines.

Example aperture block::

    begin   aperture flat_HIRESY_20170727_m 159 197.116 1024.
        image   flat_HIRESY_20170727_m
        aperture        159
        beam    159
        center  197.116 1024.
        low     -32. -1023.
        high    30. 1025.
        background
            xmin -32.
            xmax 30.
            function chebyshev
            order 1
            sample INDEF
            ...
        axis    1
        curve   9
            1.          # function type: 1=Chebyshev, 2=Legendre
            5.          # polynomial order
            4.          # y_min (domain start)
            2044.       # y_max (domain end)
            -8.457813   # coefficient 0
            82.94567    # coefficient 1
            -8.489762   # coefficient 2
            -0.01563... # coefficient 3
            -0.00845... # coefficient 4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# IRAF function-type sentinels carried verbatim into the ``ApertureEntry``.
FUNCTION_TYPE_CHEBYSHEV: int = 1
FUNCTION_TYPE_LEGENDRE: int = 2


@dataclass(frozen=True, slots=True)
class ApertureEntry:
    """One aperture record from an IRAF ``database/ap*`` file.

    Attributes:
        order: echelle order number ``m``.
        center_x: aperture center along the cross-dispersion axis (1-indexed pixels).
        center_y: aperture center along the dispersion axis (1-indexed pixels).
        low: signed lower bound of the aperture, in pixels relative to ``center_x``
            (typically negative).
        high: signed upper bound of the aperture, in pixels relative to ``center_x``
            (typically positive).
        function_type: 1 = Chebyshev, 2 = Legendre. See :data:`FUNCTION_TYPE_CHEBYSHEV`
            and :data:`FUNCTION_TYPE_LEGENDRE`.
        poly_order: number of polynomial coefficients (degree + 1 in IRAF convention).
        y_min: dispersion-axis domain start. The trace polynomial is evaluated on
            ``y_norm = (2y - y_min - y_max) / (y_max - y_min)`` ∈ [-1, 1].
        y_max: dispersion-axis domain end.
        coefficients: polynomial coefficients of length ``poly_order``.
        background_sample: IRAF ``sample`` string for background fitting, e.g.
            ``"-22:-12,8:18"`` or ``"INDEF"``.
    """

    order: int
    center_x: float
    center_y: float
    low: float
    high: float
    function_type: int
    poly_order: int
    y_min: float
    y_max: float
    coefficients: tuple[float, ...]
    background_sample: str = "INDEF"


def parse(path: Path | str) -> dict[int, ApertureEntry]:
    """Parse an IRAF ``database/ap*`` file.

    Args:
        path: path to the database file (e.g.
            ``reference/HIRES-Y/database/apflat_HIRESY_20170727_m``).

    Returns:
        ``{order: ApertureEntry}`` for every aperture block in the file.
        Later blocks for the same order *overwrite* earlier ones (matches
        WARP's "last definition wins" behavior — see ``apertureSet.__init__``
        comment "search the last definition in the aperture file").

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if a block has malformed ``curve``, ``center``, ``low``,
            ``high``, or coefficient data.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    entries: dict[int, ApertureEntry] = {}

    block_starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("begin")]
    block_starts.append(len(lines))  # sentinel for the last block's end

    for block_idx in range(len(block_starts) - 1):
        block = lines[block_starts[block_idx]: block_starts[block_idx + 1]]
        entry = _parse_block(block)
        entries[entry.order] = entry  # later definitions overwrite earlier

    return entries


def _parse_block(block: list[str]) -> ApertureEntry:
    """Parse one ``begin aperture ... `` block into an :class:`ApertureEntry`."""
    # The "begin" line: "begin\taperture\t<image>\t<order>\t<centerx>\t<centery>"
    begin_tokens = block[0].split()
    # begin_tokens[0] = "begin", [1] = "aperture", [2] = image name, [3] = order, ...
    if len(begin_tokens) < 6:
        raise ValueError(f"malformed begin line: {block[0]!r}")
    order = int(begin_tokens[3])

    center_x: float | None = None
    center_y: float | None = None
    low: float | None = None
    high: float | None = None
    background_sample: str = "INDEF"
    function_type: int | None = None
    poly_order: int | None = None
    y_min: float | None = None
    y_max: float | None = None
    coefficients: tuple[float, ...] | None = None

    i = 0
    while i < len(block):
        stripped = block[i].lstrip()
        if stripped.startswith("center"):
            parts = stripped.split()
            center_x = float(parts[1])
            center_y = float(parts[2])
            # Next two lines are "low" and "high" per WARP's parser convention.
            low = float(block[i + 1].split()[1])
            high = float(block[i + 2].split()[1])
        elif stripped.startswith("sample"):
            # Only the *first* "sample" we see is the background sample
            # (inside the background block). The curve sub-block doesn't
            # have a sample line.
            parts = stripped.split(maxsplit=1)
            if background_sample == "INDEF" and len(parts) > 1:
                background_sample = parts[1].strip()
        elif stripped.startswith("curve"):
            curve_count = int(stripped.split()[1])
            # IRAF curve block layout:
            #   ftype, yorder, ymin, ymax, then yorder coefficients.
            #   curve_count == 4 + yorder.
            curve_lines = block[i + 1: i + 1 + curve_count]
            if len(curve_lines) != curve_count:
                raise ValueError(
                    f"order {order}: expected {curve_count} curve lines, got {len(curve_lines)}"
                )
            function_type = int(float(curve_lines[0].split()[0]))
            poly_order = int(float(curve_lines[1].split()[0]))
            y_min = float(curve_lines[2].split()[0])
            y_max = float(curve_lines[3].split()[0])
            coefficients = tuple(float(line.split()[0]) for line in curve_lines[4: 4 + poly_order])
            if len(coefficients) != poly_order:
                raise ValueError(
                    f"order {order}: expected {poly_order} coefficients, got {len(coefficients)}"
                )
        i += 1

    if center_x is None or center_y is None or low is None or high is None:
        raise ValueError(f"order {order}: missing center / low / high")
    if function_type is None or poly_order is None or y_min is None or y_max is None or coefficients is None:
        raise ValueError(f"order {order}: missing curve block")

    return ApertureEntry(
        order=order,
        center_x=center_x,
        center_y=center_y,
        low=low,
        high=high,
        function_type=function_type,
        poly_order=poly_order,
        y_min=y_min,
        y_max=y_max,
        coefficients=coefficients,
        background_sample=background_sample,
    )
