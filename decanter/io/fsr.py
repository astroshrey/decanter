"""Parse WARP's Free Spectral Range tables.

Source files: ``warp/FSR/FSR_winered_*.txt`` (one row per echelle order
per WINERED mode, columns: ``order``, ``lambda_min``, ``lambda_max`` in
vacuum Å). Used by stage 13 to clip each order to its FSR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FSREntry:
    """One row of the FSR table."""

    order: int
    lambda_min: float  # vacuum Å
    lambda_max: float  # vacuum Å


def load(path: Path | str) -> dict[int, FSREntry]:
    """Read an FSR file, indexed by echelle order.

    Args:
        path: path to ``FSR_winered_*.txt``.

    Returns:
        ``{order: FSREntry}`` for every row in the file.
    """
    entries: dict[int, FSREntry] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Skip headers like "WIDE: m=42--61"
        if len(parts) < 3:
            continue
        try:
            order = int(parts[0])
            lam_min = float(parts[1])
            lam_max = float(parts[2])
        except ValueError:
            continue
        entries[order] = FSREntry(order=order, lambda_min=lam_min, lambda_max=lam_max)
    return entries
