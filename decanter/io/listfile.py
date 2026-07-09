"""Parse WARP-style input lists (object / sky frame pairs).

Format (one pair per line, whitespace-separated, ``#`` comments OK)::

    OBJECT_NAME SKY_NAME [ap=LO:HI] [bg=REGION] [ws=SHIFT]

Example from a real TOI2109 reduction (``TOI2109.txt``)::

    WINA00053571 WINA00053572
    WINA00053572 WINA00053571
    WINA00053573 WINA00053574

Optional ``ap=``, ``bg=``, ``ws=`` tokens carry per-frame overrides
(manual aperture, background sample region, manual wavelength shift)
that downstream stages consume when their corresponding ``Config``
flags are set.

WARP equivalent: ``warp/config.py:config.inputDataList`` (lines 79-156).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FramePair:
    """One row of a WARP-style input listfile.

    Attributes:
        object_name: raw frame name (without ``.fits`` suffix).
        sky_name: paired sky frame name (without ``.fits`` suffix).
        aperture_low: optional manual aperture lower bound (pixels from center).
        aperture_high: optional manual aperture upper bound (pixels from center).
        background_region: optional background sample-region string, e.g.
            ``"-22:-12,8:18"``.
        manual_shift: optional manual wavelength shift (units depend on
            WARP's convention — see PLAN_FULL.md Spike C).
    """

    object_name: str
    sky_name: str
    aperture_low: float | None = None
    aperture_high: float | None = None
    background_region: str | None = None
    manual_shift: float | None = None


def parse(path: Path | str) -> list[FramePair]:
    """Read a WARP-style input listfile.

    Args:
        path: path to the listfile.

    Returns:
        A list of :class:`FramePair`, one per non-blank/non-comment line.

    Raises:
        ValueError: if a line has fewer than two whitespace-separated tokens.
        FileNotFoundError: if the listfile is missing.
    """
    pairs: list[FramePair] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError(
                f"{path}:{lineno}: expected at least 'OBJECT SKY' tokens, got {line!r}"
            )
        kwargs: dict[str, object] = {}
        for tok in tokens[2:]:
            if tok.startswith("ap="):
                lo_s, hi_s = tok[len("ap="):].split(":", 1)
                kwargs["aperture_low"] = float(lo_s)
                kwargs["aperture_high"] = float(hi_s)
            elif tok.startswith("bg="):
                kwargs["background_region"] = tok[len("bg="):]
            elif tok.startswith("ws="):
                kwargs["manual_shift"] = float(tok[len("ws="):])
            # Silently ignore unknown tokens — WARP does too.
        pairs.append(FramePair(object_name=tokens[0], sky_name=tokens[1], **kwargs))  # type: ignore[arg-type]
    return pairs
