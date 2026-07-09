"""Stage 1 — sky subtraction (A/B nod frame-pair difference).

WARP equivalent: ``Warp_sci.py:336`` — literally one line:
``iraf.imarith(obj, '-', sky, output)``. Pixel-by-pixel subtraction
of the paired sky frame from the object frame. No scaling, no master
sky, no header munging beyond what IRAF auto-applies.

Output suffix: ``_s``.
Parity target: bit-identical to WARP modulo the header allow-list
in ``decanter.io.headers.ALLOWED_HEADER_KEYS`` (see PLAN_FULL.md
§Validation tolerance table).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile

# Characters that WARP's `readInputDataHeader` strips when sanitizing
# the OBJECT header into a filename component (warp/config.py:177).
_OBJNAME_BAD_CHARS: tuple[str, ...] = (" ", "'", "\"", "#", "/")


def _sanitize_objname(raw: str) -> str:
    """Sanitize the OBJECT header value into a filename component.

    WARP equivalent: ``warp/config.py:177`` (the chained ``replace``
    calls in ``readInputDataHeader``).
    """
    name = raw
    for ch in _OBJNAME_BAD_CHARS:
        name = name.replace(ch, "_")
    return name


def sky_subtract(obj: NDArray, sky: NDArray) -> NDArray:
    """Pure task function: ``obj - sky`` element-wise.

    Mirrors ``iraf.imarith(obj, '-', sky, output)`` for the
    no-scaling, no-mask sky subtraction WARP performs at
    ``Warp_sci.py:336``. Inputs must have matching shape.
    """
    if obj.shape != sky.shape:
        raise ValueError(f"shape mismatch: obj={obj.shape} sky={sky.shape}")
    return obj - sky


def run(config: Config, workdir: Path, listfile: Path, **kwargs: Any) -> None:
    """Subtract the paired sky frame from each object frame.

    For each ``(object_name, sky_name)`` pair in the listfile, reads
    ``workdir / "{object_name}.fits"`` and ``workdir / "{sky_name}.fits"``,
    writes ``workdir / "{OBJNAME}_NO{i}_s.fits"`` containing
    ``object_data - sky_data``. ``OBJNAME`` comes from the object frame's
    ``OBJECT`` header, sanitized; ``i`` is the 1-based pair index.
    """
    del config  # not consumed by s01
    del kwargs  # forward-compatibility hook only
    pairs = parse_listfile(listfile)
    for i, pair in enumerate(pairs, start=1):
        obj_data, obj_header = _fits.read_image(workdir / pair.object_name)
        sky_data, _ = _fits.read_image(workdir / pair.sky_name)
        diff = sky_subtract(obj_data, sky_data)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))
        out_path = workdir / f"{objname}_NO{i}_s.fits"
        _fits.write_image(out_path, diff, obj_header, overwrite=True)
