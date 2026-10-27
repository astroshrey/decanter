"""Parse WARP-style ``Key: value`` parameter files (``Warp_sci.py -p``).

Example: ``~/Disk/winered/parameter_files/nominal_params.txt``::

    Apscatter: yes
    Cosmic Ray Correction: yes
    Set cut range: 1.05, 1.3
    Sky Emission: yes
    Measure Shift: yes
    Correct Shift: yes

The port is literal, quirks included: WARP matches each key as a
case-sensitive substring of the line (so ``Selected Orders: 163`` with a
capital ``O`` is silently ignored, exactly as WARP ignores it), and the
value is the first whitespace token after the first ``:``.

WARP equivalent: ``warp/config.py:config.readParamFile`` (line 326).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_YN = {"yes": True, "no": False}


def parse(path: Path | str) -> dict[str, Any]:
    """Read a parameter file into :class:`decanter.config.Config` field values.

    Returns a dict of the fields the file sets (keys are ``Config`` field
    names), suitable for ``dataclasses.replace(Config(), **parsed)``.

    WARP equivalent: ``warp/config.py:config.readParamFile``.
    """
    out: dict[str, Any] = {}
    for line in Path(path).read_text().splitlines(keepends=True):
        def val() -> str:
            return line.split(":")[1].split()[0]

        def yn(field: str) -> None:
            out[field] = _YN[val()]

        if line.find("Apscatter") != -1:
            yn("flag_apscatter")
        if line.find("Manual Aperture") != -1:
            yn("flag_manual_aperture")
        if line.find("Cosmic Ray Correction") != -1:
            yn("flag_bpmask")
        if line.find("Extract all orders") != -1:
            yn("reduce_full_data")
        if line.find("Selected orders") != -1:
            if val() != "no":
                parts = line.split(":")[1].split(",")
                out["selected_orders"] = tuple(int(p.split()[0]) for p in parts)
        if line.find("Background Subtraction") != -1:
            if val() != "none":
                out["flag_skysub"] = True
                out["skysub_mode"] = val()
        if line.find("Set cut range") != -1:
            if val() != "no":
                parts = line.split(":")[1].split(",")
                out["cutrange_list"] = tuple(float(p.split()[0]) for p in parts)
        if line.find("CUTRANSFORM flux") != -1:
            out["fluxinput"] = val()
        if line.find("Sky Emission") != -1:
            yn("flag_skyemission")
        if line.find("Measure Shift") != -1:
            yn("flag_wsmeasure")
        if line.find("Correct Shift") != -1:
            yn("flag_wscorrect")
        if line.find("Manual Shift") != -1:
            yn("flag_wsmanual")
        for key, field in (("Cosmic ray threshold sigma", "CR_threshold"),
                           ("Cosmic ray maximum sigma", "CR_max_sigma"),
                           ("Cosmic ray Var/Ave ratio", "CR_var_ratio"),
                           ("Cosmic ray ratio between slit positions", "CR_slitpos_ratio")):
            if line.find(key) != -1:
                try:
                    out[field] = float(val())
                except ValueError:
                    pass   # WARP warns and keeps the default
        if line.find("Cosmic ray fix sigma") != -1:
            yn("CR_fix_sigma")
        if line.find("Extract 2d spectrum") != -1:
            yn("flag_extract2d")
    return out
