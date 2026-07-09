"""decanter pipeline configuration.

Mirrors WARP's ``warp/config.py:config`` class (lines 53-77) as an
immutable dataclass instead of a mutable object built up by side
effect. Defaults are kept identical to WARP so a fresh ``Config()``
reproduces ``Warp_sci.py`` behavior on the no-flag path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Static pipeline configuration.

    A ``Config`` instance is created once from the parameter file (or
    CLI query) and passed unchanged through every stage. Per-frame
    state (paths, headers, intermediate products) flows separately;
    this dataclass holds only run-wide settings.

    All defaults match ``warp/config.py:config.__init__`` so that an
    unmodified ``Config()`` reproduces WARP's no-flag behavior.
    """

    # --- Pipeline flags (mirror WARP's flag_* attributes) ---
    flag_apscatter: bool = True
    flag_manual_aperture: bool = False
    flag_skysub: bool = False
    flag_bpmask: bool = True
    # Default flipped 2026-05-13 to match WARP's new default: extract the
    # sky-frame spectrum alongside the object's. Doubles 2D/1D work but
    # produces sky_emission outputs the user actually wants. Fast mode
    # still leaves this False.
    flag_skyemission: bool = True
    flag_wsmeasure: bool = True
    flag_wscorrect: bool = True
    flag_wsmanual: bool = False
    flag_extract2d: bool = False

    # --- Reduction modes / ranges ---
    skysub_mode: str = "none"  # one of: none, average, median, minimum, fit
    cutrange_list: tuple[float, ...] = (1.05, 1.30)
    fluxinput: str = "no"  # cutransform flux-conservation flag

    # --- Cosmic-ray detection (warp/badpixmask.py:cosmicRayMask) ---
    CR_threshold: float = 10.0
    CR_var_ratio: float = 2.0
    CR_slitpos_ratio: float = 1.5
    CR_max_sigma: float = 20.0
    CR_fix_sigma: bool = False

    # --- Misc ---
    saturation_thres: float = 35000.0
    frame_number_limit: int = 28
    # Default flipped 2026-05-13 to match WARP's new default: reduce ALL
    # echelle orders, not just 163. The user's primary scientific
    # interest is the He I triplet (1083 nm, order 163), but the full
    # echelle range comes along for the ride. To restrict to a subset,
    # set ``reduce_full_data=False`` with a non-empty ``selected_orders``.
    reduce_full_data: bool = True
    selected_orders: tuple[int, ...] = (163,)

    @classmethod
    def fast_mode(cls) -> Config:
        """Return the parameter set used by ``Warp_sci.py --fastMode``.

        WARP equivalent: ``warp/config.py:config.setFastModeParam`` (line 385).
        """
        return cls(
            flag_apscatter=True,
            flag_manual_aperture=False,
            flag_skysub=False,
            flag_bpmask=False,
            flag_skyemission=False,
            flag_wsmeasure=False,
            flag_wscorrect=False,
            flag_wsmanual=False,
            flag_extract2d=False,
            skysub_mode="none",
            cutrange_list=(1.05,),
            fluxinput="no",
        )

    @classmethod
    def from_parameter_file(cls, path: Path) -> Config:
        """Read a WARP-style ``Key: value`` parameter file.

        WARP equivalent: ``warp/config.py:config.readParamFile`` (line 317).
        """
        raise NotImplementedError("parameter-file parser not yet ported")
