"""Configuration for the wavelength-calibration pass.

Every width here is a velocity, not a pixel count: WINERED samples at about
0.96 km/s per pixel in HIRES-Y and HIRES-J and 5.1 km/s per pixel in WIDE, so
a bound in pixels does not carry between modes. Pixel-space quantities are
derived per order from the measured dispersion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Nominal resolving power by WINERED instrument mode with the 100 um slit.
#: Seeds the LSF bounds and sizes the masks; the effective line width is
#: fitted per order and reported alongside.
NOMINAL_RESOLVING_POWER: dict[str, float] = {
    "WIDE": 28_000.0,
    "HIRES-Y": 68_000.0,
    "HIRES-J": 68_000.0,
}

MODES: tuple[str, ...] = ("oh_static", "oh_refit", "hybrid_static", "hybrid_refit")
AUTO_MODE = "auto"
ZERO_POINTS: tuple[str, ...] = ("absolute", "relative")
ASSEMBLIES: tuple[str, ...] = ("ladder", "decomposition")


def default_cache_root() -> Path:
    """Per-user cache used when the caller supplies no wavecal data paths."""
    override = os.environ.get("DECANTER_WAVECAL_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "decanter" / "wavecal"


@dataclass(frozen=True, slots=True)
class WavecalConfig:
    """Settings for one wavelength-calibration run.

    Attributes:
        mode: which reference and which per-exposure treatment. ``"auto"``
            selects ``hybrid_refit`` for HIRES-Y/HIRES-J and
            ``hybrid_static`` for WIDE. ``hybrid_*``
            uses telluric absorption where the order carries it and OH
            emission elsewhere; ``oh_*`` uses OH alone. ``*_refit`` refits the
            template amplitudes for every exposure with the line positions and
            the LSF held fixed, which is what keeps a saturated telluric
            line's centroid from moving with airmass.
        zero_point: ``"absolute"`` anchors the solution to the telluric rest
            frame, so a static per-order error in the dispersion solution is
            corrected too. ``"relative"`` centres each order on its own median
            over exposures, which measures only the time-variable part.
        assembly: ``"ladder"`` gives a direct measurement priority wherever one
            is trusted and interpolates the rest per exposure. ``"decomposition"``
            fits a static per-order offset plus a per-exposure common mode over
            the whole series.
        species: telluric molecules to include, in HITRAN naming.
        fsr_cut: which FSR cut to read; None takes the widest present.
        resolving_power: nominal R. 0 means look it up from the frame's
            ``INSTMODE``.
        shift_search_kms: half-width of the single-pass telluric shift search.
            The default is +/-25 km/s. Atmospheric prealignment remains an
            explicitly enabled two-pass procedure.
        lsf_sigma_bounds: multiples of the nominal LSF sigma that bound the
            fit. The lower bound matters: without it the optimiser widens the
            template until it can absorb stellar features.
        edge_trim_resolution_elements: how much of each order end to ignore,
            in resolution elements.
        telluric_rms_threshold: an order counts as telluric-rich when the RMS
            of its fitted transmission template exceeds this.
        telluric_peak_threshold / oh_peak_threshold: minimum CCF peak for a
            direct measurement to be accepted.
        telluric_refit_closure_kms: maximum allowed displacement of the
            nonlinear per-exposure refit from its accepted fixed-support CCF
            seed. A larger move is not self-consistent under the independent
            post-correction CCF and the direct seed is retained instead.
        oh_shift_search_kms: OH search half-width around the order's fitted
            static offset. Centering the search matters when the laboratory
            zero point and the whole-night drift do not both fit inside the
            telluric search window.
        oh_refit_window_kms: local half-width for the per-exposure OH
            band-amplitude and shift refit around the CCF estimate.
        oh_rich_min_lines: minimum number of individually detected OH lines
            for an order to anchor the solution.
        star_clean: divide out an empirical stellar template before the
            telluric fit. A no-op on featureless stars; important on cool ones.
        oh_tie: how to bring OH onto the telluric scale in absolute mode.
        atmospheric_prealign: run a first physical solve, robustly pool its
            broad CCF evidence from telluric-rich object orders and OH-rich
            paired-sky orders into one time-variable shift per exposure,
            register every order by that common mode, rebuild the atmospheric
            templates, and run the requested fine solve again. Tellurics take
            priority in orders rich in both references. This is the
            physical-reference alternative to WARP WAVSHIFT.
        atmospheric_search_kms / atmospheric_step_kms: half-width and spacing
            of the broad common-mode CCF grid used by atmospheric prealignment.
    """

    mode: str = AUTO_MODE
    zero_point: str = "absolute"
    assembly: str = "ladder"
    species: tuple[str, ...] = ("H2O", "CH4", "O2")
    fsr_cut: float | None = None
    resolving_power: float = 0.0
    shift_search_kms: float = 25.0
    lsf_sigma_bounds: tuple[float, float] = (0.5, 2.0)
    edge_trim_resolution_elements: float = 12.0
    telluric_rms_threshold: float = 0.05
    telluric_peak_threshold: float = 0.30
    telluric_refit_closure_kms: float = 0.50
    oh_peak_threshold: float = 0.35
    oh_shift_search_kms: float = 18.0
    oh_refit_window_kms: float = 6.0
    oh_rich_min_lines: int = 4
    star_clean: bool = True
    oh_tie: str = "global_constant"
    atmospheric_prealign: bool = False
    atmospheric_search_kms: float = 50.0
    atmospheric_step_kms: float = 0.25
    linelist_dir: str = ""
    cache_dir: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in (AUTO_MODE, *MODES):
            raise ValueError(
                f"mode must be {AUTO_MODE!r} or one of {MODES}, got {self.mode!r}"
            )
        if self.zero_point not in ZERO_POINTS:
            raise ValueError(f"zero_point must be one of {ZERO_POINTS}")
        if self.assembly not in ASSEMBLIES:
            raise ValueError(f"assembly must be one of {ASSEMBLIES}")
        if self.oh_tie not in ("none", "global_constant"):
            raise ValueError("oh_tie must be 'none' or 'global_constant'")
        low, high = self.lsf_sigma_bounds
        if not 0 < low < high:
            raise ValueError("lsf_sigma_bounds must satisfy 0 < low < high")
        if self.shift_search_kms <= 0:
            raise ValueError("shift_search_kms must be positive")
        if self.atmospheric_search_kms <= 0:
            raise ValueError("atmospheric_search_kms must be positive")
        if self.atmospheric_step_kms <= 0:
            raise ValueError("atmospheric_step_kms must be positive")
        if self.telluric_refit_closure_kms <= 0:
            raise ValueError("telluric_refit_closure_kms must be positive")
        if self.oh_shift_search_kms <= 0:
            raise ValueError("oh_shift_search_kms must be positive")
        if self.oh_refit_window_kms <= 0:
            raise ValueError("oh_refit_window_kms must be positive")

    # ------------------------------------------------------------------

    @staticmethod
    def default_mode_for(instmode: str | None) -> str:
        """Return the physical wavecal default for a WINERED band/mode."""
        normalized = str(instmode or "").strip().upper().replace("_", "-")
        if normalized == "WIDE" or "WIDE" in normalized:
            return "hybrid_static"
        if normalized in {"Y", "J", "HIRES-Y", "HIRES-J"}:
            return "hybrid_refit"
        raise ValueError(
            f"cannot choose an automatic wavecal mode for INSTMODE={instmode!r}; "
            f"set mode explicitly to one of {MODES}"
        )

    def resolved_for(self, instmode: str | None) -> WavecalConfig:
        """Resolve automatic mode and cache paths without changing this config."""
        root = default_cache_root()
        return replace(
            self,
            mode=(self.default_mode_for(instmode) if self.mode == AUTO_MODE else self.mode),
            linelist_dir=(self.linelist_dir or str(root / "hitran")),
            cache_dir=(self.cache_dir or str(root / "opacity")),
        )

    @property
    def uses_telluric(self) -> bool:
        return self.mode == AUTO_MODE or self.mode.startswith("hybrid")

    @property
    def uses_oh(self) -> bool:
        return True

    @property
    def per_exposure_refit(self) -> bool:
        if self.mode == AUTO_MODE:
            raise ValueError("resolve automatic wavecal mode against INSTMODE first")
        return self.mode.endswith("_refit")

    def resolving_power_for(self, instmode: str | None) -> float:
        """Nominal R for a frame, from the config or the instrument mode."""
        if self.resolving_power > 0:
            return float(self.resolving_power)
        if instmode and instmode in NOMINAL_RESOLVING_POWER:
            return NOMINAL_RESOLVING_POWER[instmode]
        raise ValueError(
            f"no nominal resolving power for INSTMODE={instmode!r}; "
            f"set WavecalConfig.resolving_power explicitly"
        )

    def resolution_element_kms(self, instmode: str | None) -> float:
        """The FWHM of one resolution element, in km/s."""
        from decanter.wavecal.solution import C_KMS

        return C_KMS / self.resolving_power_for(instmode)

    def lsf_sigma_kms(self, instmode: str | None) -> float:
        """Nominal Gaussian sigma of the instrument profile, in km/s."""
        return self.resolution_element_kms(instmode) / 2.354_820_045

    def lsf_sigma_bounds_kms(self, instmode: str | None) -> tuple[float, float]:
        nominal = self.lsf_sigma_kms(instmode)
        low, high = self.lsf_sigma_bounds
        return nominal * low, nominal * high
