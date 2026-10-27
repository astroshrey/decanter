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

    @property
    def transform_flux(self) -> bool:
        """``CUTRANSFORM flux`` as ``transform flux=`` receives it.

        WARP passes the parameter-file token to pyraf unchanged
        (``cutransform.py:137``); :func:`pyraf_bool` is what pyraf and IRAF
        make of it (``yes``/``no`` in any case, ``y``/``n``/``Y``/``N``), and it
        raises for a token on which WARP stops ("ERROR in 2d image
        reduction").
        """
        return pyraf_bool(self.fluxinput, "CUTRANSFORM flux")

    @property
    def apall_background(self) -> str:
        """apall's ``background`` for the object extraction.

        ``Background Subtraction`` other than ``none`` sets ``flag_skysub``
        and is passed to apall as typed (warp/config.py:346-349,
        Warp_sci.py:437-438, 561-562); pyraf minimum-matches it against
        apall's choices (:func:`apall_background`), so ``no`` is ``none`` (with
        ``flag_skysub`` still on) and ``med`` is ``median``.  ``none`` without
        ``flag_skysub``.
        """
        return apall_background(self.skysub_mode) if self.flag_skysub else "none"

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

        Fields the file does not mention keep their defaults (WARP starts
        from ``config.__init__`` too).

        WARP equivalent: ``warp/config.py:config.readParamFile`` (line 326).
        """
        from dataclasses import replace

        from decanter.io.parameter import parse
        return replace(cls(), **parse(path))


#: apall's ``background`` choices (``Background Subtraction``;
#: apextract/apall.par ``none|average|median|minimum|fit``).
BACKGROUND_MODES: tuple[str, ...] = ("none", "average", "median", "minimum", "fit")


def _pyraf_strip_quotes(value: str) -> str:
    """pyraf ``irafutils.stripQuotes``: one pair of enclosing quotes off,
    doubled quotes inside made single."""
    if value[:1] == '"':
        value = value[1:]
        if value[-1:] == '"':
            value = value[:-1]
        value = value.replace('""', '"')
    elif value[:1] == "'":
        value = value[1:]
        if value[-1:] == "'":
            value = value[:-1]
        value = value.replace("''", "'")
    return value


def _pyraf_remove_escapes(value: str) -> str:
    """pyraf ``irafutils.removeEscapes`` (``quoted=0``): the backslash before a
    quote, and backslash-newlines, removed."""
    for esc in ('\\"', "\\'"):
        i = value.find(esc)
        while i >= 0:
            value = value[:i] + value[i + 1:]
            i = value.find(esc, i + 1)
    i = value.find("\\\n")
    while i >= 0:
        value = value[:i] + value[i + 2:]
        i = value.find("\\\n", i + 1)
    return value


def pyraf_choice(value: str, choices: tuple[str, ...], what: str = "parameter") -> str:
    """The choice an enumerated IRAF string parameter takes from ``value``.

    pyraf strips quotes and escapes, then minimum-matches the value against
    the choice list, case-sensitively: an exact choice, else the one choice
    it is a prefix of (``med`` -> ``median``, ``no`` -> ``none``); a value
    that is no choice's prefix (``Median``, ``avg``) or several choices'
    (``m``: median, minimum) raises ``ValueError`` -- pyraf refuses it, the
    IRAF task never runs and WARP stops.  A ``)``-indirection (a reference
    to another parameter) is not modelled: ``NotImplementedError``.

    pyraf equivalent: ``tools/basicpar.py:_StringMixin`` (``_coerceOneValue``,
    ``checkOneValue``) with ``tools/minmatch.py:MinMatchDict`` (pyraf 2.2.4).
    """
    v = _pyraf_remove_escapes(_pyraf_strip_quotes(str(value)))
    if v[:1] == ")":
        raise NotImplementedError(f"'{what}: {value}' is a pyraf parameter indirection "
                                  f"(not modelled)")
    if v in choices:
        return v
    hits = [c for c in choices if v and c.startswith(v)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ValueError(f"'{what}: {value}' is ambiguous for pyraf ({'|'.join(hits)}); "
                         f"WARP stops on it")
    raise ValueError(f"'{what}: {value}' is not one of {'|'.join(choices)} or a prefix of "
                     f"one (pyraf refuses it); WARP stops on it")


def apall_background(value: str) -> str:
    """apall's ``background`` from WARP's ``Background Subtraction`` token
    (:func:`pyraf_choice` over :data:`BACKGROUND_MODES`): ``a``..``average``,
    ``me``..``median``, ``mi``..``minimum``, ``f``..``fit``, ``n``..``none``.

    Checked with pyraf 2.2.4 in WARP's environment (``echelle.apall``
    ``setParam``), and by WARP runs with ``med`` and ``no``
    (docs/BITEXACT.md).
    """
    return pyraf_choice(value, BACKGROUND_MODES, "Background Subtraction")


def pyraf_bool(value, what: str = "parameter") -> bool:
    """What an IRAF boolean task parameter is when WARP passes ``value``.

    pyraf's coercion (``tools/basicpar.py:_BooleanMixin._coerceOneValue``,
    ``tools/irafglobals.py:_Boolean._compare``) takes ``yes``/``no`` in any
    case and a one-letter ``y``/``n`` in either case.  It also accepts
    ``INDEF`` (any case), but the task then stops with IRAF error 603
    "Parameter not a legal boolean" (checked with ``transform``), so WARP
    stops: ``ValueError``, as for every other string pyraf refuses (``true``,
    ``1``, ``on``, ``ye``, quoted ``'yes'``) and for the empty string (an unset
    parameter to pyraf; a parameter-file token is never empty).  A
    ``)``-indirection is not modelled (``NotImplementedError``).
    """
    if value is True or value is False:
        return bool(value)
    if isinstance(value, str):
        low = value.lower()
        if low in ("yes", "y"):
            return True
        if low in ("no", "n"):
            return False
        v2 = _pyraf_strip_quotes(value.strip())
        if v2 == "" or v2.upper() == "INDEF":
            raise ValueError(f"'{what}: {value}' is INDEF (or unset) to pyraf, which IRAF "
                             f"refuses (error 603, not a legal boolean); WARP stops on it")
        if v2[:1] == ")":
            raise NotImplementedError(f"'{what}: {value}' is a pyraf parameter indirection "
                                      f"(not modelled)")
    raise ValueError(f"'{what}: {value}' is not a boolean pyraf accepts (yes/no in any "
                     f"case, y/n); WARP stops on it")
