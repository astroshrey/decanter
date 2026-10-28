"""Auto-discover a calibration set from a directory on disk.

A typical decanter/WARP reduction directory contains a
``calibration_data/input_files.txt`` that lists the calibration FITS
files (flat, mask, comp, aperture-database name, aptrans name) used.
:class:`decanter.Calibration` is the in-memory dataclass that bundles
those paths together so callers don't thread 7 kwargs through every
task function.

Eventually decanter will compute these calibrations itself; for Phase 1
we read the WARP-supplied set.
"""
from __future__ import annotations

import re as _re
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits as _afits

from .._localpaths import CAL_DATA_ROOTS as _CAL_DATA_ROOTS
from ..io import headers as _headers

# The FSR wavelength-bound table (stage 13) is shipped with the package so a
# reduction works out of the box; overridable per-call via ``fsr_table``.
_BUNDLED_FSR_TABLE = Path(__file__).resolve().parent.parent / "data" \
    / "FSR_winered_20190610.txt"


class LockedAperturesWarning(UserWarning):
    """``Calibration.from_dir`` was given a WARP reduction root and loaded its
    frame-NO1 aperture databases, which lock every frame's extraction to NO1's
    WARP trace."""


class CalibrationFallbackWarning(UserWarning):
    """A calibration file was not next to ``input_files.txt`` (or in
    ``extra_cal_roots``) and was taken, by name, from another calibration
    set under the per-machine data roots."""


class CalibrationMismatch(ValueError):
    """Raised when a calibration set does not match the science frame it is
    being applied to (different instrument mode, slit, or grating setting)."""


@dataclass(frozen=True)
class InstrumentConfig:
    """The WINERED configuration a frame or calibration set was taken in.

    Determined by the ``INSTMODE`` / ``SLIT`` / ``SETTING`` FITS keywords,
    which together fix the echelle geometry and wavelength solution. A
    calibration set is only valid for science frames sharing these three.
    ``period`` (the LCO run, e.g. ``LCO24b``) is provenance only — a set from
    an adjacent run with the same mode/slit/setting is still usable.
    """

    mode: str      # INSTMODE, e.g. "HIRES-Y", "HIRES-J", "WIDE"
    slit: str      # SLIT, e.g. "100", "140", "200"
    setting: str   # SETTING, e.g. "2", "4"
    period: str    # PERIOD, e.g. "LCO24b" (provenance, not part of identity)

    @classmethod
    def from_header(cls, header) -> InstrumentConfig:
        """Read the config keywords from a FITS header (missing -> '?')."""
        def field(key: str) -> str:
            return str(_headers.get(header, key, default="?")).strip()
        return cls(mode=field("INSTMODE"), slit=field("SLIT"),
                   setting=field("SETTING"), period=field("PERIOD"))

    @property
    def tag(self) -> str:
        """Compact identity string, e.g. ``WIDE100_setting4``."""
        return f"{self.mode}{self.slit}_setting{self.setting}"

    def matches(self, other: InstrumentConfig) -> bool:
        """True if mode, slit and setting agree (period is ignored)."""
        return (self.mode, self.slit, self.setting) == \
               (other.mode, other.slit, other.setting)


@dataclass(frozen=True)
class Calibration:
    """Bundle of WARP calibration product paths used by every reduction.

    Attributes:
        flat: master flat-field FITS (e.g. ``flat_HIRES-Y100_*_mscmn.fits``).
        static_bp_mask: pre-built static bad-pixel mask FITS.
        apdb_multihole: aperture database for the multihole reference
            (multi-order trace).
        apdb_apsc: aperture database used by s03 (apscatter) — typically
            the flat-field aperture set.
        comp: comp-lamp FITS used to derive the dispersion solution.
        fc_dir: directory holding the ``fc<aptransname>`` files for s06.
        fc_refname: basename for the ``fc<refname>`` family (no ``fc`` prefix).
        id_dir: directory holding the ``id<comp>`` dispersion files for s12
            (commonly the same as ``fc_dir``).
        id_refname: basename for the ``id<refname>`` family (no ``id`` prefix).
        fsr_table: path to the FSR wavelength-bound table (s13).
        ref_trans_apdbs: per-order multihole *trans* reference aperture DBs
            (``ap{aptrans}_{m}trans``), a standard calibration product. These
            give the rectified-strip reference trace and the wide multihole
            slit window that WARP's ``centersearch_fortrans`` measures the
            per-frame stellar shift against — the WARP-independent extraction
            path uses them.
        trans_apdbs: optional per-order WARP-supplied *per-frame* trans
            aperture DBs (``ap*_NO{i}_sscfm_m{m}trans``, a reduction output,
            not a calibration product). When present, 1D extraction is locked
            to WARP's exact per-frame trace for bit-parity benches.
            :meth:`from_dir` on a reduction root fills it with frame NO1's
            DBs (and warns, :class:`LockedAperturesWarning`): that locks
            every frame to NO1's trace.
        instrument: the mode/slit/setting this set was taken in (read from
            the comp FITS header), or None if it could not be determined.
            Used by :meth:`assert_matches` to reject mismatched science data.
    """

    flat: Path
    static_bp_mask: Path
    apdb_multihole: Path
    apdb_apsc: Path
    comp: Path
    fc_dir: Path
    fc_refname: str
    id_dir: Path
    id_refname: str
    fsr_table: Path
    ref_trans_apdbs: dict[int, Path] | None = None
    trans_apdbs: dict[int, Path] | None = None
    instrument: InstrumentConfig | None = None

    def assert_matches(self, science_header) -> None:
        """Raise :class:`CalibrationMismatch` if this set does not match a frame.

        Compares the calibration's own instrument config against the science
        frame's ``INSTMODE`` / ``SLIT`` / ``SETTING``. No-op if either config
        is unavailable (e.g. an in-memory frame with no header, or a
        hand-built Calibration with ``instrument=None``).
        """
        if self.instrument is None:
            return
        frame = InstrumentConfig.from_header(science_header)
        if frame.mode == "?":  # no header keywords -> nothing to check against
            return
        if not self.instrument.matches(frame):
            raise CalibrationMismatch(
                f"calibration set is {self.instrument.tag} but the science "
                f"frame is {frame.tag} — wrong mode, slit, or grating setting. "
                f"(calib comp: {self.comp.name})")

    @classmethod
    def from_dir(
        cls,
        path: Path,
        *,
        fsr_table: Path | None = None,
        extra_cal_roots: tuple[Path, ...] = (),
    ) -> Calibration:
        """Auto-discover a Calibration from a calibration directory.

        ``path`` may be either (auto-detected):

          * a **calibration-set directory** that directly contains
            ``input_files.txt`` (point it anywhere — an absolute path to your
            own downloaded set works), or
          * a **reduction root** containing ``calibration_data/input_files.txt``
            (the layout WARP leaves behind).

        Referenced files are resolved relative to that directory first, then
        ``extra_cal_roots``, so a self-contained set needs no extra
        configuration. Failing both, a file is taken by name from another
        ``calibration_*`` directory under the per-machine data roots
        (:data:`decanter._localpaths.CAL_DATA_ROOTS`; sets whose
        ``input_files.txt`` names the same flat/mask/comp first) and a
        :class:`CalibrationFallbackWarning` says which. A WARP reduction root
        usually lacks the flat and mask, so pass the set it was reduced with
        in ``extra_cal_roots`` to make that choice explicit.

        Args:
            path: a calibration-set directory or a reduction root (see above).
            fsr_table: optional override for the FSR wavelength-bound table.
                Defaults to the copy bundled with the package.
            extra_cal_roots: additional directories to search for files not
                present alongside ``input_files.txt``.

        Raises:
            FileNotFoundError: if no ``input_files.txt`` is found at ``path``
                or ``path/calibration_data/``, or a referenced file is missing
                from every search root.
        """
        path = Path(path)
        if (path / "input_files.txt").is_file():
            calib_dir = path                       # a calibration-set directory
        elif (path / "calibration_data" / "input_files.txt").is_file():
            calib_dir = path / "calibration_data"  # a WARP reduction root
        else:
            raise FileNotFoundError(
                f"no input_files.txt at {path} or {path / 'calibration_data'}; "
                f"point from_dir() at a calibration-set directory (one holding "
                f"input_files.txt) or a reduction root containing "
                f"calibration_data/")

        fields = _parse_input_files(calib_dir / "input_files.txt")
        own_roots = (calib_dir,) + tuple(Path(r) for r in extra_cal_roots)
        # Fallback: published sets under the data roots, those whose own
        # input_files.txt names the same flat/mask/comp first (the set this
        # one was made from), then the rest.
        def _same_set(d: Path) -> bool:
            try:
                f = _parse_input_files(d / "input_files.txt")
            except OSError:
                return False
            return all(f.get(k) == fields.get(k) for k in ("flat file", "mask file", "comp file"))
        others = [p for root in _CAL_DATA_ROOTS for p in sorted(root.glob("calibration_*"))
                  if p.resolve() != calib_dir.resolve()]
        fallback_roots = tuple([p for p in others if _same_set(p)]
                               + [p for p in others if not _same_set(p)])
        search_roots = own_roots + fallback_roots

        flat = _resolve_warn(fields["flat file"], own_roots, fallback_roots, calib_dir)
        static_bp = _resolve_warn(fields["mask file"], own_roots, fallback_roots, calib_dir)
        comp = _resolve_warn(fields["comp file"], own_roots, fallback_roots, calib_dir)
        ap_name = fields["ap file"]
        apsc_name = fields["ap file for apscatter"]
        # The shared `database/` that hosts ap* + fc* + id* files lives next
        # to the calibration FITS files. Find it by looking for ``apN`` where
        # N = ap_name; that has to exist in the right database/ dir.
        shared_db = _find_db_with(f"ap{ap_name}", search_roots)
        if not any(shared_db in (r / "database", r) for r in own_roots):
            _warnings.warn(
                f"Calibration.from_dir({calib_dir}): no database/ with ap{ap_name} "
                f"there; using {shared_db} from another calibration set (matched by "
                f"file name only). Check it is the set this reduction used.",
                CalibrationFallbackWarning, stacklevel=2)
        apdb_multihole = shared_db / f"ap{ap_name}"
        apdb_apsc = shared_db / f"ap{apsc_name}"
        fc_refname = fields["aptrans file"]
        # id_refname is the comp filename stem (e.g. comp_HIRES-Y100_..._fm_ecall).
        id_refname = comp.stem

        # Discover the multihole trans reference apertures (a calibration
        # product): ap{aptrans}_{m}trans, one per echelle order.
        ref_trans_apdbs: dict[int, Path] = {}
        for cand in shared_db.glob(f"ap{fc_refname}_*trans"):
            try:
                mstr = cand.name.rsplit("_", 1)[1][: -len("trans")]
                ref_trans_apdbs[int(mstr)] = cand
            except (IndexError, ValueError):
                continue

        # Discover per-frame trans aperture DBs (a reduction output, present
        # only when pointed at a WARP reduction root) — best-effort. WARP
        # names them ``_sscfm_`` with Apscatter, ``_sfm_`` without
        # (Warp_sci.py:284-297).
        trans_apdbs: dict[int, Path] = {}
        for fm in ("sscfm", "sfm"):
            for cand in shared_db.glob(f"ap*_NO1_{fm}_m*trans"):
                mm = _re.search(rf"_NO1_{fm}_m(\d+)trans$", cand.name)
                if mm:
                    trans_apdbs[int(mm.group(1))] = cand
            if trans_apdbs:
                break

        if trans_apdbs:
            _warnings.warn(
                f"Calibration.from_dir({path}): a WARP reduction root, so its "
                f"frame-NO1 aperture databases ({len(trans_apdbs)} orders) are "
                f"loaded as trans_apdbs and reduce() will extract EVERY frame on "
                f"NO1's WARP trace (locked path). For decanter's own per-frame "
                f"apertures use dataclasses.replace(calib, trans_apdbs=None) or "
                f"point from_dir at the calibration set; reduce_to_warp_tree "
                f"locks each frame to its own databases via apertures=<root>.",
                LockedAperturesWarning, stacklevel=2)

        if fsr_table is None:
            fsr_table = _BUNDLED_FSR_TABLE

        # Provenance for the mismatch guard: the comp FITS carries the same
        # INSTMODE/SLIT/SETTING keywords as the science frames it calibrates.
        try:
            instrument = InstrumentConfig.from_header(_afits.getheader(comp))
        except (OSError, KeyError):
            instrument = None

        return cls(
            flat=flat,
            static_bp_mask=static_bp,
            apdb_multihole=apdb_multihole,
            apdb_apsc=apdb_apsc,
            comp=comp,
            fc_dir=shared_db,
            fc_refname=fc_refname,
            id_dir=shared_db,
            id_refname=id_refname,
            fsr_table=fsr_table,
            ref_trans_apdbs=ref_trans_apdbs or None,
            trans_apdbs=trans_apdbs or None,
            instrument=instrument,
        )


def _parse_input_files(path: Path) -> dict[str, str]:
    """Parse WARP's ``input_files.txt`` into a ``{lowercase_comment: value}`` dict.

    The file format is one ``<value>  # <Comment>`` per line, with a
    title header that we skip. Comments are normalized to lowercase so
    callers can use simple key lookups like ``fields["flat file"]``.
    """
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    fields: dict[str, str] = {}
    for ln in lines[1:]:  # skip header
        if "#" not in ln:
            continue
        name, comment = ln.split("#", 1)
        fields[comment.strip().lower()] = name.strip()
    return fields


def _resolve_warn(name: str, own_roots: tuple[Path, ...],
                  fallback_roots: tuple[Path, ...], calib_dir: Path) -> Path:
    """Find ``name`` in the set's own directory (and ``extra_cal_roots``);
    failing that, in other published sets, with a :class:`CalibrationFallbackWarning`
    naming the set it came from and any other set holding a different file of
    the same name."""
    for root in own_roots:
        if (root / name).exists():
            return root / name
    hits = [r / name for r in fallback_roots if (r / name).exists()]
    if not hits:
        raise FileNotFoundError(
            f"could not resolve calibration file {name!r} in any of "
            f"{[str(r) for r in own_roots + fallback_roots]}")
    chosen = hits[0]
    differing = [h for h in hits[1:] if h.stat().st_size != chosen.stat().st_size
                 or h.read_bytes() != chosen.read_bytes()]
    _warnings.warn(
        f"Calibration.from_dir({calib_dir}): {name} is not in that directory; "
        f"using {chosen} (another calibration set, matched by file name only)."
        + (f" DIFFERENT files of the same name exist in: "
           f"{', '.join(str(h.parent) for h in differing)}." if differing else "")
        + " Pass extra_cal_roots=[<the set this reduction used>] to choose explicitly.",
        CalibrationFallbackWarning, stacklevel=3)
    return chosen


def _find_db_with(filename: str, search_roots: tuple[Path, ...]) -> Path:
    """Return the first ``<root>/database/`` directory that contains ``filename``."""
    candidates = (
        [root / "database" for root in search_roots]
        + [root for root in search_roots if root.name == "database"]
    )
    for db in candidates:
        if (db / filename).exists():
            return db
    raise FileNotFoundError(
        f"could not find a database/ holding {filename!r} in any of "
        f"{[str(r / 'database') for r in search_roots]}"
    )
