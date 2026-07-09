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

from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits as _afits

from .._localpaths import CAL_DATA_ROOTS as _CAL_DATA_ROOTS
from ..io import headers as _headers

# The FSR wavelength-bound table (stage 13) is shipped with the package so a
# reduction works out of the box; overridable per-call via ``fsr_table``.
_BUNDLED_FSR_TABLE = Path(__file__).resolve().parent.parent / "data" \
    / "FSR_winered_20190610.txt"


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
        trans_apdbs: optional per-order WARP-supplied trans aperture DBs
            (``ap*_NO{i}_sscfm_m{m}trans``). Used to lock 1D extraction to
            WARP's exact trace for parity benches.
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
        ``extra_cal_roots``, then any ``calibration_*`` directory under the
        per-machine data roots (:data:`decanter._localpaths.CAL_DATA_ROOTS`) —
        so a self-contained set needs no extra configuration.

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
        search_roots = (calib_dir,) + tuple(extra_cal_roots) + tuple(
            p for root in _CAL_DATA_ROOTS for p in root.glob("calibration_*")
        )

        flat = _resolve(fields["flat file"], search_roots)
        static_bp = _resolve(fields["mask file"], search_roots)
        comp = _resolve(fields["comp file"], search_roots)
        ap_name = fields["ap file"]
        apsc_name = fields["ap file for apscatter"]
        # The shared `database/` that hosts ap* + fc* + id* files lives next
        # to the calibration FITS files. Find it by looking for ``apN`` where
        # N = ap_name; that has to exist in the right database/ dir.
        shared_db = _find_db_with(f"ap{ap_name}", search_roots)
        apdb_multihole = shared_db / f"ap{ap_name}"
        apdb_apsc = shared_db / f"ap{apsc_name}"
        fc_refname = fields["aptrans file"]
        # id_refname is the comp filename stem (e.g. comp_HIRES-Y100_..._fm_ecall).
        id_refname = comp.stem

        # Discover per-order trans aperture DBs (best-effort).
        trans_apdbs: dict[int, Path] = {}
        for cand in shared_db.glob("ap*_NO1_sscfm_m*trans"):
            try:
                mstr = cand.name.split("_sscfm_m")[1].split("trans")[0]
                trans_apdbs[int(mstr)] = cand
            except (IndexError, ValueError):
                continue

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


def _resolve(name: str, search_roots: tuple[Path, ...]) -> Path:
    """Find ``<root>/<name>`` for the first root in ``search_roots`` that has it."""
    for root in search_roots:
        cand = root / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"could not resolve calibration file {name!r} in any of "
        f"{[str(r) for r in search_roots]}"
    )


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
