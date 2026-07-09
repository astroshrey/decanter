"""Command-line interface — driver for :func:`decanter.reduce`.

Flag names match WARP's so a user can swap pipelines without retraining
muscle memory.

Behaviour: for each (obj, sky) pair in the listfile, call
:func:`decanter.reduce` with the loaded :class:`Calibration` and write
the per-frame outputs into ``destpath``. The legacy multi-frame
combine path (cross-frame waveshift + SNR stack) is reserved for the
future :func:`decanter.combine`; today the CLI keeps every frame's output
separate.

WARP equivalent: ``Warp_sci.py`` argparse + the multi-pair loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decanter import __version__, reduce
from decanter.calib import Calibration
from decanter.config import Config
from decanter.io.listfile import parse as parse_listfile


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Flags match ``Warp_sci.py``."""
    parser = argparse.ArgumentParser(
        prog="decanter",
        description=(
            "Pure-Python reduction pipeline for WINERED echelle spectra. "
            "Reduces each (obj, sky) frame pair in the listfile via "
            "decanter.reduce() and writes per-frame outputs to --destpath."
        ),
    )
    parser.add_argument("listfile", type=Path,
                        help="WARP-style input list (object + sky pairs).")
    parser.add_argument("-r", "--rawdatapath", type=Path, default=Path("../"),
                        help="Directory containing raw WINERED FITS frames.")
    parser.add_argument("-c", "--calibpath", type=Path, default=Path("./"),
                        help="Calibration directory: either a calibration-set "
                             "dir (holding input_files.txt) or a reduction root "
                             "containing calibration_data/. Any absolute path "
                             "works (parsed via Calibration.from_dir).")
    parser.add_argument("-d", "--destpath", type=Path, default=Path("./"),
                        help="Destination directory for reduced products.")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Preserve all intermediate FITS products "
                             "(passed as save_intermediates=True to reduce()).")
    parser.add_argument("-p", "--parameterfile", type=Path, default=None,
                        help="Path to a WARP-style 'Key: value' parameter file.")
    parser.add_argument("-f", "--fastmode", action="store_true",
                        help="Skip CR detection and wavelength shift "
                             "(Config.fast_mode).")
    parser.add_argument("--version", action="version", version=f"decanter {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    if args.parameterfile is not None:
        config = Config.from_parameter_file(args.parameterfile)
    elif args.fastmode:
        config = Config.fast_mode()
    else:
        config = Config()

    calib = Calibration.from_dir(args.calibpath)
    args.destpath.mkdir(parents=True, exist_ok=True)

    pairs = parse_listfile(args.listfile)
    for i, pair in enumerate(pairs, start=1):
        obj_path = args.rawdatapath / (
            pair.object_name + (".fits" if not pair.object_name.endswith(".fits") else "")
        )
        sky_path = args.rawdatapath / (
            pair.sky_name + (".fits" if not pair.sky_name.endswith(".fits") else "")
        )
        print(f"[{i}/{len(pairs)}] reducing {obj_path.name} - {sky_path.name}",
              flush=True)
        frame_workdir = args.destpath / f"NO{i}"
        reduce(
            obj_path, sky_path, calib,
            workdir=frame_workdir,
            save_intermediates=args.save,
            config=config,
        )
    print(f"done. {len(pairs)} frame(s) written under {args.destpath}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
