"""Apply a physical wavecal solution to existing Decanter FITS products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from decanter.wavecal.solution import C_KMS, WavecalSolution


def apply_solution_to_directory(
    solution: WavecalSolution,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> int:
    """Write corrected native-grid FITS products without resampling flux.

    Only Decanter's final ``*_VAC.fits`` products are copied. Their directory
    and filenames are preserved, while ``CRVAL1`` and ``CDELT1`` are divided
    by ``1 + v/c`` for the matching frame and physical order.

    Returns the number of spectra written.
    """
    source_root = Path(input_dir).resolve()
    destination_root = Path(output_dir).resolve()
    if source_root == destination_root:
        raise ValueError("input_dir and output_dir must be different")

    paths = sorted(source_root.glob("*/*_VAC.fits"))
    if not paths:
        raise FileNotFoundError(f"no final Decanter FITS products under {source_root}")
    if destination_root.exists() and any(destination_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination_root}; pass overwrite=True"
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for source_path in paths:
        with fits.open(source_path, memmap=False) as hdul:
            header = hdul[0].header.copy()
            data = np.asarray(hdul[0].data).copy()

        frame_id = str(header.get(
            "SERIESID", header.get("OBJFRAME", source_path.parent.name)
        )).strip()
        order = int(header["ECHORDER"])
        row = solution.frame_index(frame_id)
        column = solution.order_index(order)
        velocity = float(solution.velocity[row, column])
        if not np.isfinite(velocity):
            raise ValueError(f"no finite wavecal for frame {frame_id}, order {order}")

        scale = 1.0 + velocity / C_KMS
        header["CRVAL1"] = (float(header["CRVAL1"]) / scale,
                            "Physical-wavecal wavelength at reference pixel")
        step_key = "CDELT1" if "CDELT1" in header else "CD1_1"
        header[step_key] = (float(header[step_key]) / scale,
                            "Physical-wavecal wavelength step")
        header["WAVECAL"] = (solution.mode, "Physical wavelength-calibration mode")
        header["WAVECZP"] = (solution.zero_point, "Wavecal zero-point convention")
        header["WAVECASM"] = (solution.assembly, "Wavecal assembly method")
        header["WAVECVEL"] = (velocity, "Applied physical wavecal shift [km/s]")
        header["WAVECSRC"] = (str(solution.source[row, column]), "Wavecal source")
        header["WAVECBRK"] = (bool(solution.bracketed[row, column]),
                              "Interpolation bracketed by anchors")

        destination = destination_root / source_path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(destination, data, header, overwrite=overwrite)
        written += 1

    solution.save_npz(destination_root / "wavecal_solution.npz")
    return written
