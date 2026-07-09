"""Stage 4 — flat-fielding.

WARP equivalent: ``warp/Spec2Dtools.py:flatfielding`` (line 13) wrapping
``iraf.imarith(input, "/", flatfile, output)`` plus a ``FLAT`` header
keyword add.

OBJ path: reads ``_ssc.fits`` (s03 output), writes ``_sscf.fits``.

SKY path (when ``config.flag_skyemission`` is True): also flat-divides
the RAW sky frame (taken straight from the listfile, no apscatter or
sky-subtraction applied) and writes ``_skyNO{i}_f.fits``. Matches
``Warp_sci.py:366-368``::

    if conf.flag_skyemission:
        flatfielding(conf.skylist[i], sky_f_list[i], conf.flat_file)

The division is done at float32 precision to match IRAF ``imarith``
exactly bit-for-bit on matched-shape float32 inputs (the dtype used
for WINERED frames + master flat).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits as _astrofits

from decanter.config import Config
from decanter.io import fits as _fits
from decanter.io import headers
from decanter.io.listfile import parse as parse_listfile


def _sanitize_objname(raw: str) -> str:
    """Filename-sanitization rule shared across stages."""
    name = raw
    for ch in (" ", "'", "\"", "#", "/"):
        name = name.replace(ch, "_")
    return name


def flatfield_divide(data: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Pure task function: IRAF ``imarith``-faithful flat division.

    All arithmetic at float32 to match WARP byte-for-byte on the
    ``>f4`` WINERED inputs. Pixels where the flat is zero are set to
    zero in the output (matches IRAF imarith's div-by-zero → 0
    convention). Output dtype matches input.
    """
    if data.shape != flat.shape:
        raise ValueError(
            f"shape mismatch: data is {data.shape}, flat is {flat.shape}"
        )
    flat_f32 = flat.astype(np.float32, copy=False)
    data_f32 = data.astype(np.float32, copy=False)
    quot = np.zeros_like(data_f32)
    nonzero = flat_f32 != 0
    np.divide(data_f32, flat_f32, where=nonzero, out=quot)
    return quot.astype(data.dtype, copy=False)


def run(
    config: Config,
    workdir: Path,
    listfile: Path,
    *,
    flat_path: Path | str | None = None,
    **_unused: Any,
) -> None:
    """Divide each ``_ssc.fits`` frame by the master flat.

    Args:
        config: pipeline configuration (no flat-related flags — the
            stage always runs).
        workdir: reads ``{obj}_NO{i}_ssc.fits``;
            writes ``{obj}_NO{i}_sscf.fits``.
        listfile: WARP-style input list.
        flat_path: path to the master flat FITS. Required.

    Notes:
        Adds a ``FLAT`` header keyword to the output naming the flat
        used (mirrors ``warp/Spec2Dtools.py:16``). Output dtype matches
        input — preserves WINERED's ``>f4`` convention.

        Pixels where the flat is zero produce non-finite quotients; we
        leave them as-is (NaN/Inf). Downstream s05 ``pyfixpix`` is
        responsible for filling those via the bad-pixel mask, so we
        don't double-handle them here.
    """
    if flat_path is None:
        raise ValueError(
            "s04_flatfield requires flat_path; the calibration loader "
            "(s00) will provide this automatically once wired in."
        )

    flat_data, _flat_header = _fits.read_image(flat_path)
    pairs = parse_listfile(listfile)

    for i, pair in enumerate(pairs, start=1):
        _obj_raw, obj_header = _fits.read_image(workdir / pair.object_name)
        raw_objname = headers.get(obj_header, "OBJECT", default=pair.object_name)
        objname = _sanitize_objname(str(raw_objname))

        # --- OBJ path ----------------------------------------------------------
        data, header = _fits.read_image(workdir / f"{objname}_NO{i}_ssc.fits")
        out = flatfield_divide(data, flat_data)
        out_header = header.copy()
        out_header["FLAT"] = (str(Path(flat_path).name), "Flat-field file applied (s04)")
        out_path = workdir / f"{objname}_NO{i}_sscf.fits"
        _fits.write_image(out_path, out, out_header, overwrite=True)

        # --- SKY path (only when flag_skyemission is set) ----------------------
        if config.flag_skyemission:
            sky_data, sky_header = _fits.read_image(workdir / pair.sky_name)
            sky_out = flatfield_divide(sky_data, flat_data)
            sky_out_header = sky_header.copy()
            sky_out_header["FLAT"] = (str(Path(flat_path).name),
                                      "Flat-field file applied (s04, sky)")
            sky_out_path = workdir / f"{objname}_skyNO{i}_f.fits"
            _fits.write_image(sky_out_path, sky_out, sky_out_header, overwrite=True)
