"""Portable science products derived from a physical wavelength-calibration run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decanter.wavecal.solution import C_KMS
from decanter.wavecal.telluric import transmission_numpy


def telluric_product(run, path: str | Path) -> Path | None:
    """Save fitted telluric transmission on every calibrated exposure grid.

    The transmission is saved continuous rather than thresholded, so a
    downstream analysis can set its own ``mask = transmission < limit`` without
    refitting the atmosphere.
    """
    model = run.telluric_model
    if model is None:
        return None
    series = run.series
    solution = run.solution
    n_species = len(model.species)
    transmission = np.full(
        (series.n_frames, series.n_orders, series.n_pixels), np.nan,
        dtype=np.float32,
    )
    wavelength = np.full_like(transmission, np.nan, dtype=np.float64)
    refit = run.telluric_refit_parameters
    tau = run._telluric_tau
    if refit is not None and tau is None:
        raise RuntimeError("telluric refit parameters are present but opacity was not retained")

    for i in range(series.n_frames):
        for j in range(series.n_orders):
            velocity = float(solution.velocity[i, j])
            if not np.isfinite(velocity):
                continue
            calibrated_wave = series.wave[:, j] / (1.0 + velocity / C_KMS)
            wavelength[i, j] = calibrated_wave
            native = np.asarray(model.native_template[:, j], dtype=float)
            if refit is not None and np.all(np.isfinite(refit[i, j])):
                parameters = np.asarray(model.parameters[j], dtype=float).copy()
                fitted = np.asarray(refit[i, j], dtype=float)
                parameters[:n_species] = fitted[:n_species]
                parameters[n_species] = 0.0
                parameters[n_species + 4:] = fitted[n_species + 1:]
                native = transmission_numpy(
                    tau[:, :, j],
                    parameters,
                    str(model.family[j]),
                    n_species,
                    series.n_pixels,
                    shift=0.0,
                    with_continuum=False,
                )
            transmission[i, j] = np.interp(
                calibrated_wave,
                series.wave[:, j],
                native,
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "decanter.telluric-transmission.v1",
        "wavelength_unit": "angstrom_vacuum",
        "axes": "exposure,order,pixel",
        "threshold_semantics": "mask where transmission < user threshold",
        "species": list(model.species),
        "wavecal_mode": solution.mode,
    }
    np.savez_compressed(
        output,
        frame_ids=np.asarray(series.frame_ids, dtype="U96"),
        orders=np.asarray(series.orders, dtype=np.int32),
        wavelength_angstrom=wavelength,
        transmission=transmission,
        telluric_fit_accepted=np.asarray(run.telluric_accepted, dtype=bool),
        telluric_peak=np.asarray(run.telluric_peak, dtype=np.float32),
        template_rms=np.asarray(model.template_rms, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output


def airglow_product(run, path: str | Path) -> Path | None:
    """Save the fitted OH line support so a later run can rebuild the sky mask.

    The support is the set of pixels around detected OH lines, which SERVAL
    flags so airglow residuals do not enter the stellar RV. It lives on the
    wavecal reference grid rather than per exposure, because the fit is made
    against the time-median sky.
    """
    model = run.oh_model
    if model is None:
        return None
    series = run.series
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "decanter.oh-support.v1",
        "wavelength_unit": "angstrom_vacuum",
        "axes": "pixel,order",
        "support_semantics": "True where a detected OH line should be masked",
        "wavecal_mode": run.solution.mode,
    }
    np.savez_compressed(
        output,
        orders=np.asarray(series.orders, dtype=np.int32),
        wavelength_angstrom=np.asarray(series.wave, dtype=np.float64),
        support=np.asarray(model.support, dtype=bool),
        line_count=np.asarray(model.line_count, dtype=np.int32),
        rotational_temperature_k=np.asarray(model.rotational_temperature_k),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output
