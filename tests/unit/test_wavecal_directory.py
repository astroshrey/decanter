"""Writing native-grid products from a physical wavecal solution."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from decanter.wavecal.directory import apply_solution_to_directory
from decanter.wavecal.solution import C_KMS, WavecalSolution


def test_apply_solution_to_directory_rescales_wcs_without_resampling(tmp_path) -> None:
    source_root = tmp_path / "input"
    frame_root = source_root / "WINA00000001"
    frame_root.mkdir(parents=True)
    source = frame_root / "TARGET_NO1_sscfm_m160_fsr1.30_VAC.fits"
    header = fits.Header()
    header["OBJFRAME"] = "WINA00000001"
    header["ECHORDER"] = 160
    header["CRVAL1"] = 10000.0
    header["CDELT1"] = 0.03
    flux = np.arange(32, dtype=np.float32)
    fits.writeto(source, flux, header)

    solution = WavecalSolution(
        frame_ids=("WINA00000001",),
        orders=(160,),
        velocity=np.array([[1.5]]),
        source=np.array([["telluric"]]),
        bracketed=np.array([[True]]),
        mode="hybrid_refit",
    )
    output_root = tmp_path / "output"
    count = apply_solution_to_directory(solution, source_root, output_root)

    assert count == 1
    output = output_root / source.relative_to(source_root)
    np.testing.assert_array_equal(fits.getdata(output), flux)
    calibrated = fits.getheader(output)
    scale = 1.0 + 1.5 / C_KMS
    assert calibrated["CRVAL1"] == 10000.0 / scale
    assert calibrated["CDELT1"] == 0.03 / scale
    assert calibrated["WAVECAL"] == "hybrid_refit"
    assert calibrated["WAVECSRC"] == "telluric"
    assert (output_root / "wavecal_solution.npz").exists()
