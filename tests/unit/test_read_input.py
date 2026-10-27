"""decanter.reduce accepts str paths as well as Path objects (README usage)."""
import numpy as np
from astropy.io import fits

from decanter.api import _read_input


def test_str_and_path_inputs(tmp_path):
    p = tmp_path / "a.fits"
    fits.PrimaryHDU(np.arange(6, dtype=np.float32).reshape(2, 3),
                    header=fits.Header({"OBJECT": "x"})).writeto(p)
    for o in (str(p), p):
        data, hdr, sky, skyhdr, op, sp = _read_input(o, str(p))
        assert data.shape == (2, 3) and sky.shape == (2, 3)
        assert hdr["OBJECT"] == "x" and op == p and sp == p
