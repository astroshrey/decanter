"""dispcor's output-edge wavelengths come from the output MULTISPEC WCS, whose
w1/dw are stored as "%.14g" text (smw_swattrs) -- a Newton result on the
LCO24b setting5 m178 inverse sits 8e-7 quanta from a rounding boundary and
only the text-rounded grid gives WARP's edge (verified on a fresh WARP -s run
of TOI544 NO1 and with IRAF dispcor on a ramp spectrum)."""
from pathlib import Path

import pytest

CAL = Path.home() / "Disk/winered/calibration_data/calibration_LCO24b_setting5_HIRES-Y100"


@pytest.mark.skipif(not CAL.is_dir(), reason="LCO24b setting5 calibration set not present")
def test_setting5_m178_edge_3052():
    import decanter
    from decanter.wavelength.iraf_msp import dispcor_edges
    c = decanter.Calibration.from_dir(CAL)
    ids = tuple(str(p) for p in sorted(Path(c.id_dir).glob(f"id{c.id_refname}.0*")))
    k = [i for i, p in enumerate(ids) if p.endswith(".0178")][0]
    _, _, _, edges = dispcor_edges(ids, 4095)
    assert round(edges[k, 3052], 4) == 1367.0082
