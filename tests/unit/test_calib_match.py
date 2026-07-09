"""Instrument-config parsing, the calibration mismatch guard, and the
flexible from_dir() path resolution."""
from __future__ import annotations

import pytest
from astropy.io import fits

from decanter import Calibration, CalibrationMismatch
from decanter.calib import InstrumentConfig


def _hdr(mode="WIDE", slit="100", setting="4", period="LCO24b"):
    h = fits.Header()
    h["INSTMODE"], h["SLIT"], h["SETTING"], h["PERIOD"] = mode, slit, setting, period
    return h


def test_instrument_config_from_header_and_tag():
    cfg = InstrumentConfig.from_header(_hdr("HIRES-J", "100", "3", "LCO26a"))
    assert (cfg.mode, cfg.slit, cfg.setting, cfg.period) == \
           ("HIRES-J", "100", "3", "LCO26a")
    assert cfg.tag == "HIRES-J100_setting3"


def test_instrument_config_missing_keywords():
    assert InstrumentConfig.from_header(fits.Header()).mode == "?"


def test_matches_ignores_period():
    a = InstrumentConfig.from_header(_hdr(period="LCO24b"))
    b = InstrumentConfig.from_header(_hdr(period="LCO25a"))  # different run
    assert a.matches(b)


@pytest.mark.parametrize("mode,slit,setting", [
    ("HIRES-Y", "100", "4"),   # wrong mode
    ("WIDE", "200", "4"),      # wrong slit
    ("WIDE", "100", "2"),      # wrong setting
])
def test_matches_rejects_differences(mode, slit, setting):
    base = InstrumentConfig.from_header(_hdr("WIDE", "100", "4"))
    assert not base.matches(InstrumentConfig.from_header(_hdr(mode, slit, setting)))


def _calib(instrument):
    """Minimal Calibration carrying only an instrument config."""
    from pathlib import Path
    dummy = Path("comp_x.fits")
    return Calibration(
        flat=dummy, static_bp_mask=dummy, apdb_multihole=dummy, apdb_apsc=dummy,
        comp=dummy, fc_dir=dummy, fc_refname="x", id_dir=dummy, id_refname="x",
        fsr_table=dummy, instrument=instrument)


def test_assert_matches_passes_and_raises():
    cfg = InstrumentConfig.from_header(_hdr("WIDE", "100", "4"))
    _calib(cfg).assert_matches(_hdr("WIDE", "100", "4"))  # no raise
    with pytest.raises(CalibrationMismatch, match="wrong mode"):
        _calib(cfg).assert_matches(_hdr("HIRES-Y", "100", "2"))


def test_assert_matches_noop_without_provenance():
    # No instrument on the calib, or no keywords on the frame -> no check.
    _calib(None).assert_matches(_hdr())
    cfg = InstrumentConfig.from_header(_hdr("WIDE", "100", "4"))
    _calib(cfg).assert_matches(fits.Header())  # frame has no INSTMODE
