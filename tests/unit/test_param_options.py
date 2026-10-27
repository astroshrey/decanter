"""WARP parameter-file options: every option is accepted, a token is taken
the way pyraf takes it (apall's background minimum-matched, booleans in any
case), the tokens WARP stops on are refused, and the listfile's ap=/bg=
tokens are read the way warp/config.py:inputDataList reads them.

The accepted/refused tokens are those of a pyraf 2.2.4 probe in WARP's
environment (``echelle.apall`` / ``transform`` ``setParam``, and ``transform``
runs: ``INDEF`` passes pyraf and stops IRAF with error 603).
"""
import pytest

from decanter.api import background_mode, check_supported
from decanter.config import Config, apall_background, pyraf_bool
from decanter.warptree import read_warp_listfile


def test_default_and_fast_supported():
    check_supported(Config())
    check_supported(Config.fast_mode())


@pytest.mark.parametrize("kw", [
    dict(flag_skysub=True, skysub_mode="average"),
    dict(flag_skysub=True, skysub_mode="median"),
    dict(flag_skysub=True, skysub_mode="minimum"),
    dict(flag_skysub=True, skysub_mode="fit"),
    dict(flag_skysub=True, skysub_mode="med"),       # WARP run vfp_bgmedabbr_W
    dict(flag_skysub=True, skysub_mode="no"),        # WARP run vfpO0_fluxYes_bgno_Y
    dict(fluxinput="yes"),
    dict(fluxinput="Y"),
    dict(fluxinput="Yes"),                           # WARP run vfpO0_fluxYes_bgno_Y
    dict(flag_extract2d=True),
])
def test_options_supported(kw):
    check_supported(Config(**kw))


@pytest.mark.parametrize("kw", [
    dict(flag_skysub=True, skysub_mode="avg"),       # no choice starts with it
    dict(flag_skysub=True, skysub_mode="m"),         # median or minimum
    dict(flag_skysub=True, skysub_mode="Median"),    # case-sensitive
    dict(flag_skysub=True, skysub_mode="INDEF"),
    dict(fluxinput="true"),                          # not a pyraf boolean
    dict(fluxinput="ye"),
    dict(fluxinput="'yes'"),
    dict(fluxinput="INDEF"),                         # pyraf takes it, IRAF stops (603)
    dict(fluxinput="indef"),
])
def test_values_warp_cannot_run_refused(kw):
    with pytest.raises(ValueError):
        check_supported(Config(**kw))


def test_indirection_not_modelled():
    with pytest.raises(NotImplementedError):
        check_supported(Config(flag_skysub=True, skysub_mode=")apall.background"))
    with pytest.raises(NotImplementedError):
        pyraf_bool(")transform.flux")


def test_apall_background_minimum_match():
    want = {"none": "none", "no": "none", "n": "none", "non": "none",
            "a": "average", "av": "average", "averag": "average", "average": "average",
            "me": "median", "med": "median", "medi": "median", "median": "median",
            "mi": "minimum", "min": "minimum", "minimu": "minimum", "minimum": "minimum",
            "f": "fit", "fi": "fit", "fit": "fit", "'median'": "median", '"fit"': "fit"}
    for tok, mode in want.items():
        assert apall_background(tok) == mode, tok
    for tok in ("nonee", "averages", "avg", "m", "fitt", "Fit", "FIT", "Median", "None",
                "NONE", "INDEF", "indef", "yes", ""):
        with pytest.raises(ValueError):
            apall_background(tok)


def test_pyraf_bool():
    for v in ("yes", "Yes", "YES", "yEs", "y", "Y", True):
        assert pyraf_bool(v) is True
    for v in ("no", "No", "NO", "nO", "n", "N", False):
        assert pyraf_bool(v) is False
    for v in ("ye", "nope", "true", "True", "1", "0", "on", "off", "'yes'", '"no"',
              "INDEF", "indef", "Indef", "", " yes"):
        with pytest.raises(ValueError):
            pyraf_bool(v)
    assert Config(fluxinput="yes").transform_flux and not Config().transform_flux
    assert Config(fluxinput="Yes").transform_flux and not Config(fluxinput="nO").transform_flux


def test_background_no_keeps_the_products_flag():
    """``Background Subtraction: no``: flag_skysub on (warp/config.py:346-349),
    apall background none -- WARP still writes _none/_bg/_bgc/_bgcw."""
    cfg = Config(flag_skysub=True, skysub_mode="no")
    assert cfg.flag_skysub and background_mode(cfg) == "none"
    assert background_mode(Config(flag_skysub=True, skysub_mode="med")) == "median"
    # WARP sets skysub_mode = "none" without flag_skysub (Warp_sci.py:437-438)
    assert background_mode(Config(flag_skysub=False, skysub_mode="fit")) == "none"


def test_parameter_file_options(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("Manual Aperture: yes\nBackground Subtraction: average \n"
                 "CUTRANSFORM flux: yes\nExtract 2d spectrum: yes\n")
    cfg = Config.from_parameter_file(p)
    assert cfg.flag_manual_aperture and cfg.flag_skysub and cfg.flag_extract2d
    assert cfg.transform_flux and background_mode(cfg) == "average"
    check_supported(cfg)
    # "Background Subtraction: none" leaves the flag off (warp/config.py:346)
    p.write_text("Background Subtraction: none\n")
    assert background_mode(Config.from_parameter_file(p)) == "none"
    assert not Config.from_parameter_file(p).flag_skysub
    # any other token sets it, as typed; pyraf minimum-matches it
    p.write_text("Background Subtraction: no\nCUTRANSFORM flux: Yes\n")
    cfg = Config.from_parameter_file(p)
    assert cfg.flag_skysub and cfg.skysub_mode == "no" and background_mode(cfg) == "none"
    assert cfg.fluxinput == "Yes" and cfg.transform_flux
    check_supported(cfg)


def test_listfile_tokens(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("A B ap=-25.5:-12.25 bg=-40:-30 ws=0.5\nC D\n")
    a, b = read_warp_listfile(p)
    assert a.ap == (-25.5, -12.25) and a.bg == "-40:-30" and a.ws == 0.5
    # WARP's waveshift_man gets 0. for a line without ws= (warp/config.py:142-143)
    assert b.ap is None and b.bg is None and b.ws == 0.0
    p.write_text("A B\nC D\n")
    assert all(ln.ws is None for ln in read_warp_listfile(p))      # no ws= anywhere


def test_manual_flag_needs_window():
    import numpy as np
    import decanter
    with pytest.raises(ValueError, match="aperture"):
        decanter.reduce(np.zeros((4, 4), np.float32), None,
                        config=Config(flag_manual_aperture=True))
