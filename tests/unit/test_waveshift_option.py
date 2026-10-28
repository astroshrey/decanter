"""The ``waveshift=`` keyword of reduce_many / reduce_to_warp_tree."""
from __future__ import annotations

import inspect

import decanter
from decanter.api import _with_waveshift
from decanter.config import Config


def test_with_waveshift_sets_both_flags():
    on = Config()
    assert on.flag_wsmeasure and on.flag_wscorrect          # WARP's own default
    off = _with_waveshift(on, False)
    assert not off.flag_wsmeasure and not off.flag_wscorrect
    back = _with_waveshift(off, True)
    assert back.flag_wsmeasure and back.flag_wscorrect


def test_waveshift_none_is_the_modes_choice():
    on = Config()
    assert _with_waveshift(on, None, "warp-clone") is on     # as configured (WARP: on)
    off = _with_waveshift(on, None, "default")               # the wavecal replaces it
    assert not off.flag_wsmeasure and not off.flag_wscorrect
    for fn in (decanter.reduce_many, decanter.reduce_to_warp_tree):
        assert inspect.signature(fn).parameters["waveshift"].default is None
        assert inspect.signature(fn).parameters["mode"].default == "default"
