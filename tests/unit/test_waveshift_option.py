"""The ``waveshift=`` keyword of reduce_many / reduce_to_warp_tree (default off)."""
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
    assert _with_waveshift(on, None) is on                   # None: follow the config


def test_waveshift_off_by_default():
    for fn in (decanter.reduce_many, decanter.reduce_to_warp_tree):
        assert inspect.signature(fn).parameters["waveshift"].default is False
