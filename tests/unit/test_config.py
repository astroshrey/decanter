"""Unit tests for :class:`decanter.config.Config`."""

from __future__ import annotations

import pytest

from decanter.config import Config


def test_default_config_matches_warp_defaults() -> None:
    """``Config()`` reproduces ``warp/config.py:config.__init__`` defaults."""
    cfg = Config()
    assert cfg.flag_apscatter is True
    assert cfg.flag_bpmask is True
    assert cfg.flag_wsmeasure is True
    assert cfg.flag_wscorrect is True
    assert cfg.flag_manual_aperture is False
    assert cfg.flag_skysub is False
    assert cfg.flag_extract2d is False
    assert cfg.skysub_mode == "none"
    assert cfg.cutrange_list == (1.05, 1.30)
    assert cfg.CR_threshold == 10.0
    assert cfg.CR_var_ratio == 2.0
    assert cfg.CR_max_sigma == 20.0
    assert cfg.saturation_thres == 35000.0
    assert cfg.frame_number_limit == 28


def test_fast_mode_disables_optional_stages() -> None:
    """``Config.fast_mode()`` mirrors ``Warp_sci.py --fastMode``."""
    cfg = Config.fast_mode()
    assert cfg.flag_bpmask is False
    assert cfg.flag_wsmeasure is False
    assert cfg.flag_wscorrect is False
    assert cfg.cutrange_list == (1.05,)


def test_config_is_frozen() -> None:
    """A frozen dataclass refuses attribute mutation."""
    cfg = Config()
    with pytest.raises((AttributeError, TypeError)):
        cfg.flag_bpmask = False  # type: ignore[misc]
