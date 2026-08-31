"""Automatic wavecal data paths require no user-staged line lists."""

from __future__ import annotations

from decanter.wavecal.config import WavecalConfig
from decanter.wavecal.opacity import linelist_path


def test_default_paths_use_private_cache_and_allocate_exojax_download(monkeypatch, tmp_path) -> None:
    root = tmp_path / "wavecal-cache"
    monkeypatch.setenv("DECANTER_WAVECAL_CACHE", str(root))

    config = WavecalConfig().resolved_for("HIRES-Y")
    assert config.mode == "hybrid_refit"
    assert config.linelist_dir == str(root / "hitran")
    assert config.cache_dir == str(root / "opacity")

    path = linelist_path("H2O", config.linelist_dir)
    assert path == root / "hitran" / "H2O" / "H2O.hdf5"
    assert path.parent.is_dir()
    assert not path.exists()
