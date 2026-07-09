"""Parse WARP-style ``Key: value`` parameter files.

Example file: ``samplePapameter.txt`` in the WARP repo. Format is
forgiving: one ``key: value`` per line, ``#`` comments, whitespace
flexible. Recognized keys map to :class:`decanter.config.Config`
attributes.

WARP equivalent: ``warp/config.py:config.readParamFile`` (line 317).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse(path: Path) -> dict[str, Any]:
    """Read a parameter file into a dict of recognized settings.

    The dict can be passed to :meth:`decanter.config.Config` via
    ``replace(config, **parsed)`` (or its dataclass equivalent).

    Raises:
        NotImplementedError: not yet implemented.
    """
    raise NotImplementedError("io.parameter.parse: not yet implemented")
