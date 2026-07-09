"""Sphinx configuration for the decanter API documentation.

Builds on Read the Docs (see ``.readthedocs.yaml``) or locally with::

    pip install -e '.[docs]'
    sphinx-build -b html docs docs/_build/html
"""
from __future__ import annotations

import importlib.metadata

project = "decanter"
author = "Shreyas Vissapragada"
copyright = "2026, Shreyas Vissapragada"
try:
    release = importlib.metadata.version("decanter")
except importlib.metadata.PackageNotFoundError:  # not installed (e.g. bare checkout)
    release = "0.0.1"
version = release

extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from the source
    "sphinx.ext.napoleon",      # understand Google-style Args/Returns/Raises
    "sphinx.ext.autosummary",   # summary tables
    "sphinx.ext.viewcode",      # [source] links
    "sphinx.ext.intersphinx",   # cross-link numpy/astropy
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "decanter"
