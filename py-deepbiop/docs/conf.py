"""Sphinx configuration."""

import sys
from datetime import datetime
from pathlib import Path

# sys.path.insert(0, (Path().resolve() / "../deepbiop").as_posix())

project = "deepbiop"
author = "Yangyang Li"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "autoapi.extension",
]
source_suffix = [".rst", ".md"]
autodoc_typehints = "description"
html_theme = "furo"
autoapi_dirs = ["../deepbiop"]

myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
]
