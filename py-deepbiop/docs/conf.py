"""Sphinx configuration."""

from datetime import datetime

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
