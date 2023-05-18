# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
from pathlib import Path

project = "trieste"
# fmt: off
copyright = (
    'Copyright 2020 The Trieste Contributors\n'
    '\n'
    'Licensed under the Apache License, Version 2.0 (the "License");\n'
    'you may not use this file except in compliance with the License.\n'
    'You may obtain a copy of the License at\n'
    '\n'
    '    http://www.apache.org/licenses/LICENSE-2.0\n'
    '\n'
    'Unless required by applicable law or agreed to in writing, software\n'
    'distributed under the License is distributed on an "AS IS" BASIS,\n'
    'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n'
    'See the License for the specific language governing permissions and\n'
    'limitations under the License.\n'
)
# fmt: on
author = "The Trieste Contributors"

# The full version, including alpha/beta/rc tags
release = Path("../trieste/VERSION").read_text().strip()

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
]
add_module_names = False
autosectionlabel_prefix_document = True

# sphinx-autoapi
extensions.append("autoapi.extension")
autoapi_dirs = ["../trieste"]
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]

# TODO: remove once https://github.com/sphinx-doc/sphinx/issues/4961 is fixed
suppress_warnings = ["ref.python"]

# intersphinx
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# nbsphinx
extensions.append("nbsphinx")
nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}

# sphinxcontrib-bibtex
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# library logo location
html_logo = "_static/logo.png"

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False

# theme-specific options. see theme docs for more info
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/secondmind-labs/trieste",
    "switcher": {
        "json_url": "https://secondmind-labs.github.io/trieste/versions.json",
        "version_match": release,
    },
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}
