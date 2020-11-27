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
import os
import sys

if "READ_THE_DOCS" in os.environ:
    import subprocess

    # It is necessary for the readthedocs for finding cython docstrings
    subprocess.call("cd .. && python setup.py build_ext -i && cd ../doc", shell=True)
sys.path.insert(0, os.path.abspath(".."))
import iarray


# -- Project information -----------------------------------------------------

project = "ironArray for Python"
copyright = "2020-2021, ironArray Development Team"
author = "ironArray Development Team"

# The full version, including alpha/beta/rc tags
release = iarray.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

pygments_style = None
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/ironArray.png"
html_favicon = "_static/ironArray_logo.png"
html_show_sourcelink = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
# html_theme = "furo"

html_css_files = [
    "custom.css",
]

autodoc_member_order = "groupwise"
