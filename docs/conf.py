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
sys.path.insert(0, os.path.abspath('../ringdown/'))
sys.path.insert(0, os.path.abspath('../ringdown/waveforms/'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'ringdown'
copyright = '2021, Maximiliano Isi, Will M. Farr'
author = 'Maximiliano Isi, Will M. Farr'

# The full version, including alpha/beta/rc tags
release = 'July 12, 2021'

# -- Mock imports ---------------------------------------------------

# This helps ReadTheDocs with imports it can't handle, see
# https://read-the-docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['pymc']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinxemoji.sphinxemoji',
    'sphinxarg.ext',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_epilog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note:: This page was generated from a Jupyter notebook that can be
          `downloaded here <https://github.com/maxisi/ringdown/tree/main/docs/{{ docname }}>`_.
"""

nbsphinx_timeout = 30
