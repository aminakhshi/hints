# Configuration file for the Sphinx documentation builder.
#

import os
import sys

sys.path.insert(0, os.path.abspath('../../hints'))
sys.path.insert(0, os.path.abspath('../../examples'))
# -- Project information -----------------------------------------------------
# from hints import __version__

# Get the version and release
version = 0.1.1
release = 0.1
project = 'HiNTS'
copyright = '2024, Amin Akhshi'
author = 'Amin Akhshi'
# release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'nbsphinx_link'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'show-inheritance': True,
}

bibtex_bibfiles = ['refs.bib']
# bibtex_default_style = 'apa'



exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


templates_path = ['_templates']
# -- Options for HTML output -------------------------------------------------

html_theme = 'groundwork'
html_static_path = ['_static']
