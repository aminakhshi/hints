# Configuration file for the Sphinx documentation builder.
#

import os
import re
import sys

# Put the repository root on the path so that ``import hints`` resolves to the
# package rather than to hints/hints.py.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

def _get_version():
    """Reads the version from the package without importing its dependencies."""
    init_file = os.path.join(os.path.dirname(__file__), '..', 'hints', '__init__.py')
    with open(init_file, 'r') as handle:
        match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", handle.read(), re.M)
    return match.group(1) if match else '0.0.0'


release = _get_version()
version = '.'.join(release.split('.')[:2])
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



# 'build' is the local output directory of docs/Makefile. Without it here,
# Sphinx picks up its own previous output as source files.
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store', 'requirements.txt']

# The optional GPU backend must not be required to build the documentation.
autodoc_mock_imports = ['torch']



templates_path = ['_templates']
# -- Options for HTML output -------------------------------------------------

html_theme = 'groundwork'
html_static_path = ['_static']
