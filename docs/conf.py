# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'KAHE-pipeline'
copyright = '2026, Chase Urasaki, Michael Zhang'
author = 'Chase Urasaki, Michael Zhang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Extensions for automatic documentation
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate from docstrings
    'sphinx.ext.napoleon',          # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Link to source code
    'sphinx.ext.intersphinx',       # Link to other project docs
    'sphinx_autodoc_typehints',     # Show type hints
]

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
}

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'