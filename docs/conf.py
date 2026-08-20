# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add path to module
#sys.path.insert(0, os.path.abspath('/home/mayra/source/Nichesphere/nichesphere/nichesphere'))
#sys.path.insert(0, os.path.abspath('../'))
#sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NicheSphere'
copyright = '2025, Mayra Ruiz, James Nagai'
author = 'Mayra Ruiz, James Nagai'
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon', 
    'sphinx_book_theme',
    'nbsphinx', 
    'sphinx.ext.autosummary', 
    'sphinx.ext.intersphinx',
    "sphinx_design"]

autosummary_generate = True # Auto-generates individual API doc pages from summary tables
templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- It hides the input/output prompt numbers (In [1]: / Out [1]:).
nbsphinx_prolog = """
.. raw:: html

    <style>
        div.nbinput.container div.prompt,
        div.nboutput.container div.prompt,
        span.prompt {
            display: none !important;
            min-width: 0 !important;
            padding: 0 !important;
        }
    </style>
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#import sphinx_book_theme

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Path to the logo relative to the configuration directory
html_logo = "_static/logo.png"

# Ensure the static path is included so Sphinx finds the file
html_static_path = ['_static']

# Link custom CSS (logo sizing and styling, located in '_static')
html_css_files = [
    "custom.css",
]

# Specific parameters passed directly to the 'sphinx_book_theme'.
html_theme_options = {
    # URL of the GitHub repository
    "repository_url": "https://github.com/CostaLab/Nichesphere",
    # GitHub button in the top navigation bar
    "use_repository_button": True,
    # Show a download button (e.g., PDF or Markdown/RST)
    "use_download_button": True,
    # Full-screen reading mode
    "use_fullscreen_button": True,
    # Collapse inactive subsections in the left sidebar to keep it tidy
    "collapse_navbar": True,
    # Set the maximum depth of heading levels (H1 to H4) shown in the right-hand page TOC
    "show_toc_level": 4,
    # Control how many navigation levels deep are automatically expanded in the left sidebar
    "show_navbar_depth": 2,
}