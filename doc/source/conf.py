# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lsdensities"
copyright = "2024, Alessandro Lupo, Niccolò Forzano"
author = "Alessandro Lupo, Niccolò Forzano"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

pygments_style = "sphinx"
html_theme = "classic"
html_static_path = ["_static"]
master_doc = "index"

latex_engine = "xelatex"  # or 'pdflatex' or 'lualatex', depending on your setup

# Grouping the document tree into LaTeX files.
latex_documents = [
    (
        master_doc,
        "Documentation.tex",
        "lsdensities Documentation",
        "Alessandro Lupo",
        "manual",
    ),
]

mathjax3_config = {
    "tex": {
        "macros": {
            "argmin": r"\mathop{\rm arg\,min}\limits",
        }
    }
}
