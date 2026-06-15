"""Configuration file for the Sphinx documentation builder."""

#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "vfp"
copyright = "2024-2026, Alexander Armstrong"  # noqa : A001
author = "Alexander Armstrong, Rebecca Welbourn"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

nb_execution_mode = "off"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# stop type hints appearing in function definition.
autodoc_typehints = "none"
# docs for class has class & init docstring.
autoapi_python_class_content = "both"

html_theme_options = {
    "repository_url": "https://github.com/Armatron44/VFP",
    "use_repository_button": True,
}
