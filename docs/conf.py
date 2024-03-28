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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Vital Wave'
copyright = '2023, MW'
author = 'MW'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

extensions.append('sphinx_exec_directive')
extensions.append('matplotlib.sphinxext.plot_directive')
extensions.append('numpydoc')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

#html_theme = "sphinxawesome_theme"
html_theme = 'sphinx_rtd_theme' #'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

plot_pre_code = """
#import os
#import sys
#                
#import numpy as np
#from matplotlib import pyplot as plt
#                
#module_path = os.path.abspath(os.path.join('.'))
#                
#if module_path not in sys.path:
#    sys.path.append(module_path)
#                
##data_path = os.path.abspath(os.path.join('..\\src\\vitalwave\\example_data'))
#data_path = os.path.abspath(os.path.join('..\\data\\examples'))
#                
#print(data_path)
#                
#nd_ecg = np.load(data_path + "\\ecg_filt.npy")
#nd_ppg = np.load(data_path + "\\ppg_filt.npy")
#                
#fs = 200
#start = 0
#stop = 1000
#duration = (stop-start) / fs
"""
