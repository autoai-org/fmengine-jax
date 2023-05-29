import os
import sys

project = 'FMTrainer'
copyright = '2023, Xiaozhe Yao'
author = 'Xiaozhe Yao'

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../fmtrainer'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']
