#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup  #, find_packages
from setuptools.extension import Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# See details on Cython setuptools setup.py:
# https://stackoverflow.com/questions/32528560/using-setuptools-to-create-a-cython-package-calling-an-external-c-library
# See details on Cython distutils setup.py:

extensions = [
	Extension(
		'similarities',
		['similarities.pyx'],
		# "myPackage.myModule",
		# ["myPackage/myModule.pyx"],
		# include_dirs=[],
		libraries=['m'],  # libc
		# library_dirs=[]
	)
]

setup(
	# name = "similarities",
	# packages = find_packages(),
	# ext_modules=cythonize("src/*.pyx", include_path=[...]),
	# ext_modules = cythonize("similarities.pyx", include_path=[np.get_include()])
	ext_modules = cythonize(extensions)
)
