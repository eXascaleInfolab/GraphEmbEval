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
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# 
# Compilation: $ python setup.py build_ext --inplace

extensions = [
	Extension(
		'similarities',
		['similarities.pyx'],
		# "myPackage.myModule",
		# ["myPackage/myModule.pyx"],
		include_dirs=[np.get_include()],
		libraries=['m'],  # libc
		# library_dirs=[]
	)
]

setup(
	setup_requires=[
		'cython>=0.21',
	],
	# name = "similarities",
	# packages = find_packages(),
	# ext_modules = cythonize("src/*.pyx", include_path=[...]),
	# ext_modules = cythonize("similarities.pyx", include_path=[np.get_include()]),
	# cmdclass={'build_ext': Cython.Build.build_ext},
	ext_modules = cythonize(extensions)
)
