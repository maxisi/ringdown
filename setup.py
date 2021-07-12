#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='ringdown',
      version='0.1',
      description='Bayesian analysis of black hole ringdowns.',
      author='Maximiliano Isi, Will M. Farr',
      author_email='max.isi@ligo.org, will.farr@stonybrook.edu',
      url='https://github.com/maxisi/ringdown',
      license='MIT',
      packages=['ringdown'],
      package_data={'ringdown': ['stan/*.stan']},
      install_requires=[
            'Cython>=0.22',
            'arviz',
            'h5py',
            'lalsuite',
            'matplotlib',
            'numpy',
            'pandas',
            'pystan>=2,<3',
            'qnm',
            'scipy',
            'seaborn']
     )
