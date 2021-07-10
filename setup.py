#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gwtones',
      version='0.1',
      description='Bayesian analysis of black hole ringdowns.',
      author='Maximiliano Isi, Will M. Farr',
      author_email='max.isi@ligo.org, will.farr@stonybrook.edu',
      url='https://github.com/maxisi/gwtones',
      license='MIT',
      packages=['gwtones'],
      package_data={'gwtones': ['stan/*.stan']},
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'lalsuite',
            'pandas',
            'arviz',
            'pystan>=2,<3',
            'Cython>=0.22'
            'seaborn',
            'qnm',
            'h5py'
      ]
     )
