#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ringdown',
      version='0.1',
      description='Bayesian analysis of black hole ringdowns.',
      author='Maximiliano Isi, Will M. Farr',
      author_email='max.isi@ligo.org, will.farr@stonybrook.edu',
      url='https://github.com/maxisi/ringdown',
      license='MIT',
      packages=find_packages(),
      package_data={'ringdown': ['stan/*.stan']},
      scripts=['bin/ringdown_fit', 'bin/ringdown_pipe'],
      install_requires=[
            'arviz',
            'h5py',
            'lalsuite',
            'matplotlib',
            'numpy',
            'pandas',
            'pymc',
            'qnm',
            'scipy',
            'seaborn']
     )
