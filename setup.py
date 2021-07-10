#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gwtones',
      version='0.1',
      description='Bayesian analysis of black hole ringdowns.',
      author='Maximiliano Isi, Will M. Farr',
      author_email='max.isi@ligo.org, will.farr@stonybrook.edu;',
      url='https://github.com/maxisi/gwtones',
      packages=['gwtones'],
      package_data={'gwtones': ['stan/*.stan']},
     )
