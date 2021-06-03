#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gwtones',
      version='0.1',
      description='Bayesian analysis of black hole ringdowns.',
      author='Maximiliano Isi',
      author_email='max.isi@ligo.org',
      url='https://',
      packages=['gwtones'],
      package_data={'gwtones': ['stan/*.stan']},
     )
