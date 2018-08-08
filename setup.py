#!/usr/bin/env python3
#coding: utf-8

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = 'VulMiner',
    version = '0.1',
    packages = find_packages(exclude=('tests', 'docs')),

    entry_points = {
        'console_scripts': [
            'vulminer = entry: main',
            ],
        }

    author = 'NIPC-DL',
    author_email = 'vo4f@outlook.com',
    description = 'Vulnerable Mining Framwork with Deep Learning',
    long_description = readme,
    license = license,
    keywords = "Vulnerablity Mining, Deep Learning"
    url = 'https://github.com/NIPC-DL/VulMiner',
)
