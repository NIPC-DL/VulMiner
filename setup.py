# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name="vulminer",
    version="0.1",
    packages=find_packages(),
    scripts=['bin/vulminer.py'],
    install_requires=[''],

    package_data={
        '': ['*.yaml', ],
    },

    # metadata to display on PyPI
    author="NIPC-DL",
    author_email="verf@protonmail.com",
    description="Vulnerability Mining Fromwork with Deep Learning",
    license="MIT",
    keywords="Vulnerability Mining, Deep Learning",
    url="https://github.com/NIPC-DL/VulMiner",
)
