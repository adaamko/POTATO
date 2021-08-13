import os

from setuptools import find_packages, setup
from setuptools.command.develop import develop

setup(
    name='exprel',
    version='0.1',
    description='NLP tools at TUW Informatics',
    url='https://github.com/adaamko/exp-relation-extraction',
    author='Adam Kovacs, Gabor Recski',
    author_email='adam.kovacs@tuwien.ac.at, gabor.recski@tuwien.ac.at',
    license='MIT',
    install_requires=[
        "beautifulsoup4",
        "tinydb",
        "pandas",
        "tqdm",
        "sklearn",
        "eli5",
        "matplotlib",
        "graphviz",
        "openpyxl"
    ],
    packages=find_packages(),
    zip_safe=False)
