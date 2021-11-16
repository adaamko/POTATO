from setuptools import find_packages, setup

setup(
    name="potato",
    version="0.1",
    description="XAI human-in-the-loop information extraction framework",
    url="https://github.com/adaamko/exp-relation-extraction",
    author="Adam Kovacs, Gabor Recski",
    author_email="adam.kovacs@tuwien.ac.at, gabor.recski@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "beautifulsoup4",
        "tinydb",
        "pandas",
        "tqdm",
        "stanza",
        "sklearn",
        "eli5",
        "matplotlib",
        "graphviz",
        "openpyxl",
        "penman",
        "networkx",
        "streamlit",
        "streamlit-aggrid",
    ],
    packages=find_packages(),
    zip_safe=False,
)
