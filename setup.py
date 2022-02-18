#!/usr/bin/env python3

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

name = "stgfunc"
version = "0.0.6"
release = "0.0.6"

setuptools.setup(
    name=name,
    version=release,
    author="Setsugen no ao",
    author_email="setsugen@setsugen.dev",
    description="Vapoursynth functions and utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["stgfunc"],
    url="https://github.com/Setsugennoao/stgfunc",
    package_data={
        'stgfunc': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9'
)
