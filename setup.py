#!/usr/bin/env python3

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

setuptools.setup(
    name="stgfunc",
    version="0.1.1",
    author="Setsugen no ao",
    author_email="setsugen@setsugen.dev",
    description="VapourSynth functions and utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["stgfunc", "stgfunc.utils", "stgfunc.shaders"],
    url="https://github.com/Setsugennoao/stgfunc",
    package_data={
        'stgfunc': ['py.typed', '*.json', 'shaders/*.glsl'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10'
)
