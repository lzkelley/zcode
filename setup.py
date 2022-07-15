"""Utility functions and tools from Luke Zoltan Kelley.
"""

from setuptools import setup, find_packages

short_description = __doc__.strip()

with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

with open("README.rst", "r") as inn:
    long_description = inn.read().strip()

with open('zcode/VERSION') as inn:
    version = inn.read().strip()

setup(
    name="zcode",
    author="Luke Zoltan Kelley",
    author_email="lzkelley@gmail.com",
    url="https://github.com/lzkelley/zcode/",
    version=version,
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    # External dependencies loaded from 'requirements.txt'
    install_requires=requirements,
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Python version restrictions
    python_requires=">=3.9",
)