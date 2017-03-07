from setuptools import setup
import version

readme = open('README.rst').read()

requirements = ['progressbar33', 'enum34', 'seaborn']

setup(
    name="zcode",
    version=version.version,
    author="Luke Zoltan Kelley",
    author_email="lkelley@cfa.harvard.edu",
    description=("General, commonly used functions for other projects."),
    license="MIT",
    keywords="",
    url="https://bitbucket.org/lzkelley/zcode/",
    packages=['zcode'],
    include_package_data=True,
    install_requires=requirements,
    long_description=readme,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
)
