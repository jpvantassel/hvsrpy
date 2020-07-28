"""A setuptools based setup module."""

from setuptools import setup, find_packages
with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='hvsrpy',
    version='0.4.2',
    description='A Python package for horizontal-to-vertical spectral ratio processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/hvsr',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='horizontal-to-vertical spectral ratio hv hvsr',
    packages=find_packages(),
    python_requires = '>=3.6, <3.9',
    install_requires=['numpy<1.19.0', 'scipy', 'obspy', 'sigpropy', 'pandas', 'shapely', 'termcolor'],
    extras_require={
        'dev': ['matplotlib', 'coverage'],
    },
    package_data={
    },
    data_files=[
        ],
    entry_points={
    },
    project_urls={
        'Bug Reports': 'https://github.com/jpvantassel/hvsrpy/issues',
        'Source': 'https://github.com/jpvantassel/hvsrpy',
        'Docs': 'https://hvsrpy.readthedocs.io/en/latest/?badge=latest',
    },
)