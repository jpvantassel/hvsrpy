"""A setuptools based setup module."""

from setuptools import setup, find_packages


def parse_meta(path_to_meta):
    with open(path_to_meta) as f:
        meta = {}
        for line in f.readlines():
            if line.startswith("__version__"):
                meta["__version__"] = line.split('"')[1]
    return meta


meta = parse_meta("hvsrpy/meta.py")

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='hvsrpy',
    version=meta['__version__'],
    description='A Python package for horizontal-to-vertical spectral ratio processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/hvsrpy',
    author='Joseph P. Vantassel',
    author_email='jvantassel@utexas.edu',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

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
        'Programming Language :: Python :: 3.9',
    ],
    keywords='horizontal-to-vertical spectral ratio hv hvsr',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'obspy', 'sigpropy>=1.0.0',
                      'pandas', 'shapely', 'termcolor', 'matplotlib', 'click>8.0.0'],
    extras_require={
        'dev': ['coverage', 'sphinx-click'],
    },
    package_data={
    },
    data_files=[
    ],
    entry_points={
        'console_scripts': [
            'hvsrpy = hvsrpy.cli:cli'
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/jpvantassel/hvsrpy/issues',
        'Source': 'https://github.com/jpvantassel/hvsrpy',
        'Docs': 'https://hvsrpy.readthedocs.io/en/latest/?badge=latest',
    },
)
