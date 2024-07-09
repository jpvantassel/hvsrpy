"""A setuptools based setup module."""

from setuptools import setup, find_packages


def parse_meta(path_to_metadata):
    with open(path_to_metadata) as f:
        metadata = {}
        for line in f.readlines():
            if line.startswith("__version__"):
                metadata["__version__"] = line.split('"')[1]
    return metadata


metadata = parse_meta("hvsrpy/metadata.py")


RELATIVE_TO_ABSOLUTE_FIGURES = {
    '![Traditional HVSR processing with window rejection.](./figs/example_hvsr_figure.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/example_hvsr_figure.png?raw=true" width="775">',
    '![Azimuthal HVSR processing.](./figs/example_hvsr_figure_az.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/example_hvsr_figure_az.png?raw=true" width="775">',
    '![Spatial HVSR processing.](./figs/example_hvsr_figure_sp.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/example_hvsr_figure_sp.png?raw=true" width="775">',
    '![Multi-window example STN11_c050.](./figs/multiwindow_STN11_c050.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/multiwindow_STN11_c050.png?raw=true" width="425">',
    '![Multi-window example STN11_c150.](./figs/multiwindow_STN11_c150.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/multiwindow_STN11_c150.png?raw=true" width="425">',
    '![Multi-window example STN12_c050.](./figs/multiwindow_STN12_c050.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/multiwindow_STN12_c050.png?raw=true" width="425">',
    '![Multi-window example STN12_c150.](./figs/multiwindow_STN12_c150.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/multiwindow_STN12_c150.png?raw=true" width="425">',
    '![Single window example a.](./figs/singlewindow_a.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_a.png?raw=true" width="425">',
    '![Single window example b.](./figs/singlewindow_b.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_b.png?raw=true" width="425">',
    '![Single window example c.](./figs/singlewindow_c.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_c.png?raw=true" width="425">',
    '![Single window example d.](./figs/singlewindow_d.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_d.png?raw=true" width="425">',
    '![Single window example e.](./figs/singlewindow_e.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_e.png?raw=true" width="425">',
    '![Single window example f.](./figs/singlewindow_f.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_f.png?raw=true" width="425">',
    '![Single window example g.](./figs/singlewindow_g.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_g.png?raw=true" width="425">',
    '![Single window example h.](./figs/singlewindow_h.png)': '<img src="https://github.com/jpvantassel/hvsrpy/blob/main/figs/singlewindow_h.png?raw=true" width="425">',
}


with open("README.md", encoding="utf8") as f:
    long_description = f.read()
    for old_text, new_text in RELATIVE_TO_ABSOLUTE_FIGURES.items():
        long_description = long_description.replace(old_text, new_text)

setup(
    name='hvsrpy',
    version=metadata['__version__'],
    description='A Python package for horizontal-to-vertical spectral ratio processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jpvantassel/hvsrpy',
    author='Joseph P. Vantassel',
    author_email='joseph.p.vantassel@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='horizontal-to-vertical spectral ratio hv hvsr',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy>=1.22', 'scipy', 'obspy',
                      'pandas', 'shapely', 'termcolor', 'matplotlib',
                      'click>8.0.0', 'numba', 'PyQt5'],
    extras_require={
        'dev': ['tox', 'jupyterlab', 'coverage', 'sphinx', 'sphinx_rtd_theme', 'sphinx-click', 'autopep8'],
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
