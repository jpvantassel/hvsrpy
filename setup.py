import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hvsrpy",
    version="0.1.0",
    author="Joseh P. Vantassel",
    author_email="jvantassel@utexas.edu",
    description="A Python module for horizontal-to-vertical spectra ratio processing",
    long_description=long_description,
    long_description_content_type="text/markdown.-",
    url="https://github.com/jpvantassel/hvsr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",

        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",

        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",

        'Programming Language :: Python :: 3.7',
    ],
)
