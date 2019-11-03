import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hvsrpy",
    version="0.0.1",
    author="Joseh P Vantassel",
    author_email="jvantassel@utexas.edu",
    description="Tools for Horizontal-to-Vertical Spectral Ratio (H/V, HVSR) processing.",
    long_description=long_description,
    long_description_content_type="text/markdown.-",
    url="https://github.com/jpvantassel/hvsr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Liscence :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
)
