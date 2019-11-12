#!/bin/bash

sphinx-build -b latex . latex

cd latex

pdflatex hvsrpy.tex

pdflatex hvsrpy.tex
