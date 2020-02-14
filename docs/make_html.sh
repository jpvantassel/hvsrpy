#!/bin/bash

sphinx-build -E -b html . html
cd html
ls
/c/Program\ Files/Mozilla\ Firefox/firefox index.html
