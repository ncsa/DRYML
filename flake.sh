#!/bin/bash
if [ -z "$@" ]; then
flake8 src/dryml/
flake8 src/bin/
flake8 tests/
else
flake8 $@
fi;
