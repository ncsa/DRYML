#!/bin/bash

if [ -z "$1" ]; then
    dirs=./tests
else
    dirs=$@
fi;

pytest -x --cov=dryml ${dirs}
