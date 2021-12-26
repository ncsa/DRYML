#!/bin/bash

if [ -z "$@" ]; then
    dirs=./tests
else
    dirs=$@
fi;

pytest --cov=dryml ${dirs}
