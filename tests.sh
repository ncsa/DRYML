#!/bin/bash

if [ -z "$1" ]; then
    dirs=./tests
else
    dirs=$@
fi;

pytest --cov=dryml --ignore=./tests/old ${dirs}
