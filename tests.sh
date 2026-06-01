#!/bin/bash
set -euo pipefail

default_dirs=(
    ./tests/artifacts
    ./tests/code
    ./tests/core
    ./tests/data
    ./tests/execute
    ./tests/jax
    ./tests/models
    ./tests/multi_framework
    ./tests/tf
    ./tests/torch
)

if [ "$#" -eq 0 ]; then
    pytest --cov=dryml "${default_dirs[@]}"
elif [[ "$1" == -* ]]; then
    pytest --cov=dryml "${default_dirs[@]}" "$@"
else
    pytest --cov=dryml "$@"
fi
