#!/bin/bash
set -euo pipefail

default_dirs=(
    ./tests/artifacts
    ./tests/code
    ./tests/core
    ./tests/data
    ./tests/environments
    ./tests/execute
    ./tests/jax
    ./tests/models
    ./tests/multi_framework
    ./tests/tf
    ./tests/torch
)

run_tier() {
    local tier_name="$1"
    shift
    local tiers=()
    local markexpr=""

    case "$tier_name" in
        smoke)
            tiers=(smoke)
            markexpr="speed_smoke"
            ;;
        medium)
            tiers=(smoke medium)
            markexpr="speed_smoke or speed_medium"
            ;;
        heavy)
            tiers=(heavy)
            markexpr="speed_heavy"
            ;;
        *)
            echo "Unknown test tier: $tier_name" >&2
            exit 2
            ;;
    esac

    mapfile -t selected < <(python ./tests/tools/test_buckets.py select "${tiers[@]}")
    if [ "${#selected[@]}" -eq 0 ]; then
        echo "No tests selected for tier: $tier_name" >&2
        exit 2
    fi
    if [ "$tier_name" == "heavy" ]; then
        DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov -m "$markexpr" "${selected[@]}" "$@"
    else
        pytest --no-cov -m "$markexpr" "${selected[@]}" "$@"
    fi
}

strip_suite_paths() {
    stripped_args=()
    for arg in "$@"; do
        case "$arg" in
            tests|./tests)
                ;;
            *)
                stripped_args+=("$arg")
                ;;
        esac
    done
}

run_full() {
    strip_suite_paths "$@"
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    pytest --cov=dryml -m "speed_smoke or speed_medium" "${medium_selected[@]}" "${stripped_args[@]}"
    DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --cov=dryml --cov-append -m "speed_heavy" "${heavy_selected[@]}" "${stripped_args[@]}"
}

run_profile() {
    local unknown_only=0
    local profile_args=()
    for arg in "$@"; do
        case "$arg" in
            --unknown-only)
                unknown_only=1
                ;;
            *)
                profile_args+=("$arg")
                ;;
        esac
    done
    strip_suite_paths "${profile_args[@]}"
    local medium_output="./tests/.test-timings-medium.json"
    local heavy_output="./tests/.test-timings-heavy.json"
    local unknown_args=()
    if [ "$unknown_only" -eq 1 ]; then
        unknown_args=(--dryml-timing-unknown-only)
    fi
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    pytest --no-cov -m "speed_smoke or speed_medium" "${medium_selected[@]}" --dryml-timing-output "$medium_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov -m "speed_heavy" "${heavy_selected[@]}" --dryml-timing-output "$heavy_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    python ./tests/tools/test_buckets.py update "$medium_output" "$heavy_output"
    python ./tests/tools/test_buckets.py summary --all-files
}

if [ "$#" -eq 0 ]; then
    run_full
elif [[ "$1" == "smoke" || "$1" == "medium" || "$1" == "heavy" ]]; then
    mode="$1"
    shift
    run_tier "$mode" "$@"
elif [[ "$1" == "full" ]]; then
    shift
    run_full "$@"
elif [[ "$1" == "profile" ]]; then
    shift
    run_profile "$@"
elif [[ "$1" == "measure" ]]; then
    shift
    python ./tests/tools/measure_suite.py "$@"
elif [[ "$1" == -* ]]; then
    run_full "$@"
else
    pytest --cov=dryml "$@"
fi
