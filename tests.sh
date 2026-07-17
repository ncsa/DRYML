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
        pytest --no-cov -m "$markexpr" "${selected[@]}" "$@"
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

strip_coverage_reports() {
    coverage_stripped_args=()
    local skip_value=0
    for arg in "$@"; do
        if [ "$skip_value" -eq 1 ]; then
            skip_value=0
            continue
        fi
        case "$arg" in
            --cov-report)
                skip_value=1
                ;;
            --cov-report=*)
                ;;
            *)
                coverage_stripped_args+=("$arg")
                ;;
        esac
    done
}

run_full() {
    strip_suite_paths "$@"
    strip_coverage_reports "${stripped_args[@]}"
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    medium_coverage_core="ctrace"
    if python -c 'import sys; from importlib.metadata import version; from packaging.version import Version; raise SystemExit(sys.version_info < (3, 12) or Version(version("coverage")) < Version("7.4"))'; then
        medium_coverage_core="sysmon"
    fi
    COVERAGE_CORE="$medium_coverage_core" pytest --cov=dryml --cov-report= -m "speed_smoke or speed_medium" "${medium_selected[@]}" "${coverage_stripped_args[@]}"
    COVERAGE_CORE=ctrace pytest --cov=dryml --cov-append -m "speed_heavy" "${heavy_selected[@]}" "${stripped_args[@]}"
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
    pytest --no-cov -m "speed_heavy" "${heavy_selected[@]}" --dryml-timing-output "$heavy_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
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
