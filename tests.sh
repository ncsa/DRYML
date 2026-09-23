#!/bin/bash
set -euo pipefail

# Category directories intentionally reuse focused module names such as
# test_import_safety.py; importlib mode keeps broad collection collision-free.
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --import-mode=importlib"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

partition_process_state_tests() {
    local paths=("$@")
    process_state_selected=()
    ordinary_selected=()
    local path

    # Process-sensitive tests require a fresh interpreter so earlier optional
    # framework imports cannot invalidate their intentional mode transitions.
    for path in "${paths[@]}"; do
        [[ "$path" == ./tests/session/* ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        [[ "$path" == ./tests/runtime/* ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        [[ "$path" == ./tests/core/test_orchestrator_* ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        [[ "$path" == ./tests/dispatch/* ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        case "$path" in
            ./tests/session/*|./tests/runtime/*|./tests/core/test_orchestrator_*|./tests/dispatch/*) ;;
            *) ordinary_selected+=("$path") ;;
        esac
    done
}

run_tier() {
    local tier_name="$1"
    shift
    local tiers=()
    local tier_filter=""

    case "$tier_name" in
        smoke)
            tiers=(smoke)
            tier_filter="smoke"
            ;;
        medium)
            tiers=(smoke medium)
            tier_filter="smoke,medium"
            ;;
        heavy)
            tiers=(heavy)
            tier_filter="heavy"
            ;;
        *)
            echo "Unknown test tier: $tier_name" >&2
            exit 2
            ;;
    esac

    prepare_suite_args "$@"
    mapfile -t selected < <(python ./tests/tools/test_buckets.py select "${tiers[@]}")
    if [ "${#selected[@]}" -eq 0 ]; then
        echo "No tests selected for tier: $tier_name" >&2
        exit 2
    fi
    if [ "$tier_name" == "heavy" ]; then
        DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov --dryml-runner-tiers "$tier_filter" "${selected[@]}" "${stripped_args[@]}"
    else
        partition_process_state_tests "${selected[@]}"
        if [ "${#process_state_selected[@]}" -gt 0 ]; then
            pytest --no-cov --dryml-runner-tiers "$tier_filter" "${process_state_selected[@]}" "${stripped_args[@]}"
        fi
        if [ "${#ordinary_selected[@]}" -gt 0 ]; then
            pytest --no-cov --dryml-runner-tiers "$tier_filter" "${ordinary_selected[@]}" "${stripped_args[@]}"
        fi
    fi
}

prepare_suite_args() {
    local root_index
    if ! root_index="$(python ./tests/tools/test_buckets.py runner-args -- "$@")"; then
        return 2
    fi
    stripped_args=()
    local index=0
    for arg in "$@"; do
        if [ "$index" -ne "$root_index" ]; then
            stripped_args+=("$arg")
        fi
        index=$((index + 1))
    done
}

run_full() {
    prepare_suite_args "$@"
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    partition_process_state_tests "${medium_selected[@]}"
    pytest --cov=dryml --dryml-runner-tiers smoke,medium "${process_state_selected[@]}" "${stripped_args[@]}"
    pytest --cov=dryml --cov-append --dryml-runner-tiers smoke,medium "${ordinary_selected[@]}" "${stripped_args[@]}"
    DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --cov=dryml --cov-append --dryml-runner-tiers heavy "${heavy_selected[@]}" "${stripped_args[@]}"
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
    prepare_suite_args "${profile_args[@]}"
    local medium_output="./tests/.test-timings-medium.json"
    local process_state_output="./tests/.test-timings-process-state.json"
    local heavy_output="./tests/.test-timings-heavy.json"
    local unknown_args=()
    if [ "$unknown_only" -eq 1 ]; then
        unknown_args=(--dryml-timing-unknown-only)
    fi
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    partition_process_state_tests "${medium_selected[@]}"
    pytest --no-cov --dryml-runner-tiers smoke,medium "${process_state_selected[@]}" --dryml-timing-output "$process_state_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    pytest --no-cov --dryml-runner-tiers smoke,medium "${ordinary_selected[@]}" --dryml-timing-output "$medium_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov --dryml-runner-tiers heavy "${heavy_selected[@]}" --dryml-timing-output "$heavy_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    python ./tests/tools/test_buckets.py update "$process_state_output" "$medium_output" "$heavy_output"
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
elif [[ "$1" == -* ]]; then
    run_full "$@"
else
    pytest --cov=dryml "$@"
fi
