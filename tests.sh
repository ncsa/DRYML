#!/bin/bash
set -euo pipefail

# Category directories intentionally reuse focused module names such as
# test_import_safety.py; importlib mode keeps broad collection collision-free.
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --import-mode=importlib"
# Native Python chooses the path separator, including nested Git Bash on Windows.
PYTHONPATH="$(python -c 'import os; print(os.getcwd() + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))')"
export PYTHONPATH

phase_matched=0

run_phase() {
    local status
    if "$@"; then
        phase_matched=1
        return 0
    else
        status=$?
    fi
    if [ "$status" -eq 5 ]; then
        return 0
    fi
    return "$status"
}

finish_phases() {
    if [ "$phase_matched" -eq 0 ]; then
        echo "No tests matched any runner phase." >&2
        return 5
    fi
}

coverage_control_args() {
    pytest_coverage_args=(--no-cov)
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--cov" || "$arg" == --cov=* ]]; then
            pytest_coverage_args=()
            return
        fi
    done
}

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
        [[ "$path" == ./tests/core/test_execute_worker_context.py ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        [[ "$path" == ./tests/dispatch/* ]] && process_state_selected+=("$path")
    done
    for path in "${paths[@]}"; do
        case "$path" in
            ./tests/session/*|./tests/runtime/*|./tests/core/test_orchestrator_*|./tests/core/test_execute_worker_context.py|./tests/dispatch/*) ;;
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
        run_phase env DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov --dryml-runner-tiers "$tier_filter" "${selected[@]}" "${stripped_args[@]}"
    else
        partition_process_state_tests "${selected[@]}"
        if [ "${#process_state_selected[@]}" -gt 0 ]; then
            run_phase pytest --no-cov --dryml-runner-tiers "$tier_filter" "${process_state_selected[@]}" "${stripped_args[@]}"
        fi
        if [ "${#ordinary_selected[@]}" -gt 0 ]; then
            run_phase pytest --no-cov --dryml-runner-tiers "$tier_filter" "${ordinary_selected[@]}" "${stripped_args[@]}"
        fi
    fi
    finish_phases
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

run_maintained() {
    prepare_suite_args "$@"
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    partition_process_state_tests "${medium_selected[@]}"
    run_phase pytest --no-cov --dryml-runner-tiers smoke,medium "${process_state_selected[@]}" "${stripped_args[@]}"
    run_phase pytest --no-cov --dryml-runner-tiers smoke,medium "${ordinary_selected[@]}" "${stripped_args[@]}"
    run_phase env DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov --dryml-runner-tiers heavy "${heavy_selected[@]}" "${stripped_args[@]}"
    finish_phases
}

run_good_enough() {
    prepare_suite_args "$@"
    mapfile -t selected < <(python ./tests/tools/test_buckets.py select-profile good-enough smoke medium)
    if [ "${#selected[@]}" -eq 0 ]; then
        echo "No test files selected for profile: good-enough" >&2
        return 2
    fi
    echo "DRYML good-enough: smoke+medium with named integration representatives; package and heavy tests excluded." >&2
    partition_process_state_tests "${selected[@]}"
    if [ "${#process_state_selected[@]}" -gt 0 ]; then
        run_phase pytest --no-cov --dryml-runner-tiers smoke,medium --dryml-test-profile good-enough "${process_state_selected[@]}" "${stripped_args[@]}"
    fi
    if [ "${#ordinary_selected[@]}" -gt 0 ]; then
        run_phase pytest --no-cov --dryml-runner-tiers smoke,medium --dryml-test-profile good-enough "${ordinary_selected[@]}" "${stripped_args[@]}"
    fi
    finish_phases
}

run_coverage() {
    prepare_suite_args "$@"
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    partition_process_state_tests "${medium_selected[@]}"
    run_coverage_phase smoke,medium "${process_state_selected[@]}"
    run_coverage_phase smoke,medium "${ordinary_selected[@]}"
    DRYML_TEST_BOOTSTRAP_CONTEXTS=1 run_coverage_phase heavy "${heavy_selected[@]}"
    finish_phases
}

run_coverage_phase() {
    local tier_filter="$1"
    shift
    local append_args=()
    if [ "$phase_matched" -eq 1 ]; then
        append_args=(--cov-append)
    fi
    run_phase pytest --cov=dryml "${append_args[@]}" --dryml-runner-tiers "$tier_filter" "$@" "${stripped_args[@]}"
}

run_package() {
    prepare_suite_args "$@"
    mapfile -t selected < <(python ./tests/tools/test_buckets.py select-category package)
    coverage_control_args "${stripped_args[@]}"
    run_phase pytest "${pytest_coverage_args[@]}" "${selected[@]}" "${stripped_args[@]}"
    finish_phases
}

run_focused() {
    coverage_control_args "$@"
    pytest "${pytest_coverage_args[@]}" "$@"
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
    local output_dir="/tmp/dryml/profile"
    mkdir -p "$output_dir"
    local medium_output="$output_dir/test-timings-medium.json"
    local process_state_output="$output_dir/test-timings-process-state.json"
    local heavy_output="$output_dir/test-timings-heavy.json"
    local unknown_args=()
    if [ "$unknown_only" -eq 1 ]; then
        unknown_args=(--dryml-timing-unknown-only)
    fi
    mapfile -t medium_selected < <(python ./tests/tools/test_buckets.py select smoke medium)
    mapfile -t heavy_selected < <(python ./tests/tools/test_buckets.py select heavy)
    partition_process_state_tests "${medium_selected[@]}"
    run_phase pytest --no-cov --dryml-runner-tiers smoke,medium "${process_state_selected[@]}" --dryml-timing-output "$process_state_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    run_phase pytest --no-cov --dryml-runner-tiers smoke,medium "${ordinary_selected[@]}" --dryml-timing-output "$medium_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    run_phase env DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest --no-cov --dryml-runner-tiers heavy "${heavy_selected[@]}" --dryml-timing-output "$heavy_output" --dryml-timing-summary "${unknown_args[@]}" "${stripped_args[@]}"
    finish_phases
    python ./tests/tools/test_buckets.py update "$process_state_output" "$medium_output" "$heavy_output"
    python ./tests/tools/test_buckets.py summary --all-files
}

if [ "$#" -eq 0 ]; then
    run_good_enough
elif [[ "$1" == "smoke" || "$1" == "medium" || "$1" == "heavy" ]]; then
    mode="$1"
    shift
    run_tier "$mode" "$@"
elif [[ "$1" == "good-enough" ]]; then
    shift
    run_good_enough "$@"
elif [[ "$1" == "full" || "$1" == "exhaustive" ]]; then
    shift
    run_maintained "$@"
elif [[ "$1" == "coverage" ]]; then
    shift
    run_coverage "$@"
elif [[ "$1" == "package" ]]; then
    shift
    run_package "$@"
elif [[ "$1" == "profile" ]]; then
    shift
    run_profile "$@"
elif [[ "$1" == -* ]]; then
    run_good_enough "$@"
else
    run_focused "$@"
fi
