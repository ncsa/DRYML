# Testing Workflow

DRYML tests are organized by feature category and by speed tier.

Categories are the existing test directories, such as `core`, `environments`, `data`, `models`, `tf`, `torch`, and `jax`. Speed tiers are applied automatically from `tests/test_tiers.json` by the DRYML pytest timing plugin.

## Daily Commands

Capture a non-mutating external measurement artifact:

```bash
./tests.sh measure --output-dir /tmp/dryml-measure-smoke smoke
```

`measure` requires a fresh directory outside the repository. It preserves the
tier manifest and writes versioned `run.json`, `nodes.json`, per-phase timing
JSON and JUnit XML, and redacted stdout/stderr logs bounded to 1 MiB per stream.
The result records explicit success, invalidated, incomplete, or failure status;
unsupported modes fail at argument validation. Use `--invalidate-reason TEXT`
when host pressure or another reference condition makes an otherwise completed
sample incomparable. It is distinct from `profile`, which updates
`tests/test_tiers.json` after timing unknown nodes.
The nested repository commit is always required in a successful measurement.
When DRYML runs inside its coordination workspace, artifacts also record the
parent commit and tracked-clean state. Standalone checkouts, including normal
GitHub Actions jobs, report the parent repository as unavailable rather than
misclassifying an otherwise complete measurement.

`full` measurement enables combined coverage by default. CI and other callers
that need the same complete behavioral selection without coverage use:

```bash
./tests.sh measure --output-dir /tmp/dryml-measure-full --no-coverage full
```

On platforms without standard-library peak-child-memory support, the
machine-readable field is explicitly unavailable rather than reported as zero.

Run the fastest smoke bucket:

```bash
./tests.sh smoke
```

Run smoke plus medium tests:

```bash
./tests.sh medium
```

Run heavy tests only:

```bash
./tests.sh heavy
```

Run the full suite with coverage:

```bash
./tests.sh full --ignore tests/old --ignore tests/dev -x tests
```

The default `./tests.sh` behavior remains a full run with coverage. Internally, full runs execute smoke/medium files first and heavy files second. This avoids collecting heavyweight framework tests during the fast/mid part of the run while still producing combined coverage through `pytest-cov` append mode.
Requested XML, HTML, or other coverage reports are generated only by the heavy
phase after the combined coverage data is complete.
On Python 3.12 and newer with Coverage.py 7.4 or newer, the smoke/medium phase
uses Coverage.py's `sysmon` core for lower tracing overhead. The heavy append
phase uses `ctrace`; older Python or Coverage.py versions use `ctrace` for both
phases. Measurement artifacts record the core selected for each phase. Both
cores contribute ordinary line data to the same combined coverage target and
reports.

## Speed Tiers

`smoke` tests should be very fast and should avoid heavyweight imports, subprocesses, training, network access, dataset downloads, and framework initialization.

`medium` tests can cover Repo/Store integration, SQLite/query behavior, import-safety subprocess checks, current-environment inspection, and probe workers.

`heavy` tests include TensorFlow, Torch, JAX, Ray, multi-framework, MNIST/tfds, training, and other long integration paths.

## How Buckets Are Selected

`tests/test_tiers.json` stores the baseline policy:

```text
category_tiers:
    default tier for whole directories

path_tiers:
    file-level overrides

node_tiers:
    generated per-test overrides from profiling
```

`tests/tools/test_buckets.py` selects files for `tests.sh` before pytest collection. This matters because deselecting with `-m` alone still imports every collected test module. Selecting files first keeps smoke and medium runs from importing heavy test modules.

The pytest plugin in `tests/timing_plugin.py` then applies markers to collected tests:

```text
speed_smoke
speed_medium
speed_heavy
category_core
category_environments
...
```

You can still use ordinary pytest marker expressions:

```bash
pytest -m "speed_smoke and category_environments" tests/environments
```

## Profiling And Updating Buckets

Run a profiling pass:

```bash
./tests.sh profile --ignore tests/old --ignore tests/dev -x tests
```

This runs in two phases, writes `tests/.test-timings-medium.json` and `tests/.test-timings-heavy.json`, prints timing summaries, and merges both timing files into `tests/test_tiers.json` node-tier overrides for tests that passed.

When only newly added tests need node-tier timings, run:

```bash
./tests.sh profile --unknown-only
```

Unknown means a collected test nodeid is absent from `tests/test_tiers.json` `node_tiers`. Path tiers, category tiers, and default tiers still decide which profile phase collects the test, but only missing nodeids are executed and written to the timing output. This is the preferred workflow after adding tests to an existing file because the new tests inherit enough tier information to be collected, then get explicit node timings without rerunning all known tests.

The default thresholds are:

```text
<= 0.25s: smoke
<= 2.00s: medium
>  2.00s: heavy
```

Edit `tests/test_tiers.json` when a test should be pinned differently for semantic reasons. For example, a test that imports TensorFlow should remain heavy even if it happens to run quickly on one machine.

## Adding Tests

When adding tests:

1. Put the test in the category directory that best describes the feature.
2. Prefer small pure tests that can live in `smoke`.
3. Put integration, subprocess, SQLite, or import-safety tests in `medium` unless they are clearly heavyweight.
4. Keep framework imports, training, dataset-backed tests, and multi-framework tests in `heavy`.
5. Run `./tests.sh smoke` first, then `./tests.sh medium`, then relevant `heavy` tests.
6. Run `./tests.sh profile --unknown-only` to populate node-tier timings for new tests.

If a new file is not listed in `path_tiers`, it inherits its category tier.

## Dispatch And Code Benchmark

The deterministic dispatch/code-analysis benchmark emits one bounded versioned
JSON object. Its default mode runs only in-process scenarios with one warm-up
and 20 measured samples:

```bash
python benchmarks/dispatch_code_performance.py --mode pure
```

Managed probe and worker scenarios are explicit because they launch bounded
child processes and inspect the current Python environment:

```bash
python benchmarks/dispatch_code_performance.py --mode managed
```

Use `--pure-samples` up to 1,000, `--managed-samples` up to 100, and repeated
`--scenario NAME` selectors for focused measurements. The benchmark uses
temporary Stores, tracked standard-library targets, and fake resource inventory;
it does not use a network, GPU, downloads, credentials, datasets, or model
training.
