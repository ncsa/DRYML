# Testing Workflow

DRYML tests are organized by feature category and by speed tier.

Maintained categories include `formats`, `annotations`, `environments`,
`worlds`, `runtime`, `session`, `core`, `package`, `data`, `execute`, `models`,
`ray`, `tf`, `torch`, `jax`, and `multi_framework`. Speed tiers are applied
automatically from `tests/test_tiers.json` by the DRYML pytest timing plugin.

## Daily Commands

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

The default `./tests.sh` behavior remains a full run with coverage. Internally,
full runs execute process-global session/runtime/orchestrator tests in a fresh
phase, other smoke/medium files in a second phase, and heavy files last. This
keeps intentional late-import and terminal-publication tests isolated while
combining coverage through `pytest-cov` append mode.

## Speed Tiers

`smoke` tests should be very fast and should avoid heavyweight imports, subprocesses, training, network access, dataset downloads, and framework initialization.

`medium` tests can cover Repo/Store integration, SQLite/query behavior, import-safety subprocess checks, current-environment inspection, and probe workers.

`heavy` tests include TensorFlow, Torch, JAX, Ray, multi-framework, MNIST/tfds, training, and other long integration paths.

`package` tests build an sdist and wheel beneath `/tmp/dryml`, inspect their
contents, install the wheel into an isolated interpreter, verify exact public
exports, and prove declaration imports remain free of optional frameworks.

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
7. Keep only classifications for the tests added or changed by the current work;
   remove profiler spillover for unrelated existing nodes before review.

## CDef V2 Gates

The focused CDef V2 gate covers rejection of pre-port authority before mutation,
public signatures and exports, exact-reference transport rejection, lazy Ray
imports, and tracked documentation links/API examples. Use `tests/fixtures/`
only for minimal malformed authority fixtures; do not add migration or dual-read
fixtures. Run the focused gate before broader core tests and do not weaken its
rejection assertions.

If a new file is not listed in `path_tiers`, it inherits its category tier.

The tier-administration tests require every maintained category to have an
explicit default, reject stale metadata paths, and prove the union of all tiers
covers every maintained test file.

## Hosted Matrix

The lightweight matrix installs only package and test dependencies on Ubuntu
and Windows for Python 3.10 through 3.14, then runs smoke/medium and installed
artifact checks. Python 3.14 is explicitly framework-reduced.

The heavy matrix runs on Ubuntu for Python 3.10 through 3.13. It installs and
preflights TensorFlow, Torch, JAX/JAXlib, and Ray before heavy tests so missing
or broken frameworks fail rather than skip. Each job prints the resolved Python,
DRYML, and framework versions. Workflow configuration is not support evidence
until the jobs pass on the exact child commit.

## Context Bootstrap

Historically, test startup initialized JAX, Torch, and TensorFlow contexts for every run. That made even focused core tests pay framework import cost and could perturb concurrency tests.

Context bootstrap is now opt-in:

```bash
DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest ...
```

`./tests.sh heavy`, `./tests.sh full`, and `./tests.sh profile` enable this only for the heavy phase. Smoke and medium runs leave global contexts uninitialized unless an individual test initializes what it needs.
