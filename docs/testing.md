# Testing Workflow

DRYML tests are organized by feature category and by speed tier.

Maintained categories include `formats`, `annotations`, `environments`,
`worlds`, `runtime`, `session`, `core`, `locking`, `managed`, `package`, `data`, `execute`, `models`,
`ray`, `tf`, `torch`, `jax`, and `multi_framework`. `dispatch` is a maintained
medium category. Speed tiers are applied
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

The `medium` suite already includes `smoke`; use `medium` alone rather than
running both as a default local sequence.

Run heavy tests only:

```bash
./tests.sh heavy
```

Run the full suite with coverage:

```bash
./tests.sh full --ignore tests/old --ignore tests/dev -x tests
```

The default `./tests.sh` behavior remains a full run with coverage. Internally,
full runs execute process-sensitive session/runtime/orchestrator/dispatch tests
in a fresh phase, other smoke/medium files in a second phase, and heavy files
last. This keeps intentional late-import and terminal-publication tests
isolated while combining coverage through `pytest-cov` append mode.

For a focused run, put test paths or node IDs first and disable coverage
explicitly when it is not needed:

```bash
./tests.sh tests/core/test_repo_save_load.py --no-cov -x
```

Named `smoke`, `medium`, `heavy`, `full`, and `profile` suites accept pytest
options but reject explicit test selections; put focused paths first instead.
Options such as `-k` and `-m` further restrict each named suite and cannot
broaden its speed-tier selection.

## Policy Test Evidence

Policy-only tests can request the explicit `synthetic_environment_record`
fixture instead of scanning installed distributions. It supplies a small
controlled package inventory with portable interpreter paths and rejects real
inventory access for the duration of that test, including attempts caught by
the code under test. It is not an autouse fixture: current-environment
inspection, actual worker probes, and integration tests retain real observation
where observation itself is the contract being verified.

Test-tier administration shares one maintained-node collection per pytest
session. Both stale-node and intentional-tier checks use that snapshot;
collection failures still fail the checks. This does not cache the standalone
tier-management CLI across calls or change tier membership.

## Speed Tiers

`smoke` tests should be very fast and should avoid heavyweight imports, subprocesses, training, network access, dataset downloads, and framework initialization.

`medium` tests can cover Repo/Store integration, SQLite/query behavior, import-safety subprocess checks, current-environment inspection, and probe workers.

`heavy` tests include TensorFlow, Torch, JAX, Ray, multi-framework, MNIST/tfds, training, and other long integration paths.

`package` tests build an sdist and wheel beneath `/tmp/dryml`, inspect their
contents, install the wheel into an isolated interpreter, verify exact public
exports, and prove declaration imports remain free of optional frameworks.

`managed` tests cover synchronous lifecycle publication, selected Store authority,
checkpoint callbacks, interruption boundaries, resume/rerun recovery, local
ownership, and lightweight root/package imports. Use focused checks while working:

```bash
./tests.sh tests/managed tests/package/test_public_imports.py tests/package/test_release_artifacts.py
```

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

This runs process-sensitive, other smoke/medium, and heavy phases. It writes
`tests/.test-timings-process-state.json`, `tests/.test-timings-medium.json`,
and `tests/.test-timings-heavy.json`, prints timing summaries, and merges all
three timing files into `tests/test_tiers.json` node-tier overrides for tests
that passed.

When only newly added tests need node-tier timings, run:

```bash
./tests.sh profile --unknown-only
```

Unknown means a collected test nodeid is absent from `tests/test_tiers.json`
`node_tiers`. It does not mean recently added: the filter excludes only IDs
known specifically to `node_tiers`. Path tiers, category tiers, and default
tiers still decide which profile phase collects the test, but they do not make
a node known for this filter. Only missing nodeids are executed and written to
the timing output. This is the preferred workflow after adding tests to an
existing file because the new tests inherit enough tier information to be
collected, then get explicit node timings without rerunning all node-tier-known
tests.

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
3. Put integration, subprocess, SQLite, locking, or import-safety tests in `medium` unless they are clearly heavyweight.
4. Keep framework imports, training, dataset-backed tests, and multi-framework tests in `heavy`.
5. Run `./tests.sh medium` (which includes smoke), then relevant `heavy` tests.
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

Lightweight CI jobs print individual test names and use `pytest-timeout`'s
thread watchdog with a 180-second per-test limit (including fixture work).
A stalled test dumps thread stacks and fails the process instead of hanging
indefinitely. The job also has a 20-minute limit. Local commands have no
per-test timeout unless the caller supplies one explicitly.

The heavy matrix runs on Ubuntu for Python 3.10 through 3.13. It installs and
preflights TensorFlow, Torch, JAX/JAXlib, and pinned `ray[default]==2.56.0` before heavy tests so missing
or broken frameworks fail rather than skip. Each job prints the resolved Python,
DRYML, and framework versions. Workflow configuration is not support evidence
until the jobs pass on the exact child commit.

The separate `Existing Ray integration (Ubuntu, Python 3.12)` job uses an
ephemeral Conda coordinator and a venv derived from that same interpreter. It
builds DRYML once, installs that artifact with `dill` and
`ray[default]==2.56.0` into both existing test targets, starts a job-owned
single-node 4-CPU/0-GPU Ray fixture, and enables the real Ray and
existing-environment Execute tests. The enabled selection also covers Dispatch
local-probe/Ray-workload, Ray-probe/local-workload, Ray-probe/in-process,
same-config independent ownership, and exact Conda/venv pin identity. Its final
cleanup stops only the fixture it started. This CI preparation does not change the product contract: normal DRYML
runtime and tests require caller-supplied existing environments and an existing
Ray address, and never provision them.

The lightweight Ubuntu/Windows Python 3.10 through 3.14 matrix remains
framework-reduced, does not install Ray, and supplies common/subprocess/package
coverage. It is not native Windows Ray or GPU evidence. Workflow configuration
becomes evidence only after the remote job passes for the pushed child revision.

Enabled existing-target Ray coverage is opt-in. Do not describe it as passed until
the caller-provided Ray server, Conda prefix, and venv Python have run the exact
selection successfully. That evidence remains separate from the framework-reduced
matrix, Windows/GPU support, and final maintained-suite closeout.

## Context Bootstrap

Historically, test startup initialized JAX, Torch, and TensorFlow contexts for every run. That made even focused core tests pay framework import cost and could perturb concurrency tests.

Context bootstrap is now opt-in:

```bash
DRYML_TEST_BOOTSTRAP_CONTEXTS=1 pytest ...
```

`./tests.sh heavy`, `./tests.sh full`, and `./tests.sh profile` enable this only for the heavy phase. Smoke and medium runs leave global contexts uninitialized unless an individual test initializes what it needs.
