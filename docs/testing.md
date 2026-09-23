# Testing Workflow

DRYML tests are organized by feature category and by speed tier.

Maintained categories include `formats`, `annotations`, `environments`,
`worlds`, `runtime`, `session`, `core`, `locking`, `managed`, `package`, `data`, `execute`, `models`,
`ray`, `tf`, `torch`, `jax`, and `multi_framework`. `dispatch` is a maintained
medium category. Speed tiers are applied
automatically from `tests/test_tiers.json` by the DRYML pytest timing plugin.

## Runner Modes

The routine maintained gate is `good-enough`:

```bash
./tests.sh good-enough
```

Calling `./tests.sh` with no arguments, or with an option first, selects the
same mode. It runs smoke and medium tests selected by the checked-in
`tests/test_profiles.json` policy, excludes heavy tests and package files, and
does not enable coverage. The ordinary verification sequence is a focused test,
the affected subsystem, and then `good-enough`.

**Exhaustive runs require an explicit user request.** Earlier instrumented full
runs took roughly 2.5 hours. A generic request to implement, verify, review,
commit, push, release, or fix CI is not permission to launch one. This policy
supersedes full-closeout requirements in older plans and applies to delegated
agents too. Do not bypass it with broad direct pytest runs, `medium`, `heavy`,
`coverage`, or `profile`. Routine automated runners use `good-enough`; there is
no scheduled exhaustive job. A user selecting `exhaustive` or `coverage` in
GitHub's manual workflow dispatch is an explicit request.

Named suite modes require Bash 4 or newer. On macOS, use a modern Bash with
`bash ./tests.sh good-enough`; the system `/bin/bash` is version 3.2. Hosted
macOS CI installs Bash through Homebrew and invokes it explicitly.

The complete runner modes are:

- `./tests.sh` and `./tests.sh good-enough`: representative smoke and medium
  maintained tests, excluding package and heavy categories, without coverage.
- `./tests.sh smoke`: all smoke-tier parameter combinations without coverage.
- `./tests.sh medium`: all smoke and medium parameter combinations without
  coverage.
- `./tests.sh heavy`: all heavy-tier parameter combinations without coverage.
- `./tests.sh exhaustive`: every maintained smoke, medium, and heavy test in
  three phases without coverage. `full` is an alias.
- `./tests.sh coverage`: the same three maintained phases with `pytest-cov` and
  `--cov-append` after the first phase.
- `./tests.sh package`: all maintained package-category tests only, without
  coverage unless the caller explicitly supplies `--cov`.
- `./tests.sh profile`: every maintained test in three phases without coverage,
  followed by timing metadata updates. `--unknown-only` limits execution to
  node IDs missing from the node-tier timing baseline.
- `./tests.sh tests/path.py ...`: the path-first focused mode. It adds
  `--no-cov` by default but honors an explicit `--cov` or `--cov=...` option.

Named modes and option-first `good-enough` runs accept pytest options but reject
narrow test paths and node IDs. A lone trailing `tests` or `./tests` suite root
is accepted and removed before the runner supplies its selected files. Put
focused selections first instead.

## Daily Commands

Run the routine maintained selection:

```bash
./tests.sh good-enough
```

Restrict routine verification to an affected category when diagnosing changes:

```bash
./tests.sh good-enough -m category_core
```

Explicit paths or node IDs still support narrow reproductions and tests of new
code. Do not expand a focused invocation into an unrequested exhaustive gate.
The legacy `medium` suite includes `smoke` and all of their parameter products;
it is not the routine default.

Only when the user explicitly requests every maintained parameter combination:

```bash
./tests.sh exhaustive --ignore tests/old --ignore tests/dev -x tests
```

Only when the user explicitly requests exhaustive combined coverage:

```bash
./tests.sh coverage --ignore tests/old --ignore tests/dev -x tests
```

Exhaustive, coverage, and profiling runs execute process-sensitive
session/runtime/orchestrator/dispatch tests in a fresh phase, other
smoke/medium files in a second phase, and heavy files last. This keeps
intentional late-import and terminal-publication tests isolated. Only coverage
mode combines phase data through `pytest-cov` append mode.

For a focused run, put test paths or node IDs first. Coverage is off by default:

```bash
./tests.sh tests/core/test_repo_save_load.py -x
```

Named `smoke`, `medium`, `heavy`, `good-enough`, `exhaustive`, `full`,
`coverage`, `package`, and `profile` suites accept pytest options but reject
explicit test selections; put focused paths first instead.
Options such as `-k` and `-m` further restrict each named suite and cannot
broaden its speed-tier selection.

## Representative Policy

`good-enough` is a deterministic representative profile, not a complete or
random sample. Add just enough routine tests to cover new code's meaningful
behavior and important failure boundaries, rather than every permutation.
Keep complete parameter and integration products in the explicit exhaustive
selection. `tests/test_profiles.json` names integration functions whose
other functions may be deferred, optionally enables marginal selection for
specific multi-axis parameter products, and pins concurrency, fault-window,
cancellation, locking, and other safety-sensitive paths to exhaustive
selection. There is no random sampling or runtime cutoff.

For a separate exhaustive matrix, use `@pytest.mark.exhaustive_only` on the
function or `pytestmark = pytest.mark.exhaustive_only` on a matrix-only module.
The representative profile deselects these tests; explicit exhaustive modes
retain them. The marker takes precedence over file-level representative pins,
and marking a mandatory `must_run` node raises a collection error. Keep a small
unmarked regression alongside a matrix when its behavior belongs in the routine
gate. For a matrix whose representative rows should remain routine, use an
explicit `sampled_functions` entry instead of marking the whole matrix.

Marginal selection retains enough rows to observe every value on every
parameter axis, plus any explicitly pinned rows. It is not pairwise testing and
does not promise every interaction between axis values. `must_run` node pins
also survive a file's representative-function allowlist. All selection remains
within the runner's chosen speed tiers and can be narrowed by user filters.
Files not named by the
representative-function policy remain complete within the selected speed tiers.
New smoke or medium files are included by default, but a new function added to
a file with a curated `representative_functions` entry is omitted until that
file's function allowlist is deliberately updated. This intentional selection
gap is why the exhaustive mode remains available on explicit user request.

The routine suite has an approximate five-minute feedback goal, not a runtime
cutoff. A reference selection measured **445.03 seconds (7m25s) wall time** on
Linux with Python 3.12.13 in `big_env`, with **3,167 passed and 22 skipped**:

```bash
./tests.sh good-enough -q --durations=30 --dryml-timing-summary
```

The process-sensitive phase took 106.90 seconds and the ordinary phase took
332.53 seconds; wall time also includes runner startup. The five-minute goal is
not yet met. This measurement supersedes the older 38-minute representative
run, which used a different selection and test setup. It is not a guarantee for
other hosts: actual environment observation and worker tests remain sensitive
to the installed environment. Re-measure after changing the policy or setup.

A one-off instrumented run of that shortened selection passed the same tests
in **12m45s**. On the same 48,708 tracked-source statements, coverage was
**78.49%**, compared with **81.62%** from saved full-run data: a 3.13 percentage
point decrease, retaining 96.15% of previously covered statements. About 80% of
the lost coverage was in model implementations, optional-framework adapters,
and legacy context code. This is line coverage, not proof of identical branch
or parameter-interaction coverage. Routine runs remain uninstrumented.

## Policy Test Evidence

Policy-only tests can request the explicit `synthetic_environment_record`
fixture instead of scanning installed distributions. It supplies a small
controlled package inventory with portable interpreter paths and rejects real
inventory access for the duration of that test, including attempts caught by
the code under test. It is not an autouse fixture: current-environment
inspection, actual worker probes, and integration tests retain real observation
where observation itself is the contract being verified.

Persistence-focused tests may opt into `fixed_snapshot_environment`. It
substitutes fixed evidence only at the snapshot-capture seam when the caller
omits an observer; explicit observers and clocks remain intact. Managed
persistence tests may instead use `fixed_managed_snapshot_environment`, which
also substitutes the managed pre-observation seam while leaving requirement
collection and lifecycle behavior on production paths. These fixtures are not
autouse and are for persistence-focused tests only. In mixed local/worker files,
only the explicitly selected local cases use them. Tests of real environment
inspection, worker admission, and worker provenance retain their actual
observation paths.

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

Only when the user explicitly requests a broad profiling pass:

```bash
./tests.sh profile --ignore tests/old --ignore tests/dev -x tests
```

This runs process-sensitive, other smoke/medium, and heavy phases. It writes
`test-timings-process-state.json`, `test-timings-medium.json`, and
`test-timings-heavy.json` beneath `/tmp/dryml/profile`, prints timing summaries,
and merges all three timing files into `tests/test_tiers.json` node-tier
overrides for tests that passed. The timing files are disposable artifacts, not
repository outputs.

When the user explicitly requests profiling only node-tier-unclassified tests:

```bash
./tests.sh profile --unknown-only
```

Unknown means a collected test nodeid is absent from `tests/test_tiers.json`
`node_tiers`. It does not mean recently added: the filter excludes only IDs
known specifically to `node_tiers`. Path tiers, category tiers, and default
tiers still decide which profile phase collects the test, but they do not make
a node known for this filter. Only missing nodeids are executed and written to
the timing output. This can still select expensive tests and is not an automatic
step after adding tests. Use focused timings or the routine profile's
`--durations` output for ordinary development.

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
5. Add minimal representative checks for new behavior and important failures to
   `good-enough`. Mark separate exhaustive matrix functions/modules
   `exhaustive_only`, or curate representative functions/rows in the profile.
6. If a new function is in a file curated by `representative_functions`, update
   its allowlist when it belongs in routine coverage. Do not remove a needed
   regression or weaken assertions solely because it is slow.
7. Run focused checks and `./tests.sh good-enough`. Run broad exhaustive,
   framework, coverage, or profiling modes only on explicit user request, not as
   automatic closeout, shipping, or CI-repair work.
8. Keep only classifications for the tests added or changed by the current work;
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

The ten-job lightweight matrix installs only package and test dependencies on
Ubuntu and Windows for Python 3.10 through 3.14. Ordinary pushes and pull
requests run `good-enough`. Only manually requested `exhaustive`
or `coverage` suites run `medium` there so every smoke/medium parameter
combination is exercised. Python 3.14 is explicitly framework-reduced.

Lightweight CI jobs print individual test names and use `pytest-timeout`'s
thread watchdog with a 180-second per-test limit (including fixture work).
A stalled test dumps thread stacks and fails the process instead of hanging
indefinitely; the watchdog does not turn unfinished tests into passes or define
profile membership. Ubuntu jobs have a 20-minute job limit and Windows jobs a
40-minute limit. Local commands have no per-test timeout unless the caller
supplies one explicitly, and no selection-level performance cutoff has been
introduced.

An additional macOS Python 3.12 job also runs `good-enough`, including native
filesystem publication tests. Ubuntu and Windows native publication checks are
already included in their routine jobs, avoiding duplicate runners. Package
tests run only for manually requested exhaustive/coverage verification, through
`medium` on Ubuntu/Windows and a package-only step on macOS.

The heavy matrix runs on Ubuntu for Python 3.10 through 3.13 only when a user
manually dispatches `exhaustive` or `coverage`. It
installs and preflights TensorFlow, Torch, JAX/JAXlib, and pinned
`ray[default]==2.56.0` before heavy tests so missing or broken frameworks fail
rather than skip. Each job prints the resolved Python, DRYML, and framework
versions. Workflow configuration is not support evidence until the jobs pass on
the exact child commit.

The separately gated `Existing Ray integration (Ubuntu, Python 3.12)` job uses an
ephemeral Conda coordinator and a venv derived from that same interpreter. It
builds DRYML once, installs that artifact with `dill` and
`ray[default]==2.56.0` into both existing test targets, starts a job-owned
single-node 4-CPU/0-GPU Ray fixture, and enables the real Ray and
existing-environment Execute tests. The enabled selection also covers Dispatch
local-probe/Ray-workload, Ray-probe/local-workload, Ray-probe/in-process,
same-config independent ownership, and exact Conda/venv pin identity. Its final
cleanup stops only the fixture it started. This CI preparation does not change the product contract: normal DRYML
runtime and tests require caller-supplied existing environments and an existing
Ray address, and never provision them. Like the heavy matrix, this job runs only
for a manually requested `exhaustive` or `coverage` suite. There are no scheduled
exhaustive, framework, or Ray runs.

An Ubuntu Python 3.12 coverage job runs only when manual dispatch explicitly
selects `coverage`. It installs test and heavy dependencies and runs
`./tests.sh coverage`; ordinary and manual `good-enough` or
`exhaustive` events do not run that job.

The lightweight Ubuntu/Windows Python 3.10 through 3.14 matrix remains
framework-reduced, does not install Ray, and supplies common and subprocess
coverage. Package tests require an explicitly requested extended run. This is
not native Windows Ray or GPU evidence. Workflow configuration
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

`./tests.sh heavy`, `./tests.sh exhaustive`/`full`, `./tests.sh coverage`, and
`./tests.sh profile` enable this only for the heavy phase. Smoke, medium, and
good-enough runs leave global contexts uninitialized unless an individual test
initializes what it needs.
