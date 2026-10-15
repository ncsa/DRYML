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
The runner composes `PYTHONPATH` with the active Python's native path separator,
including when a Windows test invokes the runner through Git Bash again.

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

The Query/Method contract regressions from U1–U7 retain named routine coverage.
Curated query allowlists include main-only and empty-index completeness, malformed
authority, policy-preserving derived recovery, retained paging, and cumulative
failure/fallback accounting. Deterministic bounded thread barriers cover build-claim
waits outside fences, active same-process reader replacement, main writers, reversed
Store ordering, and cancellation cleanup. Existing SQLite concurrency cases remain
routine; the separate 18-case `test_u7_main_completeness_policy_matrix` is
`exhaustive_only` and remains available to an explicitly requested exhaustive run.

Tiny U1/U4 model and value-handoff regressions have exact medium node pins so the
heavy model categories cannot silently omit their named routine representatives.
The TF/Torch selected-call and accepted-prefix cleanup cases live in
`tests/models/test_dataset_training_integration.py`, which imports optional backends
only inside admitted test bodies. Keep backend-neutral generator factories at module
scope for stable import references, and native layer factories as explicit `F` specs;
do not pin routine cases inside the eager legacy `test_tf_training.py` or
`test_torch_training.py` modules. A fresh-process collection guard exercises the
actual file/tier/profile selection for these owners, checks exact retained node IDs,
and rejects heavyweight framework import attempts before test execution.
Four dense CPU directions cover every source/target backend; the full direction and
nested JAX products retain heavy qualification. These pins do not promote full
training matrices. Missing optional backends report skips. Collection-only checks
of explicit affected files with `--dryml-test-profile good-enough
--dryml-runner-tiers smoke,medium --collect-only` audit selection without running the
representative gate.

The ML workflow qualification representative set retains dependency-light coverage
for Method preparation/stream handoffs, recursive Experiment identity projection,
symbolic Definition quotation and history v1 fixtures, TrainState observation
persistence, Artifact/history association, closed qualification request
construction, and credential-free tiny TensorFlow/Torch/JAX Dataset-owned training
representatives.
The JAX representatives include exact managed-local and Core subprocess graph
restoration when the qualified JAX/Flax/Optax stack is installed. The routine
profile does not run TFDS preparation, the real 36-cell CPU matrix, existing Ray,
recovery interruption, or GPU execution. Those remain explicit opt-in gates.

The real TFDS NumPy adapter test is `exhaustive_only`. It requires a caller-selected
`DRYML_ML_QUALIFICATION_TFDS_DATA_DIR` and uses `download=False`; routine selection keeps
the mocked offline adapter tests while excluding that real reader.

The routine suite has an approximate five-minute feedback goal, not a runtime
cutoff. Before the Dataset/JAX integration representatives were added, a
reference selection measured **445.03 seconds (7m25s) wall time** on Linux with
Python 3.12.13 in `big_env`, with **3,167 passed and 22 skipped**:

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

`package` tests build an sdist and wheel beneath the system temporary directory's
`dryml/` child (normally `/tmp/dryml`; a caller's `TMPDIR`/`TEMP` override is honored), inspect their
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

## Native Estimator Qualification

The maintained native references and adapter contracts are
`tests/models/test_xgb_models.py` and `tests/models/test_statsmodels_models.py`.
They import native packages inside admitted fixtures/cases, never at collection,
and require Python 3.12 with the exact candidate versions: CPU XGBoost 3.4.2,
sklearn 1.9.0 for XGBoost, statsmodels 0.15.0, NumPy 2.1.3, SciPy 1.18.0,
and pandas 3.0.5. Missing packages or mismatched versions fail the requested
native cases; there is no `importorskip`. The statsmodels file does not require
sklearn. No installation or environment mutation is part of these tests.

For explicitly requested local native diagnosis, activate the existing `big_env` and place
all disposable artifacts below a caller-approved task directory. The following
assumes `$scratch` is an existing task directory (on the characterization host,
under `/data1/matthew/dryml-review-20261012`, overriding `/tmp/dryml`):

```bash
source ~/conda.sh
conda activate big_env
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
./tests.sh tests/models/test_xgb_models.py tests/models/test_statsmodels_models.py \
  --no-cov -x -q --basetemp="$scratch/pytest-native" \
  -o cache_dir="$scratch/pytest-cache" -o junit_family=legacy \
  --junitxml="$scratch/native.xml"
```

The explicit paths must precede options. JUnit `native_evidence` properties
record actual versions/platform, held-out scores/baselines, numerical errors,
native payload sizes, SHA-256 hashes, and save/load/save byte comparisons.
`tmp_path` retains the `.ubj` and full Results `.pkl` payloads under `--basetemp`;
the cache override prevents an in-checkout pytest cache. Reusing `--basetemp`
lets pytest replace that task-owned directory; choose a new subdirectory when
preserving earlier evidence. Neither focused mode nor these cases updates the
pre-existing timing JSONs. No global framework bootstrap is needed.

See
[Models native characterization](models.md#xgboost-and-statsmodels-native-characterization-u1)
for scientific and payload observations from the initial native-only references.
XGBoost reopen error was zero; complete
statsmodels Results remained semantically equal but native reserialization bytes
changed. Both maintained codecs now retain exact accepted payload bytes instead
of serializing restored native objects again.

Subsequent bounded Linux/Python 3.12 mixed-environment runs qualify adapter
quality/parity (`1e-6` XGBoost, `1e-8` statsmodels), pure selected/prepared calls,
authored short-batch exposure, candidate failure preservation, exact Repo replay,
associated terminal recovery and actual independent same-host Core workers. The
two native backend files now contain 26 XGBoost and 41 statsmodels cases. U5's
12 recovery cases passed in 340.80 seconds; 18 association/worker/Dir/Zip cases
passed in 565.91 seconds, including eight independent real workers. An earlier
combined run exceeded 720 seconds with an untraced failure marker; the complete
30-case set subsequently passed in two bounded batches. No root cause was
established for that earlier marker. These are local owning-unit results, not
hosted or isolated-installation qualification.

### Required hosted cells

`native-estimator-qualification` in `.github/workflows/tests.yaml` configures four
required support cells on every push, pull request and manual workflow dispatch:

| OS | Python | Isolated installation | Assigned cases |
|---|---|---|---:|
| Ubuntu | 3.12 | statsmodels only; no sklearn/XGBoost | 56 |
| Windows | 3.12 | statsmodels only; no sklearn/XGBoost | 56 |
| Ubuntu | 3.12 | CPU XGBoost plus sklearn; no statsmodels | 41 |
| Windows | 3.12 | CPU XGBoost plus sklearn; no statsmodels | 41 |

Each job uses an isolated strict conda-forge environment. XGBoost specifically
requires `py-xgboost=3.4.2=cpu_*`; statsmodels requires `statsmodels=0.15.0`.
Both use NumPy 2.1.3, SciPy 1.18.0 and pandas 3.0.5; only XGBoost adds
scikit-learn 1.9.0. Unavailable exact packages fail the environment solve rather
than substituting versions. DRYML's core `ml_dtypes>=0.6.0` requirement is
installed from PyPI with `--no-deps` inside that environment; conda-forge's older
`ml_dtypes` package cannot satisfy it. CI then builds DRYML and installs the wheel
with the requested extra and `--no-deps`, preserving the conda-installed CPU stack.
`pip check` and preflight validate the wheel's real extra metadata, required
versions, absence of unrelated optional model stacks and, for XGBoost, both the
Conda CPU selector/build string and `build_info()['USE_CUDA'] is False`.
Statsmodels imports OLS/Logit without sklearn/XGBoost. Python patch/build,
platform, architecture, resolved versions and Conda build facts are printed;
the codec compatibility gate itself uses Python major/minor, not patch.

Every cell executes four bounded **path-first** phases, never a broad heavy
profile or a global `-k` over backend files:

| Phase | Selection per backend | Count (XGBoost / statsmodels) |
|---|---|---:|
| native | Entire corresponding `test_xgb_models.py` or `test_statsmodels_models.py` | 26 / 41 |
| recovery | Two `test_native_*` functions in `test_experiment_recovery.py`, filtered to `xgb-regressor` or `stats-logit` | 6 / 6 |
| association | Terminal Artifact/callback function in `test_estimator_experiments.py` for both families, plus native Dir/Zip restore function in `test_state_ref_restore_into.py` | 4 / 4 |
| workers | Two independent-worker family cases plus missing/incompatible/workload-cleanup cases in `test_estimator_experiments.py` | 5 / 5 |

Collection preflight requires each phase's exact count and exact requested
node inventory. It then passes the collected node IDs directly to `./tests.sh`.
Executed call-phase JSON must match the collected inventory exactly, with every
outcome passed; JUnit must contain the required number of cases and zero skips,
failures or errors, including fixture/teardown failures. A successful pytest exit
alone is insufficient. `collected.json`, `executed.json` and `results.xml` are
uploaded per cell, including partial evidence on failure. Native thread limits
are set before imports through the job environment, with a 600-second per-test
watchdog and 60-minute job bound to allow real worker startup. These limits do
not alter product-global thread policy.

The native job enables the test-only `DRYML_TEST_INSTALLED_ESTIMATORS=1` mode.
Core workers receive only the checkout root on `PYTHONPATH` for `tests.*`
callables, never `checkout/src`. Before native fitting, coordinator and worker
verify `dryml`, `dryml.execute._worker`, the selected backend package and its
`base` module against the installed distribution's actual WHEEL/RECORD metadata:
resolved module paths must equal the recorded installation paths and module
bytes must match RECORD SHA-256/size entries. Editable installs and missing wheel
manifests fail qualification. The worker also checks its running `__main__`
bootstrap origin. Checks use resolved `Path` comparisons on Linux and Windows,
not a `site-packages` substring or a caller-provided trusted prefix. Worker
evidence is emitted before fitting; matching coordinator/worker paths and hashes
are retained in JUnit properties for successful and injected-failure cases.
Without this mode, ordinary local workers continue to use checkout source.

The separately requested hosted heavy/coverage jobs exclude these exact native
files and functions: they cannot qualify the native readers on their broader
Python/dependency matrices. The dedicated cells own their uninstrumented native
evidence; they do not add the full exact-pin matrix to routine or broad framework
CI. Explicit local path-first native runs remain available.

All four hosted cells remain **unrun/unqualified** until they pass on the exact
revision. A Linux mixed-environment run cannot qualify Windows or isolated
statsmodels-only/XGBoost-plus-sklearn installs. Configuration does not imply
branch-protection settings or remote execution. Cross-version readers, GPU,
external datasets and broader native platform combinations are not claimed.
Existing sklearn and additive codec-hook tests keep their owners and contracts;
the native rollout does not automatically close issue #38.

### Dependency-light routine coverage

`tests/models/test_estimator_contracts.py` is medium-tier and retains its complete
fast acceptance/config/spec/admission/missing-extra/control-boundary set.
`tests/data/test_materialize_supervised_numpy.py` inherits Data's medium tier;
all 47 strict collector cases remain routine without importing tensor/model
frameworks. Neither file needs a representative allowlist. The fresh-process
`test_routine_estimator_selection_keeps_fast_contracts_without_native_imports`
guard supplies only seven named owning files to the actual runner file-selection
policy, then exercises pytest's tier/profile collection with native imports
blocked. It proves these two fast files remain selected, the native backend and
worker/recovery files stay out, and the heavy native graph cases remain excluded
from the medium Core file. No default Data test is forced to run a native backend.
Installed-wheel package checks also resolve both backend manifests and construct
unfitted recipes with native imports blocked, and check extras on Python
3.10–3.14 for silent marker omission. Package tests remain outside routine runs.
The installed-artifact `test_installed_estimator_core_worker_origins` guard uses
the same Core executor and an actual worker to construct inert recipes with
native imports blocked, then compares both backends' root/bootstrap/backend
origins with the wheel manifest. It also rejects explicit `checkout/src`
overrides separately in coordinator and worker, recording failed-worker cleanup.
This package guard requires neither native extras nor Python 3.12; real native
fits remain in the dedicated qualification cells.

## Hosted Matrix

The ten-job lightweight matrix installs only package and test dependencies on
Ubuntu and Windows for Python 3.10 through 3.14. Ordinary pushes and pull
requests run `good-enough`. Only manually requested `exhaustive`
or `coverage` suites run `medium` there so every smoke/medium parameter
combination is exercised. Python 3.14 is explicitly framework-reduced.

The `tf` extra declares TensorFlow and tensorflow-datasets only for
`python_version < '3.14'`, matching that framework-reduced boundary. TFDS has no
declared minimum version: its deterministic local adapter contract is metadata
support, while installed tensorflow-datasets 4.9.10 is recorded only as the tested
current environment when its focused package checks pass. It is not evidence that
4.9.9 executed.

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

A separate Ubuntu Python 3.12 notebook job installs ipykernel 7.2.0 and
jupyter-client 8.8.0 and runs the real-kernel Dispatch regression on every push
and pull request. The general lightweight matrix does not install those optional
dependencies, so its skipped notebook case is not the notebook verification gate.
The test uses a job-owned kernel, temporary Store, and worker spool; it checks
repeated publication/query, shell liveness, task ownership, and interrupt recovery.

The heavy matrix runs on Ubuntu for Python 3.10 through 3.13 only when a user
manually dispatches `exhaustive` or `coverage`. It
installs and preflights TensorFlow, Torch, JAX/JAXlib, and pinned
`ray[default]==2.56.0` before heavy tests so missing or broken frameworks fail
rather than skip. Each job prints the resolved Python, DRYML, and framework
versions. Workflow configuration is not support evidence until the jobs pass on
the exact child commit.

The separate `JAX training qualification` matrix runs for pushes, pull requests,
and manually dispatched workflows on Python 3.12. It
installs `jax_training_test_requirements.txt` for two clean dependency sets:
the CPU JAX stack alone and the same stack with DRYML's existing `tf` and
`torch` extras. It pins JAX/JAXlib, Flax (including `flax.nnx`), and Optax,
requires their imports, and records a CPU backend with CPU devices before
running focused JAX import, first-class training, Dataset delivery, and managed
local/subprocess restoration tests. The mixed installation uses PyTorch
`2.14.1+cpu` from the official CPU wheel index, avoiding CUDA/Triton initialization
in this CPU qualification. Its preflight asserts a CPU-only Torch build and
constructs an optimizer so native import failures fail before the tests.
This file is qualification-only: DRYML's
existing lazy `jax` extra remains unchanged, and Flax/Optax are neither base
dependencies nor a new package extra. The experimental training API remains
responsible for selecting and reporting those dependencies.

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
