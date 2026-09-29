# Stage 8A Verification Record

Stage 8A implementation was locally verified through commit `48b1697` on
`v0.3-cdef-v2`. The required same-host Ray qualification has also passed after
explicitly authorized preparation of local test targets. The execution gates
recorded below are complete; baseline lint and the existing code-evolution
limitations remain unchanged.

## Implemented Scope

- Session-owned opt-in resource caching with inspectable membership, owner-bound
  leases, semantic Repo matching, physical Store matching, and noncommitting
  dependency-ordered cleanup.
- Standalone Store definitions using the existing descriptor grammar and
  existing-authority reconstruction.
- Worker setup integration, whole-call ManagedConfig transport, and common
  invocation-time state/control defaults for direct and nested managed calls.
- Six passive-annotation/function/managed decorator orders and validated ordinary
  wrapper chains, including hidden managed controls and one-shot normalization.
- Local/subprocess and Dispatch conformance tests, opt-in Ray matrix tests,
  package/import tests, and synchronized public guides.

Stage 8A introduced no durable Store/managed schema, generic backend protocol, or
setup v1.1 change. A subsequent CI-driven correction added a bounded
pre-invocation deadline marker to the generic worker error terminal and therefore
bumped that private same-version worker protocol to v4 without a compatibility
reader. Store and managed schemas, setup v1.1, and the private invocation graph v2
remain unchanged. There is no automatic workload replay, coordinator managed
report, whole-call resource planner, or multi-host qualification.

## Maintained Verification

Toolchain: Python 3.12.13 in `big_env` on the development host. Tests used
`tests.sh`; disposable pytest and coverage artifacts were directed beneath
`/tmp/dryml`. Old and development tiers remained excluded.

The required command was started as:

```sh
./tests.sh --ignore tests/old --ignore tests/dev -x tests
```

The process-sensitive partition passed. The ordinary partition exposed a test
isolation bug: an intentionally unresolved worker-cleanup test retained its
executor quota in the shared test interpreter. The regression was reproduced
with the affected ordered test pair, then the entire failure-injection executor
lifetime was isolated in a bounded child interpreter. Product cleanup semantics
were not weakened and no spool-budget reset was added.

After that test-only correction, the ordinary smoke/medium selection was rerun
using the same `tests/tools/test_buckets.py` selection and process-state exclusion
rules as `tests.sh`. The heavy partition then ran through `./tests.sh heavy`.
The already-passing process-sensitive partition was not repeated.

| Partition | Passed | Skipped |
| --- | ---: | ---: |
| Process-sensitive smoke/medium | 308 | 0 |
| Ordinary smoke/medium | 2844 | 21 |
| Heavy | 87 | 42 |
| Total | 3239 | 63 |

Focused gates also exercised resource acquisition failures, stale keys, copied
contexts, teardown retry, explicit managed configuration, wrapper snapshots,
callback shapes, malformed graph nodes, exact StateRef receipt retention,
environment/world declaration conflicts, and real subprocess cleanup evidence.
The package/import and public documentation gates passed in the maintained
selection. Expected static-coverage, fork, and optional-framework warnings were
reported rather than suppressed.

## Review Disposition

The structured review used correctness, reliability, API, testing, standards,
maintainability, performance, adversarial, and applicable-learning lenses. The
independent validator confirmed six actionable findings; all were fixed and
focused-tested:

- Explicit root managed configuration is consumed once by the worker seam.
- ManagedConfig values are rejected in result graphs before opening resources.
- World requirements attached to managed declaration carriers remain visible.
- Unresolved worker cleanup retains an owner and reports setup-exit failure;
  subsequent setup in that process is rejected until exit/restart.
- Callback `None` and empty-list shapes remain distinct across transport.
- A managed declaration rejects rebinding to unrelated owners.

The proposed relocation of lazy core execution integration was rejected by the
validator because those seams are explicitly part of the Stage 8A design.
Cross-model corroboration was unavailable under the host-attestation contract;
the local adversarial fallback ran. No hostile-deserialization or sandbox claim
was evaluated, and the separately deferred sensitive-data sentinel audit was not
performed.

## Ray Qualification

Following explicit user authorization, a separate local Ray 2.56.0 cluster was
started from `big_env` with four logical CPUs, no GPUs, and a 256 MiB object store.
A disposable venv was created with `python -m venv --system-site-packages` beneath
`/tmp/dryml/stage8a-ray-qualification/venv`. It has its own interpreter prefix and
uses the already installed Ray, dill, and current DRYML checkout from `big_env`;
no system-wide installation or changes to `big_env` were needed.

The gate received `DRYML_EXECUTE_INTEGRATION=1`, the new cluster's concrete
host/port through `DRYML_TEST_RAY_ADDRESS`, the active `big_env` prefix through
`DRYML_TEST_CONDA_PREFIX`, the disposable venv interpreter through
`DRYML_TEST_VENV_PYTHON`, and the active absolute `CONDA_EXE`.

```sh
./tests.sh tests/ray/test_core_execute_ray.py tests/ray/test_dispatch_ray.py
```

The run used `--no-cov -x -ra` and a JUnit report beneath the disposable task root.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Core Execute Ray | 12 | 0 |
| Dispatch Ray | 16 | 0 |
| Total | 28 | 0 |

This includes exact conda and venv execution pins, all six decorator orders,
ordinary wrapper composition, nested explicit managed configuration, shared-Store
result recovery, cancellation, independent probe placement, and unavailable-probe
failure without fallback. One expected `DispatchCoverageWarning` reports incomplete
static analysis of the ordinary function that invokes a nested managed call.

The real run exposed three test-fixture problems, not runtime changes:

- The core fixture now creates its required spool directory before submission.
- Wrapper assertions distinguish the inner managed final snapshot from later
  outer-wrapper effects. A worker-written marker separately proves `finally`
  executed, including the effect intentionally absent from that snapshot.
- The Dispatch matrix pins the active interpreter explicitly because its backend
  configuration disables automatic environment discovery while its declarations
  require Python compatibility evidence. It also asserts explain eligibility.

After these test-only changes, 94 local/subprocess composition and Dispatch tests
passed. The maintained suite was not repeated; its earlier results remain above.
The 28 previously skipped core/Dispatch Ray cases are now qualified by execution,
not collection. The other 14 generic Ray cases from that earlier heavy selection
were not rerun as part of this Stage 8A gate.

The qualification cluster was shut down after each run. The leftover Stage 7 Ray
cluster was also stopped at the user's request; no Ray native processes remained.
The disposable venv and qualification artifacts remain beneath `/tmp/dryml`.

## CI Corrections

The first cross-platform run exposed test portability differences: Python
3.10/3.11 wrap descriptor-binding errors during class creation, and Windows
Python 3.13+ rejects rooted paths without a drive as absolute paths. The tests
now inspect the original declaration error and use platform-absolute fixtures;
all 49 affected tests passed locally on Python 3.10, 3.11, and 3.12.

A subsequent Windows run exposed a pre-invocation deadline race. Worker-owned
expiry now carries explicit bounded protocol evidence rather than the same
type-only payload as a callable's own `TimeoutError`. Both backends preserve
ordinary error identity and require lifecycle-confirmed termination for deadline
outcomes. Private protocol v4 rejects older workers before workload admission,
and a claimed ordinary outcome cannot be overtaken by cancellation during result
decoding. The correction passed 162 focused protocol, worker, discovery,
admission, cancellation, and documentation tests, followed by another complete
28-test real-Ray qualification with no skips.

## Remaining Limitations

The additional lint check
`flake8 --select E9,F63,F7,F82 src/dryml tests --exclude tests/old,tests/dev`
reports existing undefined-name issues. All reported paths except
`core/object.py` are outside this changeset; its four annotation-name findings
also reproduce against the pre-Stage-8A baseline. Lint is not reported clean.
No project type-checker command is configured.

The accepted [long-lived code evolution limitation](solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md)
still applies. Resource caching is not hot reload or live-Object migration.
