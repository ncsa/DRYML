# Stage 8A Verification Record

Stage 8A implementation is locally verified through commit `48b1697` on
`v0.3-cdef-v2`. The required existing-target Ray qualification remains incomplete;
this record does not declare the plan's full Definition of Done satisfied.

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

No durable Store/managed schema, generic backend protocol, or setup v1.1 change
was introduced. The private invocation graph is v2 with no intermediate reader.
There is no automatic workload replay, coordinator managed report, whole-call
resource planner, or multi-host qualification.

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

## Remaining Qualification

Real-Ray execution is **not verified**. Ray 2.56.0 is installed and the 28 core
Execute/Dispatch Ray tests collect, but the required opt-in endpoint and existing
environment targets were absent:

- `DRYML_EXECUTE_INTEGRATION=1`
- `DRYML_TEST_RAY_ADDRESS`
- `DRYML_TEST_CONDA_PREFIX`
- `DRYML_TEST_VENV_PYTHON`

An absolute `CONDA_EXE` is also required and was available. No cluster or
replacement environment was provisioned. The 42 heavy-tier skips include these
28 tests and 14 existing generic Ray tests; skips are not qualification evidence.
Once caller-supplied same-host targets are configured, run:

```sh
./tests.sh tests/ray/test_core_execute_ray.py tests/ray/test_dispatch_ray.py
```

The additional lint check
`flake8 --select E9,F63,F7,F82 src/dryml tests --exclude tests/old,tests/dev`
reports existing undefined-name issues. All reported paths except
`core/object.py` are outside this changeset; its four annotation-name findings
also reproduce against the pre-Stage-8A baseline. Lint is not reported clean.
No project type-checker command is configured.

The accepted [long-lived code evolution limitation](solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md)
still applies. Resource caching is not hot reload or live-Object migration.
