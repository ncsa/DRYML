# DRYML Framework Repository

This active submodule owns DRYML implementation, tests, and durable product
documentation. The parent development workspace owns workspace policy and
records this repository's gitlink after coherent child work is complete.

## Documentation

Keep public classes, functions, methods, and types documented with responsibility,
behavior, parameters/types, return values, failure behavior, and relevant side
effects. Update `docs/` alongside changes to public APIs, persistent formats,
dispatch, concurrency, recovery, or user-visible behavior.

Durable design learnings and explicitly accepted limitations live in
`docs/solutions/`. Consult applicable entries before planning related work;
in particular, the code-shift note under `docs/solutions/architecture-patterns/`
records deferred long-lived-session, notebook, class-redefinition, and selector
questions without claiming hot-reload or live-Object migration support.

## Tests

Use `tests.sh` for DRYML verification. Start with focused files or directories,
then expand to the affected subsystem. Put explicit test paths before options
so the runner selects focused mode, for example:

```bash
./tests.sh tests/core/test_repo_save_load.py --no-cov -x
```

Use fail-fast for focused diagnosis, not for collecting related failures across
a bounded subsystem. A broad-suite failure returns work to focused/subsystem
verification under the global verification policy, not another full-suite run.

The old and development tiers, `tests/old` and `tests/dev`, are excluded unless
the user explicitly requests them.

### Verification Gates

Routine final verification proceeds from focused tests to the affected subsystem
and then the representative maintained suite:

```bash
./tests.sh good-enough --ignore tests/old --ignore tests/dev -x tests
```

The no-argument and option-first forms also select `good-enough`. This routine
gate runs smoke and medium tests without coverage, excludes package and heavy
tests, and deliberately uses the checked-in representative policy for selected
integration functions and parameter products.

Run the exhaustive suite **only when the user explicitly requests it**. Earlier
instrumented full runs took roughly 2.5 hours. Do not infer permission from a
request to implement, verify, review, commit, push, release, or fix CI. This rule
supersedes older plans and workflow instructions prescribing a full closeout.
Pass this restriction to delegated agents as well.

`./tests.sh exhaustive` (also `full`) retains every maintained combination;
`./tests.sh coverage` runs that exhaustive selection with instrumentation. Both
require an explicit user request, as do broad `medium`, `heavy`, and `profile`
runs that bypass the representative selection. Do not substitute `pytest tests`
or a broad focused-directory invocation to bypass this policy. Targeted tests
for an actual change or failure remain appropriate. Routine automated runners
and CI use `good-enough`; there is no scheduled exhaustive gate.

When adding code, add the smallest set of meaningful routine tests covering its
new behavior and important failure boundaries. Keep exhaustive parameter and
integration products out of the routine gate: mark separate matrix tests
`@pytest.mark.exhaustive_only`, or deliberately choose representative functions
and marginal matrix cases in `tests/test_profiles.json`. Keep all combinations
available to explicit exhaustive runs. Update curated-file allowlists when a
new regression belongs in `good-enough`; do not weaken assertions or use time
cutoffs to reach the feedback goal. Prefer controlled setup evidence where host
observation is incidental, but retain real observation tests at their boundaries.

## Source Ownership

`src/dryml` owns framework code. The descriptions below are target ownership
boundaries for the CDef V2 parity end state, not claims that every roadmap stage
is already implemented:

- `core` owns CDef, Object, ObjectRef, StateRef, Repo, Store, and query
  authority; `core/utils/graph` contains supported generic graph algorithms.
- `formats` owns dependency-light canonical encoding primitives.
- `locking` owns dependency-light native advisory-lock mechanics; Store and
  query-index consumers retain their paths and domain lifecycle policies.
- `annotations` is the passive typed-metadata attachment and deterministic
  collection kernel; metadata interpretation and policy stay with consumers.
- `methods` owns logical callable IR, implementation traits, implementation
  alternatives, direct selection, and reusable preparation.
- `code` owns generic code analysis; transformation is deferred. It has no
  DRYML product-package dependencies.
- `requirements` owns generic declaration, combination, report, and barrier
  protocols. `environments`, `worlds`, and `runtime` own their respective
  requirement semantics and enforcement.
- `records` provides general sidecar-record utilities rather than domain schemas
  or a second Object model.
- `managed` owns the lifecycle of operations that mutate stateful Objects,
  including interruption, checkpoint association, resume, and StateRef
  publication.
- `execute` owns execution-backend contracts and exact resolved-work transport;
  `dispatch` owns requirement coordination, candidate selection, and submission.
- `session` and `runtime` remain foundations that do not import dispatch policy
  or execution backends.
- `data`, `models`, and `artifacts` consume these foundations for user-facing
  workflows; `vis`, `metrics`, and `devtools` own their named API areas.
- `ray`, `jax`, `tf`, and `torch` are framework-specific plugin areas kept
  behind optional-backend boundaries.
- `operations` is a legacy package targeted for retirement; do not add new
  dependencies on it. Reassess legacy `context` only during parity closeout.

`dryml.execute` accepts trusted ordinary callable graphs only through an explicit
backend configuration. Its coordinator-owned spools, local worker groups, and
existing same-host Ray attachment are execution lifecycle boundaries, not Store
transport, Dispatch policy, environment provisioning, cluster provisioning, a
safe-deserialization boundary, or a cross-coordinator resource ledger. Keep Ray
optional and lazy; tests may prepare ephemeral CI fixtures, but product runtime
and ordinary integration tests use caller-supplied existing targets only.

There is no tracked `src/dryml/graph` package. Any untracked directory there is
unsupported user work and must not be inspected, edited, staged, deleted, or
used as a fixture without explicit user direction. Tracked `examples/` files are
DRYML examples; pre-existing untracked example files are user work, so stage and
test only exact paths approved for the task.

## Repository Architecture

Keep each public package independently useful and give it one coherent concept.
Lower layers do not import their consumers, domain packages own their persistent
schemas, and integration belongs in higher-level callers or narrow one-way
adapters. `dryml.core.symbol` may lazily call
`dryml.code.algorithms.lexical_dependencies` at call time for generic free-name
discovery only. No other core module may import `dryml.code`; `dryml.code` must
not import core, any DRYML product package, or dispatch policy. Keep annotations
passive. Dispatch coordinates selection while execute runs already resolved work.
Optional framework plugins must remain behind their plugin and backend
boundaries; do not introduce eager optional-framework imports into lightweight
package paths.
