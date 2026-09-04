# DRYML Framework Repository

This active submodule owns DRYML implementation, tests, and durable product
documentation. The parent development workspace owns workspace policy and
records this repository's gitlink after coherent child work is complete.

## Documentation

Keep public classes, functions, methods, and types documented with responsibility,
behavior, parameters/types, return values, failure behavior, and relevant side
effects. Update `docs/` alongside changes to public APIs, persistent formats,
dispatch, concurrency, recovery, or user-visible behavior.

## Tests

Use `tests.sh` for DRYML verification. The normal maintained selection is:

```bash
./tests.sh --ignore tests/old --ignore tests/dev -x tests
```

Run focused files or directories by passing them through `tests.sh`. The old and
development tiers, `tests/old` and `tests/dev`, are excluded unless the user
explicitly requests them. `./tests.sh full` is the maintained full selection;
`./tests.sh profile --unknown-only` profiles unclassified test tiers.

## Source Ownership

`src/dryml` owns framework code. The descriptions below are target ownership
boundaries for the CDef V2 parity end state, not claims that every roadmap stage
is already implemented:

- `core` owns CDef, Object, ObjectRef, StateRef, Repo, Store, and query
  authority; `core/utils/graph` contains supported generic graph algorithms.
- `formats` owns dependency-light canonical encoding primitives.
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
