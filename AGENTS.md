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

`src/dryml` owns framework code:

- `core` provides core modules; `core/utils/graph` contains supported generic
  graph algorithms.
- `code` provides method instrumentation and analysis; `execute` is remote
  execution; `context` is compute context.
- `data`, `models`, `artifacts`, `vis`, and `devtools` own their named API areas.
- `ray`, `jax`, `tf`, and `torch` are framework-specific plugin areas.

There is no tracked `src/dryml/graph` package. Any untracked directory there is
unsupported user work and must not be inspected, edited, staged, deleted, or
used as a fixture without explicit user direction. Tracked `examples/` files are
DRYML examples; pre-existing untracked example files are user work, so stage and
test only exact paths approved for the task.

## Repository Architecture

Keep `core` independent of `dryml.code`, and keep `dryml.code` independent of
dispatch policy. Optional framework plugins must remain behind their plugin and
backend boundaries; do not introduce eager optional-framework imports into
lightweight package paths.
