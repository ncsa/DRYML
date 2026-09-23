# Graph Querying

`Repo.query()` inspects CDef and reference authority without materializing objects or importing optional backends. Structural queries operate on CDef class and semantic parameters. `Repo.references()` returns ObjectRef and StateRef result sets for lineage, namespace, primary-path, alias, and exact-state questions.

Graph traversal records typed V2 `Parameter` and container paths. Materializing edges participate in owned object topology; `Ref` edges remain inspectable but are not constructed or saved as dependencies. Final query verification compares authoritative CDef/ObjectRef/StateRef values, not a CDef projection of an exact reference.

Store indexes are acceleration only. Rebuild scans authoritative definition and reference records, announces visible progress, and can safely replace a missing or stale derived index. A query may fail closed when current metadata is incompatible; it never treats an incompatible index as empty or current authority.

## Categorical Projection

`DefinitionQuery.categorical(*, path="$", recursive=False, drop=(),
drop_args=False, drop_class=False)` returns a new query with a selected occurrence
projected into named semantic constraints. `path` selects the root or subtree;
surrounding constraints and aliases outside that occurrence are unchanged.
`drop` names constructor parameters, `drop_args=True` removes all argument
constraints, and `drop_class=True` removes the class constraint. Controls combine
as a union and named drops are validated against the original selected traversal,
so a missing name fails even when `drop_args=True` is also supplied.

The query keeps its original occurrence authority for `exact()` and `restore()`.
Projection does not load Objects, rewrite CDefs, Store records, references, or
payloads. It prepares CDef parameter matching without class imports. An authored
Definition may require signature resolution to name positional constraints;
equal class symbols, strict matching, and classless selectors avoid class
resolution, while the default policy can resolve unequal retained class symbols
to perform its existing inheritance check.

```python
from dryml.core import field

query = (
    repo.query(pipeline_cdef)
    .categorical(drop=("seed",), recursive=True)
    .categorical(path="optimizer", drop=("learning_rate",))
    .exact(path="model")
)

# Reference filtering follows structural composition and returns StateRefs.
matching_states = (
    query.references()
    .where(field("object", "project").eq("forecasting"))
    .state_refs()
)

# Restore the original complete optimizer occurrence in another immutable query.
original_optimizer_query = query.restore(path="optimizer")
```

`exact(path=...)` reinstates the original exact CDef occurrence when no explicit
definition is supplied; an explicit replacement must be a CDef or Object.
`restore(path=...)` reinstates the original selected occurrence, including fields
removed by categorical projection. Both operations return new queries and raise
`QueryPathError` for an unconstrained query or invalid path. `where(...)` returns
a reference query, so apply later structural transformations before it.

## Metadata Predicates

`field(scope, *path)` builds an inert typed selector for `object`, `state`,
`lineage`, or `snapshot` metadata. Use it only on `ReferenceQuery.where()`;
repeated calls intersect with existing structural, reference, and metadata
filters.

```python
from dryml.core import Definition, field

states = (
    repo.references()
    .definition(Definition(Model))
    .where(field("object", "project").eq("forecasting"))
    .where(field("snapshot", "saved_at").ge(cutoff_utc))
    .state_refs()
)
```

Predicates support `exists`, `missing`, typed `eq`, literal `contains`, and typed
numeric/UTC timestamp ordering, composed with `&`, `|`, and `~`. Present null
exists; absent and present-empty mappings are distinct. Invalid scopes, paths,
operand types, non-finite numbers, naive datetimes, and expression bounds fail
explicitly. State and snapshot predicates retain exact StateRef candidates;
`object_refs()` then projects the distinct ObjectRefs of the matching states.

Terminals verify authoritative metadata after candidate selection. Malformed
records, multi-Store conflicts, unsupported evaluations, and index failures remain
errors rather than non-matches. Inspection and query never open local payloads,
materialize Objects, run environment observation, or imply execution admission.
