# Graph Querying

`Repo.query()` inspects CDef and reference authority without materializing objects or importing optional backends. Structural queries operate on CDef class and semantic parameters. `Repo.references()` returns ObjectRef and StateRef result sets for lineage, namespace, primary-path, alias, and exact-state questions.

Graph traversal records typed V2 `Parameter` and container paths. Materializing edges participate in owned object topology; `Ref` edges remain inspectable but are not constructed or saved as dependencies. Final query verification compares authoritative CDef/ObjectRef/StateRef values, not a CDef projection of an exact reference.

Store indexes are acceleration only. Rebuild scans authoritative definition and reference records, announces visible progress, and can safely replace a missing or stale derived index. A query may fail closed when current metadata is incompatible; it never treats an incompatible index as empty or current authority.

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
