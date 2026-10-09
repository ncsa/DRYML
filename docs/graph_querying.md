# Graph Querying

Query V3 has one composable identity universe. `Repo.query()` captures identity
knowledge from connected Stores and retained caches; `Store.query()` captures one
Store's authority. Construction is inert. Restrictions and terminals inspect
definitions, complete ObjectRefs, StateRefs, and captured metadata without
materializing Objects, reading payloads, allocating ObjectIds, saving, observing
the host, or running workloads.

`Repo.query(weak=True)` includes both weakly and strongly retained cache facts;
pass `weak=False` when only strong cache retention may contribute. This producer
choice does not affect connected Store knowledge. `Repo.get()` and item lookup
remain non-query live-Object operations and return `ObjectResultSet` where their
existing API returns multiple retained Objects.

```python
from dryml.core import Definition

models = repo.query().sel(Definition(Model)).cdefs().stored().collect()
model = repo.load_object(models.one())
```

`sel()` composes structural Definitions, complete CDefs, `Selector` values,
Generator selectors, and exact references. Complete Definitions remain structural;
complete CDefs and exact references retain their complete graph identity. Query
results are `IdentitySet` values with deterministic identity ordering. `collect()`
fixes membership before iteration, while `take(n)` is the explicit bounded prefix
operation and retains visible boundedness on its returned set.

`ObjectRef.object_projection()` and `StateRef.object_projection()` produce an
`ObjectSelector` for one complete realized graph. Projection preserves ObjectIds,
CDef topology, and edge roles while replacing every encountered exact reference
with a nested `ObjectSelector`; StateRef hashes do not participate. Raw ObjectRef
or StateRef values that a caller explicitly embeds in an ObjectSelector remain
exact pins. ObjectSelectors are query-only, but their `reference` property exposes
the corresponding recursively state-weakened ObjectRef when a persistence API
requires object-association authority.

Use `query.object_projection()` or `identity_set.object_projection()` to collect
and deduplicate reference members as a fixed `ObjectSelectorSet`. CDef members are
omitted because they have no realized ObjectIds. This makes multiple checkpoints
of one realized composite graph count once without merging separately constructed
graphs:

```python
runs = repo.query().state_refs().object_projection()
run_count = runs.count()
```

`categorical()`, `exact()`, and `restore()` edit the latest selector without
discarding prior query restrictions. During categorical projection,
named drops are validated against the original
selected traversal before projection; a projected name cannot hide an invalid or
absent source path.

For a stored-CDef-only query whose Stores have ready SQLite indexes, `take(n)`
uses graph-complete `(graph digest, encoded root)` keyset pages and a bounded
cross-Store frontier. The encoded tie key preserves sharing topology and genuine
digest collisions. Query V3 verifies returned CDefs against direct Store authority,
detects unmanaged stored-root sidecar mutation, and otherwise falls back to complete
authority unless `scan_policy("forbid")` or `require_indexed()` requires a visible
failure. `refresh(False)` uses this path only when the sidecar is already ready; it
never initializes or rebuilds derived state.

## Identity Restrictions

Use `cdefs()`, `object_refs()`, and `state_refs()` as lazy kind restrictions.
`stored(scope=...)` restricts the existing universe to kind-specific authoritative
membership; it never discovers extra identities. `cached(scope=..., weak=...)`
similarly restricts an existing universe to retained cache knowledge. `in_source()`
limits contribution evidence and binds a later default authority source; use an
explicit `scope=` for metadata, aliases, or membership authority when a fixed or
multi-producer expression has no unambiguous default.

Reference identity restrictions compose in the same plan: `object_id()`,
`namespace()`, `contains()`, `state_hash()`, `alias()`, and `where(field(...))`.
Metadata scopes are `object`, `state`, `lineage`, and `snapshot`; terminals retain
authority validation and report conflicts or malformed records rather than treating
them as non-matches.

## Relationships

Relationships begin with an explicitly selected root universe. `roots.nested()`
returns strict non-empty typed occurrences. Its default `EdgePolicy.ALL` includes
associations, materializing edges, and retained references. Pass `EdgePolicy.OWNED`
when owning-only traversal is intended. `through(RelationshipKind.REFERENCE)`
qualifies actual paths containing a reference hop. `owners()` and `targets()` return
new identity queries, so their result kind is selected with a normal kind
restriction.

```python
from dryml.core.query import EdgePolicy, RelationshipKind

owners = (
    repo.query().cdefs().stored()
    .nested(target, edges=EdgePolicy.ALL)
    .through(RelationshipKind.REFERENCE)
    .owners().cdefs().collect()
)
```

Raw occurrence terminals retain root, typed path, target, and contributing-source
evidence. `max_occurrences(n)` bounds only raw occurrence output; direct owner and
target projections remain existential. `max_depth()`, `max_verify()`,
`max_witnesses()`, `scan_policy()`, `require_indexed()`, and `explain()` preserve
visible execution bounds and diagnostics across composition. `explain(analyze=True)`
executes the plan; ordinary explanations report planned source cuts and residual
work without evaluation.

`scan_policy("allow")` permits required authoritative inventory scans,
`scan_policy("warn")` additionally emits one warning per scanning terminal, and
`scan_policy("forbid")` rejects the terminal before scanning. `refresh(False)`
does not reconcile derived indexes, `refresh("auto")` reconciles stale indexes
when safe, and `refresh(True)` forces reconciliation before each terminal cut.
Refresh may replace derived index sidecars but never rewrites Store authority.

Trusted selector resolution may import ordinary Python code when its existing
semantics require it. That accepted import side effect does not permit query
operators to construct Objects or execute workloads.
