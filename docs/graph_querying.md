# Graph Querying

`Repo.query()` inspects CDef and reference authority without materializing objects or importing optional backends. Structural queries operate on CDef class and semantic parameters. `Repo.references()` returns ObjectRef and StateRef result sets for lineage, namespace, primary-path, alias, and exact-state questions.

Graph traversal records typed V2 `Parameter` and container paths. Materializing edges participate in owned object topology; `Ref` edges remain inspectable but are not constructed or saved as dependencies. Final query verification compares authoritative CDef/ObjectRef/StateRef values, not a CDef projection of an exact reference.

Store indexes are acceleration only. Rebuild scans authoritative definition and reference records, announces visible progress, and can safely replace a missing or stale derived index. A query may fail closed when current metadata is incompatible; it never treats an incompatible index as empty or current authority.

## Reference-Aware Containment

Nested containment finds authoritative stored CDef roots with a non-empty typed
path to a target. The CDef entry is
`repo.query(target).nested(...)`; `target` may be a concrete CDef, an exact
`ObjectRef`, or an exact `StateRef`. The reference entry is
`repo.references().containing(target, ...)`. It is an adapter for the same
immutable nested query, so an unfiltered reference builder produces the same
owners, occurrences, policies, source scope, and explanation as the CDef entry.
It rejects already-filtered reference builders rather than reinterpreting an
authority predicate as containment. Ordinary `Repo.references()` lookup,
including `contains`, `exact`, aliases, and metadata predicates, is unchanged.

`edges="materialize"` is the compatibility default and permits only owning
materializing hops. `edges="ref"` permits only retained `Ref` hops, and
`edges="all"` permits either kind at every hop. `contains_ref=False` is the
default. Set `contains_ref=True` to retain only paths that contain at least one
selected `Ref` hop; it narrows the selected edge policy and never widens it.
Therefore `edges="materialize", contains_ref=True` is deterministically empty,
while `edges="ref", contains_ref=True` has the same non-empty path membership
as `edges="ref"`. An independently stored target does not match through its own
empty root path; it qualifies only through another stored root's non-empty path.

```python
from dryml.artifacts import CachedDataset
from dryml.core import Definition

# This is definition-only: construction and the query do not open the Dataset
# or compute the cache. `source` need not be an independently stored root.
source = Definition(MyDataset, "training").concretize()
cached = CachedDataset(source, repo=repo)
repo.save_object(cached)

owners = (
    repo.query(source)
    .nested(edges="ref")
    .owners()
    .defs()
)
assert list(owners) == [cached.definition]
```

For an exact reference target, `object_refs()` and `state_refs()` return only
their respective complete target identities. A StateRef comparison includes its
complete ObjectRef and state identity; an ObjectRef comparison includes its
complete object identity. Neither terminal coerces an exact reference into its
CDef, so a different checkpoint of the same object, or a different object with
the same CDef, does not match. `owners().defs()` returns the enclosing stored
CDefs for any target kind. Exact references are terminal values: containment
does not load Objects, read payloads, resolve target authority, or infer that a
target is independently stored, loadable, restorable, or eligible for cleanup.

Raw nested execution returns one occurrence per qualifying owner-to-target
path. Each occurrence retains the target plus `hops`, whose ordered entries pair
the direct typed path segment with its literal `materialize` or `ref` kind. This
distinguishes mixed paths such as `ref -> materialize -> ref` without treating a
reference on an unrelated branch as evidence for a materializing path.

Use `in_store(store)` after `nested()` or on the unfiltered reference builder
before `containing()` to restrict roots to that exact connected Store handle.
Source restriction happens before replica merging, counts, and occurrence
limits. Owners and target projections are structurally or completely typed
deduplicated as appropriate; identical replica witnesses merge their source
evidence. Results use canonical owner/path/hop/terminal ordering. A raw
`max_occurrences(n)` cap is one global, post-deduplication truncation, not a
claim that the graph is complete. Direct owner and exact-target projections are
existential and are not truncated by that raw-path cap; projections made from an
already bounded occurrence result remain bounded.

Reference-aware, reference-bearing, and exact-reference containment performs
authoritative root verification. `scan_policy("allow")` permits it and
`scan_policy("warn")` emits the existing warning before the scan;
`scan_policy("forbid")` and `require_indexed()` reject it. `explain()` reports
the target kind, edge policy, reference filter, source scope, scan reason, and
derived index generation evidence without performing the required residual
scan unless `analyze=True`. The legacy eligible materialize-only CDef paths keep
their indexed behavior. Store indexes remain candidate-only derived state, never
authority for root membership or a no-match conclusion.

## Symbolic Definition Selectors

`Definition.as_selector()` retains its ordinary selector behavior. Use
`Definition.loose_selector()` to project known symbolic structure into a loose
ordinary Selector; it intentionally drops unknown parameter, arithmetic, and
topology relationships. Use `Generator.support_selector()` when a query must
verify captured distribution support, linked names, derived values, ordering, and
shared-node topology exactly. Exact Generator support verification is residual to
index prefiltering and never constructs candidate Objects or factory targets. A
provider without an exact proof raises
`UnsupportedGeneratorVerificationError`; a numeric bound alone is never accepted
as a match.

Exact support remains residual because graph-distinct authoritative witnesses
cannot be collapsed into structural index rows. Stored queries use the loose
selector to prefilter indexed identities, then stream matching authoritative
roots before residual verification. The default terminal limit is 65,536
authoritative visits; prefilter rejections and duplicate source visits consume
the budget before suppression. Lower it with `max_witnesses(n)` or explicitly
disable it with `max_witnesses(None)`. Finite-support assignment work is also
bounded cumulatively for the terminal and can raise `ParameterizationLimitError`

`require_indexed()` and `scan_policy("forbid")` reject exact support because the
residual always needs complete witness verification. Exact selectors use exact
symbolic class matching and cannot be converted with `references()` or `where()`,
or rewritten with `categorical()`, `restore()`, or `exact()`. Refining a fixed
result with another exact selector is supported only when the result retained
complete immutable witness evidence; otherwise it raises `QueryDomainError`.

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
