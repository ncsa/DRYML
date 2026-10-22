# Persistent Metadata

DRYML Store 4 keeps descriptive metadata attached independently to Node object
projections and exact saved Node graphs. Metadata inspection is lightweight: it validates Store
records without materializing Objects, opening local-state payloads, importing
optional frameworks, or observing the current host.

## Four Scopes

| Scope | Target | Mutability | Meaning |
| --- | --- | --- | --- |
| `object` | Node object projection | current whole-map LWW | User annotations for one realized graph |
| `state` | exact saved Node | current whole-map LWW | User annotations for one exact saved graph |
| `lineage` | Node object projection | write once | Known UTC creation time or explicit unknown evidence |
| `snapshot` | exact saved Node | write once | Save time, environment, requirements, lineage table, diagnostics, and captured annotation copies |

Current object and state mappings are independent. `None` means no mapping is
present; `{}` is a present empty mapping; a present `None` value inside a mapping
is different from both. Values may contain bounded scalars, lists, tuples, and
string-keyed nested mappings. Booleans, integers, and floats retain distinct
types.

`Repo.set_metadata()` atomically replaces an entire current mapping and
`Repo.delete_metadata()` removes it. These are Store-local last-writer-wins
(LWW) operations, not field merges. Two writers that read the same mapping and
replace different fields can lose one update; coordinate such read-modify-write
operations in application code. The Store writer fence prevents a torn mapping,
not logical merge loss.

```python
repo.set_metadata(saved_node.object_view(), {
    "project": "forecasting",
    "labels": ["candidate", "reviewed"],
}, scope="object")
repo.set_metadata(saved_node, {"score": 0.91}, scope="state")

object_values = repo.get_metadata(saved_node.object_view(), scope="object")
removed = repo.delete_metadata(saved_node, scope="state")
```

Without `store=`, reads compare every connected Store that holds the target and
raise `MetadataConflictError` on disagreement, including absent versus present.
Writes require an explicit Store when more than one writable connected Store
holds the target. Unknown targets raise `KeyError`; malformed authority raises
`StoreAuthorityError`.

The optional `scope=` on current-mapping CRUD is `"object"` or `"state"`.
Omitting it infers state when the Node has selected state, otherwise object.
Stateless saved graphs require explicit state scope because their saved and
object-projection graph content can be equal. Invalid scopes fail before mutation.
Scope and holder authority are never inferred from a pure content-cache hit.

## Current Reader and Query Qualification

Store-4 metadata queries capture requested scopes and validate all eligible
holders in those scopes before candidate narrowing or scalar early return.
Current `Store.read_metadata(node, scope="object" | "state")` overrides remain
authoritative; their read failures are query failures, not missing fields.
Repeated object projections share one holder read per detached cut.

A published saved root requires its captured-metadata record. A missing record
is incomplete authority, not unknown lineage or an absent-field match. Exact
loads check current full-key, creation, payload, declaration and capture evidence
before application hooks. Load plans retain pure preparation and associations,
not borrowed payload readers or cross-call availability proof.

Current backend seams include `read_states`, `read_creation`, `read_node`, and
their framed-record/authentication readers. Retired StateRef/snapshot-JSON
readers and V2 LocalStateManifest hooks are not extension points for Node
records. A missing key is omitted from `read_states`' result; exceptions from a
selected reader reject the operation instead of silently choosing another
source. Owning Store authority errors may propagate directly; state-source
preflight wraps source/payload failures in `RepoLoadError`.

Pure preparation reuse is operation-local and byte-bounded. Oversized or
zero-budget pure facts use fresh decoding without rejecting a valid query.
Cold, warm, cleared, borrowed and reopened readers still validate current
authority and payloads at their owning boundaries. Archive loads require valid
current committed contents; a buffered retained Repo save is refused by the
durable-publication contract.

## Migration Only: Constructor Mixins

The persistent APIs above replace the retired `Metadata` and `UniqueID`
constructor mixins. This section is migration guidance, not a supported historical code
example. Move descriptive labels from construction to `SaveAnnotations` or the
reference-targeted CRUD methods above. Use `get_lineage_metadata()` for known
creation evidence and `get_snapshot_metadata()` for immutable save evidence.

Use `Serializable` and ObjectRef/StateRef facilities when an application needs
framework identity or exact-state identity. When a class needs a distinct
structural definition, declare an explicit ordinary constructor discriminator.
`uid` and `metadata` are no longer reserved names: a class may use them as normal
parameters, they remain part of its construction constraints, and categorical
projection removes them only through an explicit named `drop`.

This migration does not rewrite Stores, old definitions, payloads, or readers.
An incompatible retired authority fails at its actual resolution or construction
boundary rather than being silently stripped or reidentified.

## Save-Time Capture

Pass `SaveAnnotations` to any ordinary save wrapper to replace current root
mappings and include those values in a new snapshot's captured view:

```python
from dryml.core import SaveAnnotations

state_ref, report = repo.save_object(
    model,
    annotations=SaveAnnotations(
        object={"project": "forecasting"},
        state={"quality": {"status": "candidate"}},
    ),
    report_stores=True,
)
```

For a newly published saved graph, detached Store preparation reads existing
current mappings and applies requested replacements/capture in the same journaled
batch. Short publication fences revalidate exact cut/bytes before effects and
enabled index coverage is current before complete success. Readers return detached
`NodeSnapshotMetadata`. A repeated save of the same saved graph may apply
new explicit current mappings, but it retains the first complete captured view.
Later `set_metadata()` or `delete_metadata()` calls likewise do not rewrite
history.

`Repo.get_snapshot_metadata(saved_node)` returns that detached captured record.
It contains:

- `saved_at`, encoded as timezone-aware UTC Unix seconds;
- an `EnvironmentRecord` with status `known`, `incomplete`, or `unavailable`;
- an `EnvironmentRequirement` outcome `empty`, `value`, `conflict`, or
  `unavailable`, plus `complete` or `incomplete` declaration coverage;
- bounded diagnostics;
- root and primary-path lineage facts;
- captured object/state mappings, preserving absent versus present-empty.

Environment status and requirement outcome are independent. Incomplete or
unavailable evidence is retained honestly and is not converted into proof of
compatibility. A stateless root has no `ObjectId`, so its root creation fact is
`unknown`; DRYML does not infer creation from save time, file timestamps, or a
descendant. `Repo.get_lineage_metadata(node.object_view())` returns known immutable
evidence or a valid explicit unknown marker.

An ordinary save observes the actual saving process once, outside Store fences;
an already installed capture remains unchanged. Metadata reads and queries do not
observe the host. `dryml.environments.inspect_current()` and probe helpers are separate,
explicit operations. A current observation after code or package changes may
differ from stored capture without mutating it. Matching metadata is descriptive
evidence, not an execution-admission guarantee.

## Copies, Routing, And Recovery

Exact Node loads compare existing captured evidence from selected root and independently published child holders
without observing the destination host. If multiple connected Stores hold
different captures for the same saved graph, select root authority with
`source_store=`. Per-key provider overrides use `source_stores={state_key: store}`.
Capture records are selected whole; conflicting unqualified capture or lineage
evidence fails before constructors/restore hooks. Known lineage is installed on
the live identity before a later save, including a save to another Store.
Full targeted restoration performs the same fresh comparison, including lineage
on an uncached supplied target, before entering any restore hook. Source choices
select providers for the whole selected-key closure; routed children can override
that choice with their full StateKeys.

Low-level saved-graph publication is distinct from descriptive capture. Loading a
caller-supplied saved Node can use validated LocalStates without a separately
published capture; explicitly requesting unavailable captured metadata refuses.
An ordinary save to a new Store creates that Store's first capture and can therefore
differ from another Store's capture of equal saved graph content.

Routed saved Nodes carry full child StateKeys. The root Store can need another
connected Store to restore private data. Metadata inspection does not require that
data, but restoration requires a fresh active read lease and valid payloads.
Borrowed StateReadSources expire when their context or Store closes.

There is no cross-Store transaction. With `report_stores=True`, `StoreReport`
records per-target batch outcomes (`completed`, `failed`, `unattempted`, or
`uncertain`), confirmed saved graphs/full keys, mutation/cut identity and index
disposition. A metadata/index failure after authority effects is a partial
publication failure. Recovery classifies actual records and replays derived work;
it neither deletes completed graph/state authority nor reruns capture merely to
repair an index. Missing capture records remain unavailable when requested.

## Metadata Queries

Use `field()` with an identity query. Repeated `where()` calls intersect predicates.
Expressions combine with `&`, `|`, and `~`.

```python
from dryml.core import field

matches = (
    repo.query().state_refs()
    .where(field("object", "project").eq("forecasting"))
    .where(field("snapshot", "environment_status").eq("known"))
    .collect()
)
```

Scopes are `object`, `state`, `lineage`, and `snapshot`. Operators are
`exists()`, `missing()`, `eq()`, `contains()`, `lt()`, `le()`, `gt()`, and
`ge()`. Existence distinguishes present null from missing. Equality is typed.
Ordering accepts finite non-boolean numbers and documented UTC timestamp fields;
invalid scope/path/operator/type combinations fail during predicate construction
or authoritative evaluation rather than becoming non-matches. `contains()` is
literal string containment or direct sequence membership, not regex or recursive
subtree search.

Indexes may accelerate metadata predicates, but terminals verify Store authority.
A missing, dirty, corrupt, or incompatible index falls back to or rebuilds from
authoritative records. Malformed metadata, replica conflicts, and unsupported
predicates remain visible errors even when another candidate would match.

## Persistence And Sharing

Store-4 Node metadata and lineage records are framework-owned persistence.
Current annotation records are separate mutable authority. The compatibility
commitment covers these records, associations, and placement, not Python API
stability or migration of arbitrary author-owned payload codecs. The closed
`dryml-node-metadata/v1` family lives in
`metadata/{current_object,current_state,lineage,capture}/<shard>/<digest>.record`.
The same Store journal feeds `requested_metadata` index coverage. Managed lifecycle
associations remain separately unavailable. Store 2/3 inputs are rejected intact;
standalone legacy reference codecs are not a persistent-format fallback.

Sharing, copying, or replicating a real DirStore or ZipStore also shares retained
environment fields. These can include interpreter executable/prefix paths,
distribution locations, and environment names or paths. Immutable snapshot
copies retain those values even after current annotations are edited. DRYML does
not add filtering or redaction policy at this boundary; decide whether a Store is
appropriate to share. Checked-in compatibility fixtures use fixed public
synthetic values only.
