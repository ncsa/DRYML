# Persistent Metadata

DRYML Store v3 keeps descriptive metadata attached to exact `ObjectRef` and
`StateRef` identities. Metadata inspection is lightweight: it validates Store
records without materializing Objects, opening local-state payloads, importing
optional frameworks, or observing the current host.

## Four Scopes

| Scope | Target | Mutability | Meaning |
| --- | --- | --- | --- |
| `object` | exact `ObjectRef` | current whole-map LWW | User annotations for one lineage identity |
| `state` | exact `StateRef` | current whole-map LWW | User annotations for one exact state |
| `lineage` | exact `ObjectRef` | write once | Known UTC creation time or explicit unknown evidence |
| `snapshot` | exact `StateRef` | write once | Save time, environment, requirements, lineage table, diagnostics, and captured annotation copies |

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
repo.set_metadata(state_ref.object, {
    "project": "forecasting",
    "labels": ["candidate", "reviewed"],
})
repo.set_metadata(state_ref, {"score": 0.91})

object_values = repo.get_metadata(state_ref.object)
removed = repo.delete_metadata(state_ref)
```

Without `store=`, reads compare every connected Store that holds the target and
raise `MetadataConflictError` on disagreement, including absent versus present.
Writes require an explicit Store when more than one writable connected Store
holds the target. Unknown targets raise `KeyError`; malformed authority raises
`StoreAuthorityError`.

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

For a newly published snapshot, Store publication reads existing current
mappings and applies explicit replacements under one writer fence, then installs
immutable `SnapshotMetadata`. A repeated save of the same `StateRef` may apply
new explicit current mappings, but it retains the first complete captured view.
Later `set_metadata()` or `delete_metadata()` calls likewise do not rewrite
history.

`Repo.get_snapshot_metadata(state_ref)` returns that detached captured record.
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
descendant. `Repo.get_lineage_metadata(object_ref)` returns known immutable
evidence or a valid explicit unknown marker.

Saving observes the actual saving process once per new capture. Metadata reads,
queries, repeated saves of complete snapshots, and copies do not observe the
host. `dryml.environments.inspect_current()` and probe helpers are separate,
explicit operations. A current observation after code or package changes may
differ from stored capture without mutating it. Matching metadata is descriptive
evidence, not an execution-admission guarantee.

## Copies, Routing, And Recovery

Identity-preserving snapshot copies and exact loads retain the complete selected
source capture; they do not recapture the destination process. If multiple
connected Stores hold different evidence for the same `StateRef`, select the
root with `source_store=`. For independently routed exact child references, use
`source_stores={child_state_ref: store}` with `Repo.load_state_ref`,
`Repo.restore_state_ref_into`, or `load_state_ref`. Selection chooses whole
records; DRYML never combines environment from one replica with annotations or
lineage from another. Unqualified differing evidence raises
`MetadataConflictError` before construction, restoration hooks, live-cache
reuse, or target mutation.

Per-object routed snapshots record exact child `StateRef` projections. The root
Store can therefore need another connected Store to restore a child's payload.
Metadata inspection does not need that payload, but restoration fails if a
required child Store is absent. `Repo.get_snapshot_directory()` returns a
borrowed location. A DirStore path remains persistent; a ZipStore path and any
`LocalStateSource` from it are valid only until that Store handle closes.

There is no cross-Store transaction. With `report_stores=True`, `StoreReport`
records exact per-boundary states (`completed`, `failed`, `unattempted`, or
`uncertain`) for snapshots, current annotation writes, indexes, names, and
commits. If a current write fails after immutable snapshot installation, the
captured mapping remains authoritative and the failed current boundary is
reported. Recovery inspects completed authority and retries deliberately; it
does not delete completed snapshots or replay captured history over intervening
current edits.

## Metadata Queries

Use `field()` with `Repo.references().where(...)`. Repeated `where()` calls
intersect predicates. Expressions combine with `&`, `|`, and `~`.

```python
from dryml.core import field

matches = (
    repo.references()
    .where(field("object", "project").eq("forecasting"))
    .where(field("snapshot", "environment_status").eq("known"))
    .state_refs()
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

Store v3 snapshot metadata and lineage records are framework-owned persistence.
Current annotation records are separate mutable authority. The compatibility
commitment covers these records, associations, and placement, not Python API
stability or migration of arbitrary author-owned payload codecs. Store v2 is
rejected without rewriting; no in-place migration or fallback reader exists.

Sharing, copying, or replicating a real DirStore or ZipStore also shares retained
environment fields. These can include interpreter executable/prefix paths,
distribution locations, and environment names or paths. Immutable snapshot
copies retain those values even after current annotations are edited. DRYML does
not add filtering or redaction policy at this boundary; decide whether a Store is
appropriate to share. Checked-in compatibility fixtures use fixed public
synthetic values only.
