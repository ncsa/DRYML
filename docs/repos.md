# Repos and Stores

Status: draft.

Repos manage DRYML object graphs. Stores own persisted object state. Most user workflows go through a `Repo`; most filesystem details belong to a `Store`.

## Responsibilities

`Repo` responsibilities:

- track live objects
- manage strong and weak caches
- save object graphs
- load objects from definitions
- resolve aliases and main definitions
- query stored, cached, known, and nested definitions
- coordinate one or more stores

`Store` responsibilities:

- own persisted object data
- answer lightweight exact membership checks
- read and write definitions
- provide full hydration when needed
- optionally own a persistent query index

## Basic Save And Load

```python
from dryml.core import Object, Repo


class Item(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


repo = Repo()
item = Item("example", repo=repo)

repo.save_object(item)
loaded = repo.load(item.definition)
```

`repo.save_object()` saves the root object and any graph objects selected by the save plan. `repo.load()` requires an exact `ConcreteDefinition`.

## Stores

The common path-backed store is `DirStore`.

```python
from dryml.core import Repo
from dryml.core.store.dir import DirStore

store = DirStore("./my-dryml-store")
repo = Repo(stores=store)
```

A directory store uses stable definition hashes to organize object state under its object directory. It may also maintain a `.dryml/` sidecar directory for query-index metadata.

## CDef Format Compatibility

Store definitions, aliases, and main-definition references are authoritative.
New DRYML software reads both legacy V1 and bound V2 CDefs and writes new exact
identities as V2. V1 records keep their raw call fields, original hashes, and
paths; they are not automatically migrated or rewritten when read.

| Store contents and operation | New DRYML software | Old DRYML software |
| --- | --- | --- |
| Read an untouched V1-only Store | Supported | Supported |
| Write an untouched V1-only Store | Writes new identities as V2 | Supported only while the Store remains V1-only |
| Read a Store containing V1 and V2 authority | Supported | Unsupported |
| Downgrade or rollback after V2 authority exists | Restore a pre-V2 backup | Do not attempt in place |

There is no transparent V1-to-V2 migration and no recovery of historical
defaults omitted from a V1 raw call. After V2 authoritative data exists, use a
pre-V2 backup for a downgrade or rollback rather than opening the Store with
old software. Query sidecars are derived acceleration data: compatible ones
may be used, and incompatible ones are rebuilt from authoritative Store roots.

## Save Semantics

Important save entry points:

- `repo.save_object(obj)` saves one object graph root.
- `repo.save(obj)` saves an object and flushes stores.
- `obj.save()` saves through the active or provided repo.

Save options include:

- target store
- revision
- alias
- main-definition flag
- ephemeral depth

The save planner builds a concrete-definition graph, stages and validates
required object state, then atomically publishes the object root before
activating repo/store query metadata. Aliases, main definitions, path-backed
Zip archives, and replacement query sidecars are likewise staged before their
final replacement. These publication rules rely on the supported local
filesystem and Python atomic-replace behavior; they do not promise atomicity
for arbitrary `IOBase` implementations or unsupported filesystems.

When an existing object state is replaced, the Store first publishes a complete
new immutable state generation through its atomic pointer. Cooperating readers
hold a per-root reader lease throughout restoration, while a writer holds the
matching exclusive lease for publication and reclamation. Thus a reader finishes
from either its complete old state or the complete new state; once no supported
reader can use it, inactive generation directories are reclaimed and ordinary
successful updates retain only the active generation. An interruption before
pointer replacement keeps the previous state active; an interruption after it
keeps the newly pointed-to state recoverable. A later successful save reclaims
any inactive trees left by that interruption. `ZipStore` writes only the
pointer-reachable state generation when committing an archive.

## Load Semantics

Important load entry points:

- `repo.load(cdef)` loads an existing exact concrete definition.
- `repo.load_or_build(definition_or_cdef)` may construct missing objects.
- `repo.get(selector)` selects existing objects by definition selector.
- `repo.find(...)` provides a higher-level query-and-load path.

Use exact loading when you already have a concrete definition. Use query APIs when you want to discover matching stored objects.

## Query Domains

Stored:

Definitions with committed object state in a store.

Cached:

Definitions currently known through live repo caches.

Known:

Stored plus cached definitions.

Nested:

Definitions that occur inside stored object graphs, even when they are not independently stored roots.

Example:

```python
from dryml.core import Definition, SKIP_ARGS

selector = Definition(Item, SKIP_ARGS)

stored_defs = repo.query(selector).stored().defs()
known_defs = repo.query(selector).known().defs()
```

## Aliases And Main Definitions

Repos and stores can track named aliases and a main definition. These are convenience references, not separate object identities.

Use aliases when a user-facing name is useful:

```python
repo.save_object(item, alias="baseline")
```

Use main definitions when a store or repo should have one default root object.

Alias and main-reference writers validate complete payloads before changing Repo
caches or publishing bytes. On supported local filesystems, `DirStore` also
serializes alias publication across handles: independently changed names merge,
while different concurrent values for the same name fail without replacing the
authoritative alias file. Reopen the Repo (or read aliases through a fresh
Store handle) before retrying a conflict.

Path-backed `ZipStore` archives serialize publication across cooperating
processes with an archive-specific lock. A dirty handle publishes only when the
archive still has the byte digest it extracted; any intervening archive
replacement raises `ZipStoreConflictError` without replacing newer roots or
aliases. Reopen the Store, reapply the intended mutation, and commit again.
Read-only and no-op commits do not compare or rewrite archive bytes. This
stale-writer protection is not available for file-like archive streams.

## Query Indexes

Directory stores can be configured with query-index policies:

- `auto`: use SQLite when available, otherwise fall back safely.
- `sqlite`: require the SQLite persistent query index.
- `memory`: use the in-memory query path.
- `none`: avoid broad index construction; exact probes may still work.

Example:

```python
store = DirStore("./large-store", query_index="sqlite")
repo = Repo(stores=store)
```

The index is acceleration metadata. Object files remain authoritative.

### Index Administration

Repos and directory stores expose backend-neutral maintenance hooks:

```python
status = repo.index_status(store=store)
reports = repo.validate_index(store=store, thorough=True)

repo.rebuild_index(store=store)
store.reconcile_query_index()
```

Use `index_status()` for lightweight diagnostics. Use `validate_index(thorough=True)` when you want filesystem-level checks, including stored-root files and hash-path consistency. Use `rebuild_index()` to explicitly recreate the persistent index from object files.

`reconcile_query_index()` compares the SQLite sidecar with the authoritative Store contents. Missing, dirty, corrupt, incompatible, stale, or divergent indexes are rebuilt by staged replacement from Store roots. A complete replacement is validated before atomic activation; failure retains authoritative object files and does not expose a partial ready sidecar. Concurrent initial rebuild attempts coordinate through a build claim so only one process performs the Store scan.

Corrupt SQLite sidecars are quarantined before rebuild. A changed or misplaced `def.pkl` is reported as Store corruption instead of being silently indexed under the wrong identity.

Diagnostic status records include the backend, Store key, generation, schema version, row counts, journal mode, SQLite runtime version, sidecar path, and backend diagnostics such as WAL safety. Validation reports include structured issues rather than printing from library code.

### Multi-Process Use

Each process and thread opens its own SQLite connection. A worker that saves an object commits its query-index transaction after object files are published. A coordinator with an existing Store/index handle sees that committed root on its next read transaction without reconnecting.

WAL mode can allow long-lived readers and a writer to overlap on supported local filesystems. The default `auto` journal policy is conservative and uses rollback journal unless the SQLite runtime is known safe for WAL.

## Failure Model

Object state is published before persistent query-index root activation. If object publication succeeds and index update fails, the store can be marked dirty and reconciled later.

This means a missing or stale index affects performance or query completeness until reconciliation, but it should not make committed object files invalid.

## Related Docs

- [Objects and Definitions](objects_and_defs.md)
- [Query Index Backend Contracts](query_index_backend_contracts.md)
- [Artifacts API](artifacts.md)
- [Data API](data.md)
