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
from dryml.core2 import Object, Repo


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
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore

store = DirStore("./my-dryml-store")
repo = Repo(stores=store)
```

A directory store uses stable definition hashes to organize object state under its object directory. It may also maintain a `.dryml/` sidecar directory for query-index metadata.

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

The save planner builds a concrete-definition graph, saves required object state to the selected store, then updates repo/store query metadata after object files are published.

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
from dryml.core2 import Definition, SKIP_ARGS

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

`reconcile_query_index()` compares the SQLite sidecar with the authoritative Store contents. In the current v1 policy, missing, dirty, corrupt, incompatible, stale, or divergent indexes are repaired by an exclusive rebuild from Store roots. Concurrent initial rebuild attempts coordinate through a build claim so only one process performs the Store scan.

Corrupt SQLite sidecars are quarantined before rebuild. A changed or misplaced `def.pkl` is reported as Store corruption instead of being silently indexed under the wrong identity.

## Failure Model

Object state is published before persistent query-index root activation. If object publication succeeds and index update fails, the store can be marked dirty and reconciled later.

This means a missing or stale index affects performance or query completeness until reconciliation, but it should not make committed object files invalid.

## Related Docs

- [Objects and Definitions](objects_and_defs.md)
- [Artifacts API](artifacts.md)
- [Data API](data.md)
