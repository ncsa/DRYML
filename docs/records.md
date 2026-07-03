# DRYML Records

`dryml.records` provides optional store-owned JSON sidecars for metadata that should not be part of DRYML object identity. Records and specs are canonical JSON documents owned by a store. They are not `Object` instances, do not participate in the object graph, and do not change `ConcreteDefinition` hashes or object state bytes.

Sprint 1 intentionally does not change `repo.save()` policy. Calling `repo.save()` or saving an object writes the existing object layout only; callers must use `RecordStoreIO(store)` or `store.records` explicitly to write records/specs.

## Sidecar Layout

A store can remain valid with only the legacy object layout. When record APIs are used, these optional sidecars may be created lazily:

```text
records/
    items/
        <record-id>.json
    specs/
        representation/<repr-id>.json
        operation/<op-id>.json
        environment_record/<envrec-id>.json
        environment_requirement/<envreq-id>.json
        environment_spec/<envspec-id>.json
        environment_lock/<envlock-id>.json
        world/<world-id>.json
        world_requirement/<worldreq-id>.json
        runtime/<runtime-id>.json
        annotation/<annotation-id>.json
        generic/<spec-id>.json
    indexes/
        optional accelerator data
products/
    <record-id>/
        derived product bytes
```

`records/indexes/` is optional and rebuildable. Deleting it does not affect `read_record()`, `read_spec()`, or listing, because JSON files are the source of truth.

## Record Envelopes

Records use the generic `dryml.formats` envelope shape:

```json
{
  "schema": "dryml.record.v1",
  "schema_version": 1,
  "id": "record-v1-...",
  "kind": "stored_state",
  "payload": {
    "storage": [
      {
        "kind": "object-dir",
        "subject_cdef_id": "cdef-v4-...",
        "path": ".",
        "role": "default-state"
      }
    ]
  },
  "metadata": {
    "writer": "dryml.records"
  }
}
```

The record ID is computed from `schema`, `schema_version`, `kind`, and `payload`. It excludes `id`, `metadata`, file path, store locator, mtimes, and index contents. If a timestamp, backend version, or environment fact should affect identity, put it in `payload`, not `metadata`.

## Spec Envelopes

Specs use the same envelope pattern with family-specific schemas and ID prefixes:

| Family | Directory | Schema | Prefix |
|---|---|---|---|
| `representation` | `representation` | `dryml.representation.v1` | `repr` |
| `operation` | `operation` | `dryml.operation.v1` | `op` |
| `environment_record` | `environment_record` | `dryml.environments.record.v1` | `envrec` |
| `environment_requirement` | `environment_requirement` | `dryml.environments.requirement.v1` | `envreq` |
| `environment_spec` | `environment_spec` | `dryml.environments.spec.v1` | `envspec` |
| `environment_lock` | `environment_lock` | `dryml.environments.lock.v1` | `envlock` |
| `world` | `world` | `dryml.world.v1` | `world` |
| `world_requirement` | `world_requirement` | `dryml.world_requirement.v1` | `worldreq` |
| `runtime` | `runtime` | `dryml.runtime.v1` | `runtime` |
| `annotation` | `annotation` | `dryml.annotation.v1` | `annotation` |
| `generic` | `generic` | caller supplied | `spec` |

Operation, world, runtime, and annotation specs are placeholders in Sprint 1. They are interned JSON specs only; they do not implement execution, dispatch, worlds, providers, or runtime behavior.

## References

`RecordRef(record_id)` and `SpecRef(spec_id, kind=None)` validate content IDs without choosing a store. `LocatedRecordRef(store_ref, record_id)` and `LocatedSpecRef(store_ref, spec_id, kind=None)` add a store locator string for a specific copy.

Record refs serialize compactly as the raw record ID. Located refs serialize as objects:

```json
{
  "store_ref": "dryml.core2.store.dir.DirStore:/path/to/store",
  "record_id": "record-v1-..."
}
```

Located refs are stable within the current repo/session. They are not guaranteed to be globally dereferenceable.

## StorageRef

`StorageRef` is pure metadata. It stores logical, store-relative pointers and never persists physical object shard paths.

Supported forms:

```python
StorageRef.object_dir("cdef-v4-...", path=".", role="default-state")
StorageRef.product_dir("record-v1-...", path="derived/output", role="artifact")
StorageRef.blob("blob-v1-...", role="weights")
```

Paths must be POSIX-style relative paths. Absolute paths, drive-prefixed paths, backslashes, empty components, and `..` traversal are rejected. `path="."` means the logical root.

`RecordStoreIO.resolve_storage_ref()` resolves `object-dir` refs to the existing object directory and `product-dir` refs under `products/<record-id>/`. Product directories may be created with `create=True`; object directories are never fabricated. Blob resolution is a placeholder until blob storage exists.

## Store IO

Use `RecordStoreIO(store)` or the convenience `store.records` property:

```python
from dryml.records import RecordStoreIO, make_record

records = RecordStoreIO(store)
record = make_record(kind="stored_state", payload={"storage": []})
located = records.write_record(record)
loaded = records.read_record(located.record_id)
```

Writes use canonical JSON bytes and atomic temp-file-plus-replace writes. Rewriting the same ID with identical bytes is idempotent. Rewriting the same ID with different bytes is rejected unless the caller explicitly opts into overwrite behavior.

## Non-Goals

Records/specs are not Objects, do not subclass `Object`, and are not stored under `objects/`. There is no SQLite or index dependency for correctness. Recursive scanners, provenance queries, operation call semantics, dispatch/runtime/world behavior, adapter resolution, and automatic save/export record policies are future work.
