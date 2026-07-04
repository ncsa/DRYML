# DRYML Records

`dryml.records` provides optional store-owned JSON sidecars for metadata that should not be part of DRYML object identity. Records and specs are canonical JSON documents owned by a store. They are not `Object` instances, do not participate in the object graph, and do not change `ConcreteDefinition` hashes or object state bytes.

Save/export record policies are explicit. The default policy is now `descriptive`: calling `repo.save()` without a policy writes the existing object layout and emits direct descriptive sidecars for saved object state. Use `record_policy="none"` when a save must write object bytes only and create no `records/`, no `products/`, and no record reference index.

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
        ref-index-v1.json
        ref-index-v1.dirty
products/
    <record-id>/
        derived product bytes
```

`records/indexes/` is optional and rebuildable. Deleting it does not affect `read_record()`, `read_spec()`, or listing, because JSON files are the source of truth.

## Save And Export Policies

The public policy values are:

| Policy | Behavior |
|---|---|
| `none` | Save object bytes only. Do not create record/spec/product/index sidecars. |
| `descriptive` | Default. Save object bytes and emit direct `stored_state` records plus the default object-state representation spec. For copy/export planning, include explicit seed records/specs only and do not expand referenced specs. |
| `closure` | Include seed records and specs referenced by those records/specs. Do not follow provenance ancestry or products by default. |
| `provenance` | Explicitly include record lineage such as `derived_from`, consumed/produced records, and existing execution/adapter/probe records that mention seeds. |
| `all` | Include all records and specs in the selected source store. Existing product directories are included by default for `all`; indexes are still omitted as authoritative data. |

Policy options are available through `RecordPolicyOptions`. `include_products=None` means the policy default: false for `none`, `descriptive`, `closure`, and `provenance`, and true for `all`. Set `include_products=True` to copy existing product directories for included records. Records without product directories are valid and are skipped. Set `rebuild_index=True` to rebuild `records/indexes/ref-index-v1.json` after writes or copies. Set `overwrite_sidecars=True` only when replacing existing record/spec/product sidecars is intended. Indexes are derived and should be rebuilt in the destination rather than copied.

Example descriptive save:

```python
repo.save(obj, record_policy="descriptive")
```

For each saved object action, `descriptive` writes a direct `stored_state` record like:

```json
{
  "subject_cdef_id": "cdef-v4-...",
  "representation_id": "repr-v1-...",
  "storage": [
    {
      "kind": "object-dir",
      "subject_cdef_id": "cdef-v4-...",
      "path": ".",
      "role": "default-state"
    }
  ],
  "save": {
    "reason": "explicit-root",
    "minimum_root_depth": 0
  }
}
```

If a save revision is supplied, it is recorded under `payload.save.revision`. The payload does not include absolute paths, source/destination store refs, mtimes, or index facts. The storage ref is logical and resolves relative to the current store through `RecordStoreIO.resolve_storage_ref()`.

The default object-state representation spec is in family `representation`, schema `dryml.representation.v1`, kind `dryml.object_state`, and has a stable payload describing the object-dir default-state layout:

```json
{
  "format": "dryml.object_state",
  "storage_kind": "object-dir",
  "role": "default-state",
  "description": "Default DRYML object state layout written under objects/."
}
```

Record closure copy/export uses authoritative record/spec JSON, not copied indexes:

```python
from dryml.records import copy_record_closure, plan_record_closure, record_export_include_paths

report = copy_record_closure(
    source_store,
    destination_store,
    seed_records=[record_id],
    policy="closure",
)

plan = plan_record_closure(source_store, seed_records=[record_id], policy="closure")
include_paths = record_export_include_paths(plan)
```

`record_export_include_paths(plan)` returns paths such as `records/items/<record-id>.json`, `records/specs/representation/<repr-id>.json`, and `products/<record-id>/` when products are included. It omits `records/indexes/` by default. Passing these paths to `ZipExportStore` preserves store-relative object-dir/product-dir refs; reopen the destination and rebuild the reference index if an index is needed there.

Repo-level federation is exposed as `repo.records`. It delegates to each store's `RecordStoreIO` and supports locating records/specs, reading located refs, querying CDef mentions, finding operation specs, and copying one unambiguous closure to a destination store.

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

Record envelopes accept only these top-level keys: `schema`, `schema_version`, `id`, `kind`, `payload`, and `metadata`. Semantic fields must live under `payload`. Sprint 1 record kinds are a closed set: `stored_state`, `data`, `execution`, `adapter`, `program`, `probe_report`, `compatibility_report`, and `lowering_report`.

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

Operation specs now support Sprint 2 `function_call` and `method_call` payloads through `dryml.operations`. They are interned JSON specs only; they do not implement execution, dispatch, worlds, providers, or runtime behavior. World, runtime, and annotation specs remain metadata placeholders.

Spec envelopes accept only these top-level keys: `schema`, `schema_version`, `id`, `kind`, `payload`, and `metadata`. Semantic fields must live under `payload` for the same identity reason as records.

## References

`RecordRef(record_id)` and `SpecRef(spec_id, kind=None)` validate content IDs without choosing a store. Unqualified `SpecRef` values must still use a known Sprint 1 spec prefix and schema version. `LocatedRecordRef(store_ref, record_id)` and `LocatedSpecRef(store_ref, spec_id, kind=None)` add a store locator string for a specific copy.

Record refs serialize compactly as the raw record ID. Located refs serialize as objects:

```json
{
  "store_ref": "dryml.core2.store.dir.DirStore:/path/to/store",
  "record_id": "record-v1-..."
}
```

Located refs are stable within the current repo/session. They are not guaranteed to be globally dereferenceable.

## Reference Scanning

`dryml.records.scanner` scans validated record/spec `payload` JSON for reserved DRYML references. Metadata and top-level IDs are not scanned by default because they are not semantic identity fields.

The scanner recognizes:

- raw CDef IDs such as `cdef-v4-...` as materializing CDef mentions;
- non-materializing CDef refs such as `ref(cdef-v4-...)`;
- reserved content IDs such as `record-v1-*`, `op-v1-*`, `repr-v1-*`, `envreq-v1-*`, `worldreq-v1-*`, `world-v1-*`, and `runtime-v1-*`;
- exact literal escapes such as `{"$literal": "cdef-v4-..."}` as opaque values that produce no mention.

Malformed literal escapes and malformed reserved-looking strings fail loudly instead of being ignored.

Known typed keys are also recognized and validated. Examples include `subject_cdef_id`, `owner_cdef_id`, `input_cdef_ids`, `output_cdef_ids`, `operation_id`, `representation_id`, `environment_requirement_id`, `world_requirement_id`, `world_id`, `runtime_id`, `record_id`, and `derived_from`. Prefix mismatches such as `operation_id="repr-v1-*"` are rejected. For currently known DRYML record/spec prefixes, typed keys also require the current schema version, such as `record-v1-*` and `op-v1-*`. The future `env-v*` prefix accepted by `environment_id` is prefix-compatible only until that schema is introduced.

Scanner output uses deterministic JSON Pointer paths such as `/payload/storage/0/subject_cdef_id`.

## Reference Index

`RecordStoreIO.rebuild_ref_index()` rebuilds a store-local JSON index at:

```text
records/indexes/ref-index-v1.json
```

The index is derived data. It contains source records/specs plus scanner mentions and can be deleted and rebuilt from authoritative JSON sidecars. It is canonical JSON and does not use SQLite. A valid index whose stored `store_ref` no longer matches the current store is treated as stale: `refresh="auto"` rebuilds it, while `refresh=False` raises a validation error.

When `write_record()` or `write_spec()` changes canonical bytes after an index exists, `RecordStoreIO` writes `records/indexes/ref-index-v1.dirty`. Idempotent writes of identical bytes do not mark the index dirty.

Query helpers include:

```python
records.find_mentions(target_id=cdef_id, target_kind="cdef")
records.find_records_mentioning_cdef(cdef_id)
records.find_specs_mentioning_cdef(cdef_id, family="operation")
records.find_operation_specs_for_cdef(cdef_id, cdef_semantics="materialize")
```

`refresh="auto"` rebuilds when the index is missing, dirty, or corrupt. `refresh=True` always rebuilds. `refresh=False` requires a present, clean, valid index and raises a clear index error otherwise.

## StorageRef

`StorageRef` is pure metadata. It stores logical, store-relative pointers and never persists physical object shard paths.

Supported forms:

```python
StorageRef.object_dir("cdef-v4-...", path=".", role="default-state")
StorageRef.product_dir("record-v1-...", path="derived/output", role="artifact")
StorageRef.blob("blob-v1-...", role="weights")
```

Paths must be POSIX-style relative paths. Absolute paths, drive-prefixed paths, backslashes, empty components, and `..` traversal are rejected. `path="."` means the logical root.

`RecordStoreIO.resolve_storage_ref()` resolves `object-dir` refs through `store.object_dir_for_cdef_id(cdef_id)` when the store provides that hook. The base `Store` hook matches the current `objects/<first-two-hex>/<digest>` layout and requires a full 64-hex CDef digest. Future store backends can override the hook without changing record JSON. Product-dir refs resolve under `products/<record-id>/`. Product directories may be created with `create=True`; object directories are never fabricated. Blob resolution is a placeholder until blob storage exists.

Product-dir refs may omit `record_id` inside a record payload:

```json
{
  "kind": "product-dir",
  "path": ".",
  "role": "target-state"
}
```

This is a self product-dir ref. It resolves against the containing record ID, avoiding the circular identity problem of putting a record's own ID in its payload before the ID exists:

```python
storage = StorageRef.self_product(path=".", role="target-state")
path = store.records.resolve_storage_ref(storage, record_id=record["id"], create=True)
```

Explicit cross-record product refs still include `record_id` and continue to resolve under that other product root. The reference scanner indexes explicit `record_id` product refs but does not invent a self-reference mention for omitted IDs.

## Typed Records And Representations

The generic JSON envelope remains authoritative. `StoredStateRecord`, `DataRecord`, `ProgramRecord`, and `AdapterRecord` are ergonomic wrappers that validate payload shape and round-trip through `make_record(...)`. Existing descriptive save records parse as `StoredStateRecord`; their object bytes remain under `objects/` and are not moved.

`RepresentationSpec` wraps the existing `family="representation"` spec envelope. `RepresentationRequirement` supports conservative deterministic checks by exact representation ID, kind, version, parameter equality, required trait subset, and storage-kind subset. Unknown provider/framework semantics do not imply compatibility.

Store-local discovery can scan source JSON without the optional index:

```python
store.records.find_records(kind="stored_state", subject_cdef_id=cdef_id)
```

Repo-level helpers in `dryml.records` include `find_stored_state_records(...)`, `find_compatible_state_record(...)`, and `resolve_state_record(...)`. Normal not-found and unsupported outcomes return structured `RecordResolutionReport` data instead of raising.

## Product Manifests

`ProductWriteSession` stages bytes first, computes a `ProductManifest`, attaches the canonical record ID, moves files under `products/<record-id>/`, then writes the record sidecar:

```python
with ProductWriteSession(store.records) as session:
    session.write_text("state.json", "{}")
    manifest = session.manifest()
    record = StoredStateRecord(
        subject_cdef_id=cdef_id,
        representation_id=repr_id,
        storage=(StorageRef.self_product(role="target-state"),),
        manifest=manifest.to_json(),
    )
    result = session.commit_record(record.to_envelope())
```

Manifest paths are relative POSIX paths and include byte size plus SHA-256 digest. Product bytes do not affect CDef identity; they only affect record identity if their manifest is placed in the record payload.

## Adapter Lineage

`AdapterRecord` is lineage for representation conversion. It records source/target record IDs, source/target representation IDs, produced records, `derived_from`, status, and diagnostics. It is not a replacement for `ExecutionRecord`; execution provenance remains optional and is not required for load/adapt.

## Store IO

Use `RecordStoreIO(store)` or the convenience `store.records` property:

```python
from dryml.records import RecordStoreIO, make_record

records = RecordStoreIO(store)
record = make_record(kind="stored_state", payload={"storage": []})
located = records.write_record(record)
loaded = records.read_record(located.record_id)
```

Writes use canonical JSON bytes and atomic temp-file-plus-replace writes. Rewriting the same ID with identical bytes is idempotent. Rewriting the same ID with different bytes is rejected unless the caller explicitly opts into overwrite behavior. Changed record/spec writes mark the optional reference index dirty when index tracking already exists.

## Non-Goals

Records/specs are not Objects, do not subclass `Object`, and are not stored under `objects/`. There is no SQLite or index dependency for correctness. Full dispatch v2, generated `ExecutionRecord` provenance, real framework adapters, blob storage, compiler/JIT execution, and the public Artifact API remain future titled-sprint work.
