# DRYML Representations And Adapters

Representation specs describe stored state, data, and program product formats without changing `ConcreteDefinition` identity. Records and products are store-owned sidecars, not `Object` instances.

## Representation Specs

Use `RepresentationSpec` and `make_representation_spec(...)` for `family="representation"` specs:

```python
from dryml.records import make_representation_spec

spec = make_representation_spec(
    "fake.raw_state",
    version="1",
    traits=("loadable-state",),
    storage_kinds=("product-dir",),
)
```

`RepresentationRequirement` is used for queries. Compatibility is conservative: exact representation ID, exact kind, exact version, exact requested parameter values, required trait subset, and required storage-kind subset. Provider-specific semantic compatibility can be added later through provider reports; unknown semantics are not assumed compatible.

Minimum fake kinds used by tests are `dryml.object_state`, `fake.raw_state`, `fake.normalized_state`, `fake.data.table`, and `fake.program.ir`.

`dryml.numpy_sequence` is the lightweight managed CachedDataset representation.
It stores backend-neutral dense NumPy leaves in bounded row/byte `.npz` shards,
supports nested dict/list/tuple tensor trees, and uses one compact `index.json`
for row ranges, tree metadata, sizes, and digests. It does not use one file per
element and does not import optional dataframe or framework backends.

## Typed Records

`StoredStateRecord` points at loadable object state. Descriptive saves still store bytes under `objects/`; wrappers only validate the record sidecar.

`DataRecord` points at product-dir or object-dir data products such as metrics, tables, arrays, previews, or cached data. It is not the public Artifact API.

`ProgramRecord` points at IR/codegen/compiler-like products. Real compiler/JIT execution is out of scope.

`AdapterRecord` preserves source/target lineage for representation conversion. It records `source_record_id`, `source_representation_id`, `target_record_id`, `target_representation_id`, `produced_records`, and `derived_from`.

When a `DataRecord` or `StoredStateRecord` source has managed ownership, the adapter target preserves its `realization_id` and `output_slot`. This identifies conversion as another representation of the same realization; an independent recomputation must use a different realization ID.

## Resolution

Resolution scans authoritative JSON and does not require `records/indexes/`:

```python
from dryml.records import RepresentationRequirement, resolve_state_record

result = resolve_state_record(
    repo,
    cdef_id,
    RepresentationRequirement(kind="fake.normalized_state"),
    adapters=registry,
)
```

An existing compatible stored-state record is selected before planning adapters. `status="ok"` means `selected` is directly loadable and satisfies the requested representation. If an adapter path is available but has not run yet, resolution returns `status="requires_adapter"` with `adapter_plan` and `adapter_source`; `selected` remains empty so callers do not mistake the source record for the target representation. Normal `not_found`, `unsupported`, and missing or invalid representation-spec outcomes are reported structurally.

## Fake Adapter Example

The local adapter runner is a temporary fake/test hook, not dispatch v2. It runs in-process and requires a registered callback:

```python
from dryml.records import AdapterDescriptor, AdapterRegistry, RepresentationRequirement

registry = AdapterRegistry()

def runner(context):
    context.session.write_text("normalized.txt", "normalized")
    return {}

registry.register(
    AdapterDescriptor(
        "fake.normalize",
        RepresentationRequirement(kind="dryml.object_state"),
        RepresentationRequirement(kind="fake.normalized_state"),
    ),
    runner=runner,
)
```

`run_adapter_plan(...)` resolves source storage, gives the runner a `ProductWriteSession`, writes target products under `products/<target-record-id>/`, writes the target state/data/program record with a self product-dir ref, writes an `AdapterRecord`, and returns located refs.

Failed local adapter attempts are returned as structured `AdapterExecutionResult` failures. They are not persisted as failed records; durable failure provenance is not implemented.

The local adapter runner does not implement real Torch, TensorFlow, JAX,
DeepSpeed, Conda-worker, subprocess-dispatch, cancellation, or worker-handshake
execution. Dispatch does not consume adapter plans or emit `ExecutionRecord`
provenance for them.

## Product Identity

Self product-dir refs avoid record-ID circularity:

```json
{"kind": "product-dir", "path": ".", "role": "target-state"}
```

Resolve them with the containing record ID. Product bytes do not affect CDef identity. Product manifests can affect record identity because they live in the record payload. Use `validate_product_availability(...)` before treating product-backed records as complete or before exporting/copying product-backed closures.

## Non-Goals

`ExecutionRecord` is optional provenance and is not required for load/adapt. The
shipped object-based `dryml.artifacts` API is distinct from store-owned
`DataRecord` products; `DataRecord` does not implement or select Artifacts.
Framework-specific adapters and compiler/JIT lowering are deferred.
