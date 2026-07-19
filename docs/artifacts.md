# Artifacts API

Status: draft.

Artifacts are repo-backed computed payloads. They are useful for values that are derived from other objects, expensive to recompute, or useful to persist independently from model definitions.

## Artifact Base Class

`Artifact` extends `Serializable` and provides a common base for computed values.

Important methods:

- `compute()`: produce or update the artifact payload.
- `exists()`: check whether the artifact appears to have persisted state.

Artifacts can use repo/store locations to save computed payloads.

## Why Artifacts Exist

Definitions should describe identity and construction. Artifacts represent outputs.

Examples:

- scalar metrics
- aggregate metrics
- cached dataset materializations
- generated reports
- model evaluation outputs

## Scalar Artifacts

Initial scalar artifact types include:

- `Scalar`
- `ScalarAgg`
- `ScalarAvg`
- `Accuracy`

These types are intended for storing and aggregating scalar-like computed values.

## Cached Datasets

`CachedDataset` is a normal lightweight `Dataset` whose completed bytes are a
managed, record-backed realization. Its definition stores the source as a
non-materializing `RefCDef` plus element spec, cardinality, and order metadata;
loading the definition or reading a completed cache does not construct the
source.

The first realization must choose the NumPy sequence representation explicitly:

```python
cached = CachedDataset(source)
cached.compute(
    store=store,
    representation="numpy-sequence",
    shard_rows=1024,
    shard_bytes=64 * 1024 * 1024,
)
```

Normal iteration resolves only the compatible active `DataRecord` in the
default repository scope. `cached.view(repo=repo)` and
`cached.view(store=store)` pin iteration to one authority. Missing or incomplete
results raise; iteration never computes the source implicitly. A rerun keeps the
old active realization readable until the new result completes and activates.

NumPy sequence products contain bounded `.npz` shards and one compact
`index.json`; files scale with shards rather than rows. Reads verify the
record-owned product manifest and the sequence index/shard sizes and digests.
CachedDataset never treats a source prefix as a completed result.

A completed NumPy cache can derive a Parquet representation without rerunning
its source:

```python
result = cached.request_representation("parquet", store=store)
if result.status != "ok":
    print(result.issues)
```

The request is restricted to the active realization. Existing forms are reused;
adapter absence, optional-dependency absence, and adapter failure are structured
outcomes and leave active selection unchanged. Parquet supports flat scalar or
one-dimensional fixed-shape rows and requires the optional `parquet` extra.

`cached.tensorflow_view(...)` and `cached.torch_view(...)` return lightweight
iterables over NumPy or Parquet records. Their `support()` method reports a
missing optional framework without importing it, and the framework is imported
only when the view is iterated.

Exact resume is capability-based over the complete Dataset pipeline. Indexed
sources have durable row cursors, and stateful stages must checkpoint all state
(for example, shuffle RNG and buffer contents). Replay-only or unknown stages
may run but cannot claim exact continuation; an interrupted pending realization
then requires explicit rerun.

## Basic Pattern

```python
from dryml.artifacts import Artifact


class MyArtifact(Artifact):
    def compute(self):
        # Compute and persist payload here.
        return None
```

The artifact's definition identifies what the artifact represents. Managed
artifact payloads such as CachedDataset realizations are Store-owned records and
products rather than ordinary Object state.

## Repos And Locations

Artifacts use repo-backed locations. The artifact can ask the current or provided repo where its state should live.

This keeps artifact identity in the DRYML object graph while allowing payloads to be stored in the selected repo/store.

## Common Pitfalls

- Do not store large computed payloads in constructor arguments.
- Keep `compute()` deterministic when possible.
- Store enough definition metadata to know what the artifact represents.
- Keep computed values out of definition identity; managed values belong in
  records and products rather than ordinary Object state.
- Use `exists()` as a convenience check, not as a complete validation of correctness.

## Related Docs

- [Objects and Definitions](objects_and_defs.md)
- [Repos and Stores](repos.md)
- [Models API](models.md)
- [Data API](data.md)
