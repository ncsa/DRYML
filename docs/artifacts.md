# Artifacts API

Status: current.

Artifacts are lightweight Object identities for computed results. Managed
Artifact bytes belong to Store-owned typed records and products, not to the
Artifact's object directory.

## Artifact Base Class

`Artifact` extends `Serializable` and provides a logical base for computed values.

Important methods:

- managed `compute()`: start, resume, reuse, or explicitly rerun a realization.
- `exists(repo=..., store=...)`: check for a completed active managed result.

`exists()` is a convenience over managed status, not a filesystem-presence test.

## Why Artifacts Exist

Definitions should describe identity and construction. Artifacts represent outputs.

Examples:

- scalar metrics
- aggregate metrics
- cached dataset materializations
- generated reports
- model evaluation outputs

## Scalar Artifacts

Scalar artifact types include:

- `Scalar`
- `ScalarAgg`
- `ScalarAvg`

`Scalar` is immediate definition data and has no second persistence lifecycle.
`ScalarAgg` and `ScalarAvg` preserve direct definition-only evaluation, but a
Store-backed `compute` publishes a managed scalar `DataRecord`; use `read(...)`
to read its active value. The draft `Accuracy` class is removed without an alias
or payload migration.

## Classification Metrics

Classification metrics consume logical outputs, not hydrated inputs:

```python
from dryml.metrics import CategoricalAccuracy, ConfusionMatrix

accuracy = CategoricalAccuracy(
    experiment.train.result,
    test_cache.compute.result,
    labels=(0, 1, 2),
)
confusion = ConfusionMatrix(
    experiment.train.result,
    test_cache.compute.result,
    labels=(0, 1, 2),
)

accuracy.compute(store=store)
confusion.compute(store=store)
print(accuracy.value(store=store))
print(confusion.matrix(store=store))
```

Computation resolves the model and cache as one exact stable input vector,
hydrates a fresh model only for execution, and iterates only the pinned completed
cache record. Missing or incomplete inputs fail without dependency computation.
Loading a definition or reading a completed result does not materialize either
input or import TensorFlow, Torch, or PyArrow.

Sparse integer or string labels and unambiguous one-hot labels/predictions are
supported. Sparse values may be scalars or have one singleton trailing
dimension; floating singleton values remain score-like and are not treated as
sparse labels. Labels must be declared in stable order. For confusion matrices,
rows are true labels and columns are predicted labels in that order. Empty input,
unknown labels, out-of-range vector widths, malformed one-hot values, and tied
predictions fail without publishing or replacing an active result. Changed
model/cache activation makes a completed metric stale until
`metric.compute.rerun(...)` is explicit.

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
artifact payloads are Store-owned records and products rather than ordinary
Object state. See [ADR 0009](adr/0009-managed-operation-lifecycle.md).

## Repos And Stores

Managed calls accept an explicit Repo or Store and otherwise use one unambiguous
active default. Definitions do not retain a Store. Multiple competing Store
results fail explicitly instead of using Repo order.

## Common Pitfalls

- Do not store large computed payloads in constructor arguments.
- Keep `compute()` deterministic when possible.
- Store enough definition metadata to know what the artifact represents.
- Keep computed values out of definition identity; managed values belong in
  records and products rather than ordinary Object state.
- Use `exists()` as a convenience check; result readers still verify records,
  representation metadata, manifests, and bytes.

## Related Docs

- [Objects and Definitions](objects_and_defs.md)
- [Repos and Stores](repos.md)
- [Models API](models.md)
- [Data API](data.md)
