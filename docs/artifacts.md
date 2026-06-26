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

`CachedDataset` represents a dataset artifact. It can be used when a transformed or generated dataset should be materialized and reused rather than recomputed each time.

## Basic Pattern

```python
from dryml.artifacts import Artifact


class MyArtifact(Artifact):
    def compute(self):
        # Compute and persist payload here.
        return None
```

The artifact's definition identifies what the artifact represents. The saved state stores the computed payload.

## Repos And Locations

Artifacts use repo-backed locations. The artifact can ask the current or provided repo where its state should live.

This keeps artifact identity in the DRYML object graph while allowing payloads to be stored in the selected repo/store.

## Common Pitfalls

- Do not store large computed payloads in constructor arguments.
- Keep `compute()` deterministic when possible.
- Store enough definition metadata to know what the artifact represents.
- Store computed values as state, not identity.
- Use `exists()` as a convenience check, not as a complete validation of correctness.

## Related Docs

- [Objects and Definitions](objects_and_defs.md)
- [Repos and Stores](repos.md)
- [Models API](models.md)
- [Data API](data.md)
