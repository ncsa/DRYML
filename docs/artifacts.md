# Artifacts

`Artifact` is an abstract `Serializable` computed output. Concrete subclasses
must implement a `@managed_operation()` `compute(..., *, managed)` method and a
boolean `ready` property. `ready` describes whether the subclass's current
content is usable; it must not compute work or materialize input references.
Artifact does not discover results, rerun work, cache inputs, or impose a common
payload layout.

`Value[T]` is an abstract Artifact for one completed result. Its `value()` method
returns `T`, including a deliberately stored `None`, only when `ready` and the
result envelope both report usable content. Otherwise it raises
`ArtifactNotReadyError` without computing or loading inputs. There is no public
value setter. Concrete authors install newly computed values through the
protected `_install_value_payload(...)` boundary and may implement the protected,
side-effect-free `_validate_value_result(result)` hook to enforce their domain.

Value persists only a versioned result envelope through `Serializable` hooks;
ordinary subclass fields, input graphs, iterators, accumulator state, and managed
control records are not captured. Subclasses that need additional durable state
must use distinct hook filenames and validate their own files. A saved Artifact
returns a StateRef, and `Repo.load_state_ref()` restores exact Value results
without recomputation or input materialization. Invalid result payloads fail
restoration rather than becoming ready values; existing core restoration
invalidation handles failed in-place restores.

`CachedDataset` is an intentionally abstract Stage 4 placeholder. The retired
mutable Store-root cache implementation has no compatibility surface.
