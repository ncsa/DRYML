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

`Fold[T]` is a concrete non-resumable `Value[T]` for a declared streaming
`Dataset` reduction. Its constructor accepts a non-materializing `Ref[AutoRef]`
source plus an initializer `Method`, an `Accumulator`, and an optional finalizer
`Method`; every dependency is retained before computation. Constructing or
persisting a Fold does not load its source, allocate carry, or invoke/select a
Method. `compute()` loads the source only through the managed invocation's
selected Repo, uses one iterator, initializes from its first real observation,
then transitions that same observation exactly once and closes the iterator on
every exit. Source specs must declare uniform element or batched meaning; empty
sources and zero-sized observations are errors. Fold validates each actual
observation against that declared spec, uses the first item to resolve dynamic
coordinate facts while retaining a dynamic batch axis, then holds coordinate
shape, backend, and device stable for the invocation.

Each Fold invocation owns fresh carry and is not resumable. The accumulator
receives observation and carry as separate Method inputs, and selected carriers
validate initializer, transition, and finalizer contracts. Fold keeps working
state and iterator resources invocation-local, so sharing declared Methods or
running independent Folds does not share carry. Complete terminal results are
normalized and installed atomically. NumPy, Torch CPU, and TensorFlow CPU
results stay native throughout initialization, transitions, and finalization;
only the completed terminal result is converted to a host float or NumPy array
tree for persistence. A failure or interruption before that point
leaves the old (or absent) result unchanged; an explicit managed rerun starts a
new iterator and carry. A later managed state/control-publication failure still
raises honestly even though the complete live result remains ready.
