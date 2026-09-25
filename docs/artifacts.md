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

`dryml.artifacts.mean(src, mode="global" | "coordinate")` and
`dryml.artifacts.quantile(src, q, mode=..., capacity=..., seed=0)` return inert
Fold Artifacts with all initializer, transition, and finalizer Methods already
declared. They accept no update/finalizer injection. Global mode reduces every
scalar coordinate in logical observation order; coordinate mode excludes the
declared batch axis and keeps one state per remaining coordinate. Uneven batches
therefore preserve the same population and weighting as unbatched input.

Mean stores native float64 sums and int64 counts and accepts signed integer,
float32/64, and boolean observations. Quantile stores at most `capacity * C`
float64 values, where `C` is one globally or the coordinate count, plus bounded
metadata and working-batch storage. It accepts signed integer and float32/64
observations, uses a private seeded Threefry/Algorithm R sampler, and is exact
with linear interpolation while the population fits capacity. After overflow it
is a deterministic sample estimate: endpoint results are reservoir extrema, not
guaranteed stream extrema, and no worst-case rank/tail bound is claimed. Fixed
seed, backend, version, CPU device, and logical item order reproduce membership;
partitioning into uneven batches does not change that order. Quantile rejects
boolean, non-finite, empty, unsigned, complex, object, sparse, ragged,
mixed-backend, shape-changing, invalid q/capacity/seed inputs before publication.
Both factories normalize `src` at the existing `Ref[AutoRef]` function boundary:
construction retains an inert concrete source reference, does not save or load
the source, and rejects a raw soft `Definition`. Count, population, or draw
overflow, non-finite native intermediate arithmetic, and changed carry
shape/dtype fail before a successor state or terminal payload is installed.

## Evaluation Metrics

`dryml.metrics.regressor_mae(test_ds, model, mode=..., x="x", y="y")` and
`regressor_mse(...)` declare inert Fold evaluations. They assemble a `Map`,
`Project`, `Select`, `Pipe`, `Diff`, and `Abs` or `Squared` graph as a complete
concrete definition before supplying it to Fold. No Dataset is traversed, model
is loaded, state is selected, or input is saved during factory construction.
Their mean denominator is the selected global or coordinate population, so an
uneven final batch has the same meaning as individual observations.

`classifier_confusion_matrix`, `classifier_accuracy`, and `classifier_f1`
likewise accept non-materializing Dataset/model references and require explicit
prediction and target label Methods. They never decode logits, probabilities,
or one-hot values implicitly. A caller can use `ArgMax` or another declared
Method in the supplied conversion graph when that conversion is intended.
Incoming `StateRef` inputs remain exact nested references in that declared graph:
a later model save does not replace the selected snapshot. Factory construction
never saves an unpersisted source or model. A completed metric Fold can restore
its result from its own StateRef without source/model payloads or managed control
records; a later rerun still requires the retained input references and fails
when that authority is unavailable.

`ConfusionCounts` uses fixed ordered, unique, homogeneous exact `int` or `str`
classes. It consumes scalar labels or matching non-empty one-dimensional label
batches, uses truth rows and prediction columns, and keeps native int64 carry
on NumPy, Torch CPU, and TensorFlow CPU. NumPy accepts string and integer
labels; Torch and TensorFlow accept integer labels. Unknown labels, shape or
backend mismatch, and int64 count overflow raise before a successor count
matrix is returned.

`AccuracyFromConfusion` and `F1FromConfusion` accept only non-empty square,
nonnegative signed-integer matrices. Accuracy and weighted F1 return zero for a
zero-total matrix. Undefined per-class F1 terms are zero; macro includes those
zeros, weighted uses truth support, micro aggregates counts, and `average="none"`
preserves class order. Binary F1 requires exactly two classes and an explicit
valid `positive_index`; that argument is invalid for every other averaging mode.
To produce confusion, accuracy, and F1 from one traversal, declare one Fold with
an `AccumulatorGroup`, matching grouped initializer, and a `Project` finalizer.
Calling the convenience factories independently intentionally creates independent
evaluation streams.
