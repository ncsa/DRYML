# Data API

Status: draft.

The DRYML Data API provides reusable, repo-backed dataset objects and dataset transformations. Datasets are normal DRYML objects, so they can be saved, queried, composed, and used as part of larger object graphs.

## Dataset Contract

`Dataset` is an abstract iterable dataset type. Every concrete Dataset subclass
must implement `__iter__`; legacy `__len__` remains optional because cardinality
can be unknown. The supported source, mapped, and structural dataset classes
implement iteration and remain constructible.

Important expectations:

- A dataset should be re-iterable.
- `iter(dataset)` should produce a fresh iterator.
- `dataset.spec` describes one yielded element.
- `yield_cardinality()` returns the declared number of yielded values as a
  finite, unknown, or infinite `Cardinality`. It normalizes legacy integer and
  `Cardinality` `__len__` declarations without opening a cursor.
- `example_cardinality()` returns an example total when declared TensorSpec
  batch metadata or an explicit source total proves it. Unbatched values have
  one example per yield; fixed uniform batches multiply yield cardinality;
  dynamic batches remain unknown unless an explicit total is available.
  A zero-yield dataset has zero examples regardless of batch metadata.
- `examples_in(value)` validates one runtime TensorSpec tree and returns its
  exact positive example count. Batched leaves must expose non-rank-zero,
  nonempty matching leading dimensions and honor fixed batch declarations;
  coherent unbatched trees count as one example.
- Dataset composition propagates an example total only when it can prove one:
  `Batch` retains source examples (and removes a dropped remainder), `Unbatch`
  turns proved batched examples into yields, and `Take`/`Skip` use declared yield
  ranges. `Shuffle` retains only full-membership totals, `Zip` requires aligned
  branch yield and example totals, and `Chain` requires compatible batch
  semantics. `Map` is unknown by default; pass `preserves_examples=True` only
  when every result preserves its input count, which is checked at iteration.
- `peek()` returns one element without permanently consuming the dataset.

These count methods do not open, peek, scan, or otherwise consume a Dataset,
and they import no optional tensor backend. Generic values without a complete
TensorSpec tree retain normal iteration behavior but have unknown metadata
example cardinality and cannot be runtime-counted. Yield positions, cursor
positions, and `Take(source, n)` remain yield-based: `Take` still requests
exactly `n` source yields and reports exhaustion rather than silently clipping
to a short source declaration.

`dryml.artifacts.CachedDataset` implements this same contract after completion.
Its persisted output spec is the codec's actual NumPy-backed `SpecTree`, so
`Map`, `Batch`, `Unbatch`, cursors, and other ordinary Dataset consumers do not
branch on cache type or codec. A new or progress-only cache has no consumable
spec; only a completed StateRef restores an iterable Dataset.

### Cursors and Exact Bounds

`dataset.iterator()` creates an independent closeable `DatasetCursor`. Its
`position` counts consumed yields, `skip(n)` requires a nonnegative exact
integer and either advances exactly `n` values or raises `DatasetExhaustedError`
with requested and actual counts, and `close()` releases its owned iterator.
Dataset objects do not retain a shared cursor. Array and NPY-file sources use
equivalent private indexed advancement, so skipping does not read discarded rows
or files.

Use `with dataset.iterator() as cursor:` around consumer work. All acquired
resources get one reverse-order close attempt on EOF, failure, interruption, or
context exit; a source/operator/consumer error remains primary when cleanup also
fails. Explicit closure stops later reads but does not prove completion.
`cursor.observed_eof` is positive root-output termination evidence: only a
designated iterator pull ending or a successfully satisfied output bound can
establish it. Strict skip/Take/native count failures never establish it. A skip
shortfall retains its exact counts and remains primary over cleanup failures;
invalid counts leave the cursor untouched, and zero is a no-op even after closure.
Genuine EOF is recorded before cleanup and survives cleanup failure. Unexpected
`StopIteration` from acquisition, Method execution, validation, or cleanup is
reported as `RuntimeError` with the original exception as cause.

`GeneratorDataset` factory acquisition remains lazy, including epoch cursors.
Closing an unread cursor opens nothing. Once opened, the cursor retains and
closes the actual factory iterator, including at a finite declared cutoff;
ordinary structural/combine operators explicitly own their child traversals.

`Take(source, n)` requires a nonnegative exact integer and has finite
cardinality `n`. It yields exactly `n` values or raises `DatasetExhaustedError`
after its available prefix; `Take(source, 0)` does not open the source. The
older `Skip(source, n)` remains forgiving when a source ends before its prefix.
For declared seed-aware `GeneratorDataset` sources, `Take` also carries an
optional logical `epoch`; `Repeat(Take(...))` advances selected epochs by default
without changing its strict yield count. `fixed_prefix=True` intentionally reuses
one epoch. Opaque generator factories receive no new replay or seed requirement.
A completed `CachedDataset` instead persists one selected realization: repeating
that cache replays its stored values and does not advance the source's epoch seed.

### Detached traversal continuation

`Dataset.open(*, epoch=None, start=None, state=None, acceptance="delivery")`
and `PreparedDataset.open(...)` can resume an explicitly captured `DatasetState`.
The record is an immutable ordinary value, **outside the Object graph**: it has no
ObjectId, Dataset property, LocalState, Repo, producer, frame, invoker or acceptance
token. `state.cdef` describes the frozen prepared recipe; `state.state_ref` is
optional selected-state metadata, not a restoration command or retention grant.
`state.position`, `state.replay_recovery_policy` and
`state.replay_recovery_enabled` inspect saved facts without source access.

```python
from dryml.data import ArrayDataset, DatasetState, Shuffle
import numpy as np

dataset = Shuffle(ArrayDataset(np.arange(100)), buffer_size=8, seed=17)
with dataset.open() as traversal:
    view = traversal.next_epoch()
    first = next(view)
    state = traversal.capture_state()  # explicit detachment, not a per-yield cost
    state.save("input-cut")           # destination must be a new directory

loaded = DatasetState.load("input-cut")
loaded.validate(dataset)              # metadata compatibility only
with dataset.open(state=loaded) as resumed:
    remaining = list(resumed.next_epoch())
```

The supplied target may be independently materialized but must match the complete
configuration, occurrence/sharing partition, source selections and required state,
normalized specs, operator controls, and protocols. Preparation freezes those
facts; later changes to mutable Dataset controls do not change the prepared record
binding. A newly prepared edited target is incompatible. Each open revalidates;
past validation is not authorization. With a record, omitted epoch/start use its
saved position; explicitly supplied values must equal it. Two opens create
independent live state and cannot mutate the record or each other.

`load(path)` reads only that directory's metadata and owned bytes. `validate(target)`
may prepare source-inert selections and inspect supplied array contents, but neither
API instantiates a Dataset, calls a producer/factory/state hook, restores a referenced
Object, opens referenced Stores, or imports optional tensor backends to rebuild
values. Native buffers decode to inert host evidence and reconstruct lazily at
explicit reattachment. `open(state=...)` selects indexed/direct/replay restoration
per occurrence; producers remain lazy until ordinary demand.

#### Recovery strategies and replay policy

* **ArrayDataset:** indexed restoration fetches no discarded rows. Prepare/open do
  not copy the full source. Capture owns its required source bytes, so later
  producer/consumer mutation cannot alter a saved record. Keep borrowed array
  contents stable during capture and traversal; concurrent unversioned mutation is
  unsupported. Readonly views alone do not establish stable ownership. There is no
  per-yield hash or mutation monitor.
* **Ordinary GeneratorDataset:** call `traversal.enable_replay_recovery()` or
  `enable_replay_recovery_if_needed()` to trust deterministic replay. Restoration
  recreates original initialization with the actual effective seed and physical
  pass, consumes exactly the saved fetched-source prefix once, and restores
  downstream buffers/phases independently. A factory may return an inner generator
  function through the ordinary acquisition path. No suspended frame is serialized.
* **Direct producer:** subclass the public `RecoverableGenerator` abstract iterator
  and implement `__next__`, `capture_state() -> ContinuationValue`, and
  `restore_state(state: ContinuationValue) -> None`. The inherited `__iter__` returns
  self. State must include the complete next-yield cursor/RNG/buffer state. Capture
  invokes hooks only on request; restore generates zero historical yields. Hook
  exceptions propagate. Optional `close()` is called once through cursor ownership.
* **Other sources/operators:** ordinary live iteration remains available; unsupported
  exact capture fails explicitly. NPY-file, custom/native-source and dynamically
  selected Map execution are not certified by this continuation format.

The default traversal-local policy is `"disabled"`. Manual enable selects
`"enabled"` and is not weakened by the conditional helper. Manual disable selects
`"disabled"` and cancels future conditional fallback; direct/indexed capability
and already captured records remain valid. Re-enable uses all retained source
positions, including work done while disabled. Toggling never consumes/replays or
reseeds a source, changes the Dataset/preparation, or modifies another traversal.
Small configuration is frozen once; large unversioned mutable factory arguments
are identified by bounded structural/byte preflight (256 small entries and at most
2 KiB per eligible NumPy configuration leaf, with aggregate array bytes charged to
the structural budget). They
are borrowed for live iteration and cannot retroactively gain original replay
evidence. Saved constructor-declared source dependencies can instead carry exact
SavedNode selections and use the caller's normally materialized target values.
The record neither embeds live Objects nor implicitly captures their state.
Stateful Method dependencies likewise need an exact saved target or Method state
selection; a constructor recipe alone is insufficient. Keep those selected live
source values unchanged during the traversal. Data does not infer current mutable
payload contents from an old save receipt or serialize a live Model implicitly.

The conditional helper is source-inert/idempotent: generator-free or declared
direct-only trees are no-ops; arbitrary factory products are classified only at
normal acquisition, including future Chain/Repeat/pass occurrences. Direct
capability requires the **actual consumed iterator** to satisfy the ABC, iterator
identity, and callable state methods; duck typing or hooks on an unrelated iterable
wrapper do not qualify. A failed direct restore never falls back to replay.
`replay_recovery_enabled` is true for explicit enablement or an active conditional
replay occurrence; an unresolved conditional decision alone is false. An initial
conditional record can be captured/saved/closed without calling any factory.
Disabling after opening a record does not cancel its already selected one-time
source reconstruction.

For mixed Zip trees, indexed, replay and direct branches keep separate fetched
counts and strategy records, including over-pulls, Shuffle buffers and unequal
section tails. Shared Dataset slots are independent execution occurrences. Zip
output offsets are never used to replay the whole tree or duplicate downstream
transitions. Unknown implicit EOF retains prior physical-pass nonempty evidence;
observing an empty resumed suffix normalizes its boundary once, with the existing
handoff holding next-pass acquisition until released.

Notebook functions/classes need retrievable source and supported stable imports.
A notebook-defined RecoverableGenerator class can be captured from retained cell
source using its own method filenames/line anchors, reconstructed without the live
binding in a clean process, and restored directly. Missing/conflicting/ambiguous
source rejects; a changed captured body is incompatible even under the same name.
There is no name-only/pickle fallback, IPython requirement, hot reload, or live-class
migration contract. Target reconstruction is caller-controlled, separate from inert
DatasetState reading.

#### Capture cuts, values, and publication

Capture and policy mutation require a live owner-thread quiescent cut. An outstanding
public candidate or in-flight pull raises `TraversalOwnershipError`. Staged but
unexposed lookahead is represented without accepting it. Raw/converted candidate
aliases still share one single-use authority. After closure/rejection, capture can
return only an exactly matching prior record (position, policy and boundary facts);
otherwise it fails without reopening a source. Dynamic unsupported buffers fail
capture without rolling back accepted progress or damaging earlier records.

`ContinuationValue` admits Python None/bool/bounded int/float/complex/string/bytes,
list/tuple/string-keyed dict trees, supported fixed-storage nonobject unstructured
NumPy arrays/scalars and registered dense native tensor buffers (Torch, TensorFlow,
JAX). Repeated object aliases within the supported tree are preserved; arbitrary
overlapping strided-view storage aliases are not a portable contract. PCG64's
128-bit state is lossless. Live Objects, frames, generators, callables, resources,
object/structured/metadata-bearing dtypes and opaque values are rejected in buffers.
Import-bearing enum/type or extension-dtype constructor metadata is not qualified
for inert continuation IO either; ordinary live execution remains available.
Writable values retained by buffering operators are owned at their algorithmic
lifetime boundary; immutable/native-functional buffers avoid indiscriminate copies.

`DatasetState.save(path) -> Path` publishes a new v1 directory, with no overwrite
mode. A stable sibling name advisory lock covers the authoritative final absence
check and same-filesystem staged rename. Concurrent cooperating thread/process
writers produce one complete winner and one `FileExistsError`; readers see no
destination or a complete inventoried record. Permission, short-write, inventory,
interruption and cleanup failures preserve prior destinations and primary errors.
An interruption around rename may leave a complete committed record; inspect that
destination before retrying. Supported local locking/atomic-rename semantics are
required, not distributed/network-filesystem coordination.

`ContinuationFormatError` rejects malformed/incomplete/corrupt/unsupported records;
`ContinuationMismatchError` rejects a different supplied target;
`UnsupportedContinuationError` rejects unqualified capture. Filesystem, source,
factory, hook and cleanup failures retain their causal exceptions where possible.
Standalone saving never publishes a DRYML Object, managed checkpoint, or retention
root. Keep required external source states under an existing saved owner or explicit
Repo retention. Reclaimed state must be rematerialized through normal Core rules
and fails when unavailable; metadata in this directory cannot resurrect it.

Cost boundaries are distinct: ordinary delivery maintains counters/local buffers;
capture detaches/validates required source and pending bytes (possibly synchronizing
native leaves); save encodes/publishes files; restore loads chosen snapshots,
seeks indexed positions, installs direct state, or replays only recorded ordinary
source prefixes. No capture tree, direct state hook, encoding, source-sized clone,
or recovery-only device barrier runs on normal yields, with recovery enabled or
disabled. Consumer/trainer checkpoint association is a separate integration layer.

### Dynamic Dense Elements

Dense Dataset values may vary in sample dimensions as well as batch size. For
example, `TensorSpec("float32", shape=(Dynamic, 5), batch=Dynamic, backend="numpy")`
admits consecutive tensors shaped `(2, 3, 5)`, `(1, 7, 5)` and `(3, 2, 5)`.
Map, composed Methods, explicit-output Models and prepared stream graph ports
retain that contract. Counts still use only the leading batch length, so changing
a sample length does not change the number of examples represented by a batch.

Batch can stack equally shaped samples in one group while later groups have
different shapes. It cannot stack unequal sample shapes within the same dense
batch: ordinary and prepared execution propagate the collation failure. Unbatch
preserves the declared Dynamic sample dimensions and splits actual batch lengths.
Zero-sized sample axes are distinct from empty batches; Dataset runtime example
counting still rejects a zero-example batch.

GeneratorDataset can infer varying dimensions from a SpecHint while retaining an
explicit example_count. Sampling closes its cursor on success or failure; observed
example totals are checked on subsequent normal traversals once the spec exists.

### Ragged Elements

`Batch(source, size, ragged=True)` packs variable dense sample axes using NumPy
row partitions or native TensorFlow ragged tensors. Fixed-shape labels remain
dense and opaque leaves retain list batching. Existing ragged samples gain an
outer partition with ordinary Batch as well. Unbatch removes one partition,
preserving empty rows and returning dense samples when the ragged rank reaches
zero. Ordinary and prepared graphs share the same collation and split contracts.

Cast, scalar Scale, composition, and authored Models can preserve ragged values or
produce dense outputs. Example counts use the outer population, independently of
row lengths. See [Ragged Tensors](ragged.md) for complete examples and contracts.

### Padding And Masks

`Padding(length, return_mask=True)` returns `{"values": dense_tensor,
"mask": bool_tensor}` from one geometry decision. Sample axes exclude batch;
mask axes stop before unchanged trailing features. Fixed targets or per-input
maxima, left/right placement and explicit truncation work in ordinary/prepared
pipelines. Real values equal to the fill value remain valid. NumPy/TensorFlow
accept ragged input; all four tensor backends provide dense padding. See
[Padding And Validity Masks](padding.md) for full contracts and examples.

### Opaque Python Elements

For a fixed numerical array such as sampled probe coordinates, the
[`NumpyArray` Serializable container](numpy.md) keeps the bytes in saved state
instead of constructor metadata, with small searchable `shape`/`dtype` arguments
and zero initialization. It is a payload container, not a Dataset; retain its
SavedNode as a dependency of the Dataset that evaluates or traverses it.

`GeneratorDataset` can declare raw Python values without describing their class
or content:

```python
from dryml.core import TensorSpec
from dryml.data import GeneratorDataset, Batch

objects = GeneratorDataset(make_meshes, spec=TensorSpec("object", layout="python"))
batches = Batch(objects, 32)
```

Here `make_meshes` is the user's factory. No factory runs during declaration or
planning. Each opaque source yield counts as one example, even when the yielded
object itself is a list or dict. `Batch` keeps the original references in an
outer Python list, and `Unbatch` splits only that declared outer batch. A source
already yielding list/tuple batches declares `batch=Dynamic` or a fixed batch
size on the same object spec. Mixed object/tensor spec trees preserve their
declared structure and collate/split each leaf by its own spec.

Use `Map(objects, conversion_method)` to produce fixed-shape numerical tensors;
the user-owned Method interprets each object's content and declares its numerical
output spec. Known-spec Method validation and ordinary/prepared Dataset cursors
do not inspect opaque values' classes, attributes, or array representations.
Object declarations do not choose a serialization codec for raw object content.

Object values can remain opaque across multiple Map stages and only become
tensors when a user Method produces them. Tensor-to-object outputs and mixed
object/tensor intermediate trees use the same specs in ordinary and prepared
graphs; Project and Select preserve the declared boundaries.

Alternatively, `spec=SpecHint(samples=2)` asks `GeneratorDataset` to sample two
values during construction and infer unknown Python leaves as opaque specs.
`SpecHint(batch_mode="batched", samples=2)` infers nonempty list/tuple batches of
opaque leaves and merges different lengths to `Dynamic`. Hints are normalized
to ordinary definition metadata. Use an explicit spec for object-valued
containers, empty batches, or construction without factory execution.

### Local Traversals And Epochs

`Dataset.open(*, epoch=None, start=None, acceptance="delivery")` prepares once
and returns a lazy `DatasetTraversal`. `PreparedDataset.open(...)` reuses its
frozen controls, topology, source bindings and selected invokers. Each traversal
owns independent live cursors, RNGs and buffers; multiple traversals may interleave.
Reading an unread traversal's `position` or closing it opens no producer.

```python
from dryml.data import Epoch, Batch

prepared = Epoch(Batch(source, 32), len=100).prepare()
with prepared.open() as traversal:
    for epoch, elements in traversal.epochs(stop=3):
        for batch in elements:
            consume(batch)
```

`Epoch(src, *, len)` requires a positive **exact int** and sections incoming
yields without reopening or reseeding upstream. It retains a nonempty short final
section; an empty source creates no section. Adjacent views share one execution.
Without Epoch, each finite physical pass is one implicit epoch; the data layer
reopens it for the next pass. A genuinely empty pass terminates traversal. A
provably infinite boundary-free root rejects epoch-oriented opening before
acquisition; UNKNOWN is supported without a scan and does not promise termination.
Existing `iterator()`/`__iter__()` remain independent flattened physical-pass APIs.

`traversal.next_epoch()` returns a borrowed finite `EpochView`, or raises
`StopIteration` at terminal completion. `traversal.epochs(*, stop=None)` yields
`(ordinal, view)` pairs; stop is an exclusive logical epoch. One unfinished view
or active enumeration may borrow an individual traversal. Finish each view before
requesting another. The traversal, view and one-level cursor are context managers.
Closing an unfinished view aborts its owner; exhausted views remain exhausted and
closing them is a no-op. Closing never drains discarded tails. Source, operator,
consumer and cleanup failures follow the cursor primary-exception policy above.
Pulls and mutations are owner-thread-only; recursive advancement and competing
consumers raise `TraversalOwnershipError` (a `TraversalError`/`RuntimeError`).

`epoch` and `start` default to zero; start counts final outputs in the selected
epoch. Explicit positional seeking may consume earlier sections. An exact-end
start is valid: epoch enumeration normalizes past that completed section, while
`Dataset.iterate(*, epoch=None, start=None)` and `PreparedDataset.iterate(...)`
return an empty one-level cursor for it. Larger offsets raise
`DatasetExhaustedError(requested, yielded)`. `iterate()` owns a private traversal,
returns authored elements (including `(x, y)` pairs) unchanged, and its EOF proves
only selected-epoch completion. New epoch/start/stop/len counts reject bool, NumPy
integers, integer subclasses and floats with `TypeError`; negative values and
stop before the effective epoch raise `ValueError`. Stop equal to that epoch
acquires nothing; reaching stop leaves the traversal usable.

`TraversalPosition` reports accepted `epoch`, `start`, `sequence`,
`section_complete` and `root_complete`. A fixed final boundary normalizes on its
last delivery without another producer pull. Unknown natural EOF is observed
later; exact EOF never fabricates another explicit view. Implicit physical EOF
retains a boundary handoff before another pass is acquired; it is not terminal
traversal EOF. Nonempty-pass evidence includes values discarded during positional
seeking, independently of whether its selected suffix was empty.

For explicit acceptance use `open(acceptance="explicit")`, then `view.reserve()`
and `traversal.accept(candidate)` or `traversal.reject(candidate)`. At most one
`PendingElement` is outstanding. It carries a borrowed value, proposed position
and known-section-completion fact with an opaque single-use token, not a detached
continuation record. Acceptance performs no capture/serialization. A converted
value carrier may alias the same token; accepting/rejecting an alias consumes all
aliases. Stale/foreign tokens fail without progress. Rejection aborts speculative
execution while retaining the accepted position; tokens cannot be pickled.

Only one explicit section axis is supported. Nested Epoch, mixed boundary modes
in Chain/Zip, and unsupported custom section compositions reject during planning
before acquisition. Boundary-free custom operators retain ordinary eager fallback.

| Operator above Epoch | Section behavior |
|---|---|
| Map | One output per input; forwards boundaries |
| Batch | Flushes/drops short groups per section; never combines sections |
| Unbatch | Finishes the current input's splits before its boundary |
| Shuffle | Drains its section-local buffer; one continuing RNG stream |
| Take / Skip | Count a global prefix; Skip preserves crossed empty sections; Take remains strict and can terminate midsection |
| Repeat | Reopens only at child root completion; forwards and renumbers sections |
| Chain | Concatenates and renumbers sectioned children; skips empty child roots |
| Zip | Ends at the shortest aligned section, then discards unmatched tails before the next section |

Empty transformed sections remain visible. For example,
`Batch(Epoch(source, len=3), 4, drop_remainder=True)` has empty sections, rather
than false root EOF. `Epoch(Batch(source, 2), len=3)` groups six source elements
per full section; `Batch(Epoch(source, len=3), 2)` yields batches of two and one
within each three-input section.

`yield_cardinality()` accounts for per-section Batch remainders and Zip tail
discards. It returns UNKNOWN when declarations cannot prove exact membership.
`EpochView.cardinality` is exact from its opening offset or UNKNOWN;
`max_elements` is a separate optional upper bound. An upper bound is never used
as an exact-count shortfall check. These queries neither scan nor acquire sources.

### Prepared Stream Graphs

`dataset.method_graph()` returns an inert `MethodGraph` view for a Dataset
pipeline. It does not open, peek, select from, or otherwise consume a source.
Call `graph.learn(strategy="local")` before `graph.iterator()`; preparation
records qualified local selections from declared specs and opens no sources.
`graph.iterator()` then returns an independent closeable graph cursor. Optional
positional specs supplied to `learn` are assertions of the graph's declared
source specs, and `output_spec` is an assertion of its declared result spec.
Mismatched assertions, unsupported strategies, unqualified operators, and
Method-selection failures raise before source execution.

The prepared subset is `Map`, `Batch`, `Unbatch`, `Take`, `Skip`, `Repeat`,
`Shuffle`, `Epoch`, `Zip`, and `Chain`.
It preserves their ordinary order and cardinality rules: Map emits one result
per input, Batch retains short final batches unless `drop_remainder=True`,
Unbatch emits split items in order, Zip stops at the shortest source with its
existing left-to-right positional over-pull, and Chain opens/advances sources in
declaration order. Take preserves its strict count and logical seed coordinates.
Other Dataset operators keep their eager behavior but reject
graph planning explicitly.

Preparation publishes a complete frozen occurrence tree only after source
binding, inference, assertions, and selection succeed. It retains Batch size,
remainder/ragged policy, Take count/epoch/fixed-prefix controls, Skip/Repeat counts,
Shuffle buffer/seed, Epoch length, Map preservation
policy and Method selection, normalized specs/cardinalities, and the ordered
Zip/Chain shape (including nested dict/list/tuple containers). Editing the
original recipe later does not change existing prepared execution or its output
metadata. Fresh preparation observes recipe edits. Failed `graph.learn()`
replacement preserves the previous tree, selections, and existing cursors.

Concrete Array, Generator, NPY-file, TFDS and Torch sources supply Data-owned
acquisition adapters. Array tree structure, file ordering/loading policy,
generator factory/seed/cardinality declarations and small factory argument
containers are retained at preparation. Small plain dict/list/tuple argument
trees (up to 256 visited values per argument) are detached and reconstructed for
each cursor, preserving shared-container aliases. Container cycles in this
configuration raise `ValueError` before acquisition. Larger argument trees,
arrays, native objects and opaque values remain payload bindings by reference;
preparation never clones their full contents. Their existing external mutation,
ownership and availability requirements still apply. Freezing configuration is
not a snapshot of externally mutable payload bytes or of arbitrary Python code.
Custom source implementations retain their trusted compatibility opener and do
not gain a frozen-configuration or recovery guarantee.

One graph cursor owns all source and selected output iterator resources it
acquires. It closes them once in reverse acquisition order on exhaustion,
explicit close, acquisition/body failure, or a consumer body error. Reusing one
Dataset occurrence in branches opens independent cursors and independent
unknown-spec discovery buffers; it is not teeing, memoization, or deduplication.
`skip()` has the normal exact cursor contract, and reopening a graph creates a
fresh traversal rather than serializing an iterator or generator frame.

Prepared Dataset boundaries may retain one dense NumPy/TensorFlow/Torch/JAX handoff
edge when a downstream Method has no direct compatible implementation. The
adapter is applied while advancing the Dataset cursor, before native model
forward/loss/backward/tape or compiled work begins. Dataset preprocessing is
therefore data-only: it is not an end-to-end autodiff, tracing, or `tf.function`
contract. Batch and Unbatch select the existing declared-backend stack/split
operation once per prepared occurrence when available, reusing it across
independent cursors; their generic Python fallback remains equivalent
for order, short batches, cardinality, and cursor cleanup. Zip and Chain retain
the Python fallback unless compatible source-native composition is explicitly
provided by all inputs.

### Native Training Inputs

`dataset.prepare()` returns a dependency-light `PreparedDataset` for one native
training invocation. Construction plans the qualified Dataset/Method graph once
without opening a source. Its `execution_level` is `"stream"` when that qualified
plan runs and `"eager"` when an unqualified operator retains ordinary eager
Dataset iteration. Selection failures in qualified operators still fail during
preparation; eager fallback is only for unqualified Dataset operators.
The qualified plan's output `spec` is returned with detached container structure;
native cursor finite-count checks use the same preparation's captured cardinality,
not later Dataset edits. Existing independent flattened iteration remains lazy
and does not extract producer state, build continuation records, serialize values,
or scalarize a selected native/batched Method. Public continuation/session APIs
are not supplied by this preparation substrate.

`PreparedDataset.iterator()` opens an independent closeable cursor each time.
`training_batches(preparation)` applies one already selected x/y handoff per
authored batch and returns a closeable cursor. It preserves the x/y tree, dtype,
shape, batch axis, batch order, and short final batch. It checks finite declared
yield counts at exhaustion and raises `DatasetExhaustedError` for a short source.
Closing, conversion failure, source failure, or exhaustion closes the owned
cursor. Reads never advance accepted-update progress, exposure, or checkpoint
state; successful trainer updates remain their sole owners.
Explicit native training and validation loops use the same ownership context,
so update failures remain primary over source-close failures and accepted-prefix
accounting is retained.

Backend plugins deliberately remain separate from the generic seam:

- `dryml.tf.training_data.as_training_dataset(prepared, preparation)` produces
  a native `tf.data.Dataset` without a second batch, shuffle, or repeat. Each
  native traversal reopens a DRYML cursor.
- `dryml.torch.training_data.as_training_dataset(prepared, preparation)` returns
  a native `IterableDataset`. `as_data_loader(dataset)` is available only for
  wrappers that require a loader; it fixes `num_workers=0` and `batch_size=None`
  so it neither duplicates a source nor adds a hidden batch axis. Explicit
  loops may use `iter_training_batches` directly.
- `dryml.jax.training_data.iter_training_batches(prepared, preparation)` returns
  a closeable iterator of JAX-native batches. It does not introduce JAX training
  state or compilation.

Importing `dryml.data.native` or any training-data plugin imports neither its
heavy framework nor the unsupported historical `dryml.data.tf` and
`dryml.data.torch` wrappers. Framework imports occur only when a native adapter
is requested or a retained handoff executes.

## Source Datasets

Common source dataset classes:

- `GeneratorDataset`
- `ArrayDataset`
- `NpyFileDataset`
- `TFDSAdapter`
- `TorchDatasetAdapter`

`TFDSAdapter` loads a selected TFDS split and delivers either native TensorFlow
values or NumPy values. `data_dir` is an optional string local TFDS root; `None`
uses TFDS's normal default. `download` is an exact boolean and defaults to
`True` for compatibility, so ordinary use can cause TFDS download/network and
filesystem preparation side effects. Set `download=False` with a prepared
`data_dir` to require local-only loading: a missing split then fails through
TFDS rather than downloading. `download_config` accepts a TFDS `DownloadConfig`
object and is forwarded only when supplied, so callers can control TFDS
preparation behavior. Adapter construction imports TFDS and propagates its
import, split, local filesystem, and download failures; NumPy delivery avoids
importing DRYML's TensorFlow spec backend.

`GeneratorDataset` accepts an optional explicit `example_count` for opaque
or dynamically batched sources. A preserving `Map` retains this total without
opening the generator: for three yields with batch sizes `[2, 2, 1]`, a declared
`example_count=5` remains five examples after `Map(..., preserves_examples=True)`.
Ranges covering the whole source (including `Take(source, 3)` when the declared
yield cardinality is three, and `Skip(source, 0)`) retain the total as well.
Proper dynamic-batch subsets remain unknown because the total alone does not
locate examples within batches. Conflicting provable declarations still raise,
and iteration still validates observed source totals and preserving Map outputs.

It also supports a declared `seed_aware=True`
factory protocol: a base `seed` and versioned epoch derivation supply a deterministic
`seed` keyword to `iterator_for_epoch(epoch)`, allowing direct epoch reopening
without traversing earlier epochs.

The historical modules `dryml.data.tf.dataset` and
`dryml.data.torch.dataset` are unsupported legacy APIs. They are not current
exports and are not compatible with this Dataset contract.

Example:

```python
import numpy as np

from dryml.core import TensorSpec
from dryml.data import ArrayDataset

dataset = ArrayDataset(
    np.arange(12, dtype="float32").reshape(3, 4),
    spec=TensorSpec("float32", shape=(4,)),
)
```

## Transforming Data

`Map` applies one Method or a pipeline of Methods to each source element. Method
types and authoring helpers are owned by `dryml.methods`, not `dryml.code`.
`Map` resolves declaration metadata during construction through pure selection
and selected-output inference, retaining it in `_spec`. Its `spec` getter is
inert. Invalid inference still fails during construction; known missing,
conflicting, or ambiguous selection can now fail at this same boundary. Missing
Dataset specs raise without peeking. Construction runs no body, opens no source,
compiles no selected Method target, and activates no optional adapter. Ordinary
canonical Object construction still uses its own constructor signature boundary.

Each ordinary/graph cursor retains its own executable selection. When backend
alone is unknown, declaration metadata contains only partial inference facts,
and each occurrence may retain one first value to complete producer-input
backend facts before conversion planning. Nested Pipe successors receive the
completed selected outputs. Discovery never changes shared Dataset specs.
At this runtime-only first-value boundary, core discovery lazily activates a
missing native TensorFlow/Torch/JAX adapter before consulting its detector; a
framework plus its DRYML Model import is sufficient, without manually importing
the adapter plugin. Registered custom detectors remain in use, and nested value
trees still report mixed backends as a selection conflict. An unavailable native
dependency raises `ContextError` naming the backend. Discovery does not copy
values or relax selected-call placement, layout, or gradient validation.
Selections do not change the Method's eager, learning, or cached state.
Prepared graphs build output facts bottom-up; captured outputs drive successors,
example validation, and Batch/Unbatch dispatch without per-yield selection.
Native training still requires adequate declared producers before source
acquisition and does not use first-cursor discovery to repair its declarations.

When a selected Method explicitly declares `iteration_independent`, a Map cursor
may delegate discarded values to its source cursor without invoking the Method.
Otherwise it transforms the discarded prefix normally. The fast path may omit
data-dependent errors in discarded transformed values, but preserves source
counts, exhaustion, and validation for delivered values.

```python
from dryml.data import Map, Scale

scaled = Map(dataset, Scale(0.5))
```

Common transformation methods:

- `Pipe`: compose methods.
- `Project`: project nested structures.
- `Select`: select values by path.
- `Cast`: change dtype.
- `Flatten`: flatten tensor-like values.
- `Scale`: multiply/shift values.
- `ArgMax`: compute argmax along an axis.

## Numerical Reduction Methods

`Diff`, `Abs`, `Squared`, and `Equal` are reusable native numerical Methods.
`Diff`, `Abs`, and `Squared` promote signed `int8`/`int16`/`int32`/`int64` and
`float32`/`float64` inputs to float64 before arithmetic; they reject boolean
arithmetic, broadcasting, mixed backends, non-finite inputs, unsigned/complex,
object, sparse, and ragged values. `Equal` accepts supported numeric or boolean
equal-shaped values and returns a native boolean tensor.

`ArrayMean(axis=None)` and `ArrayQuantile(q, axis=None)` reduce one bounded
native array. Their axis declarations accept `None`, one integer, or a unique
tuple of integers. Mean accepts booleans and returns float64; quantile rejects
booleans, preserves tuple request order and duplicates, and uses native linear
interpolation. Empty selected populations, invalid axes, non-finite values, and
unsupported dtypes raise rather than changing population semantics.

`MeanInitial`, `MeanUpdate`, and `MeanFinalize` are the reusable declared
sum/count program behind the named mean factory. `ReservoirInitial`,
`ReservoirUpdate`, and `ReservoirQuantile` are the corresponding bounded
Algorithm R program. They keep carry tensors in NumPy, Torch CPU, or TensorFlow
CPU through initialization, transitions, and finalization. They do not collect
the Dataset or use a global random generator. Mean carry is a finite float64
sum with a nonnegative scalar int64 count; transitions reject changed carry
shape/dtype, non-finite intermediate sums, and count wrap before returning a
next carry. Reservoir carry is float64 storage plus scalar int64 population and
draw counters and a two-limb int64 key. Fill consumes no draw; each attempted
post-fill draw, including a rejection, advances the counter once. Invalid
population/draw metadata and int64 wrap fail before arithmetic or sampling.

The evaluation factories in `dryml.metrics` build their source projections from
these Data Methods: `Project(prediction=Pipe(Select(x), model),
target=Select(y))`, followed by an explicit prediction/target handoff to
independent host NumPy storage, then `Diff` and either `Abs` or `Squared` for
regression. Classification factories use the same host boundary and require
caller-supplied label Methods;
`ArgMax` remains an explicit label conversion rather than an implicit classifier
policy. This terminal evaluation policy may copy one concrete accelerator result
to the host; it does not make general mixed-backend Method inputs or accelerator
handoffs valid.

`Fold` retains its Dataset through `Ref[AutoRef]`. The declaration, its CDef, and
an exact completed Fold state retain the selected reference rather than an owned
Dataset payload. The Dataset is materialized only by `Fold.compute()` through its
selected managed Repo; a result-only `StateRef` remains readable after that input
is unavailable, while a later fresh rerun fails normally. During compatible
recovery Fold applies its saved processed-yield count to a new closeable cursor;
it never persists or transfers the prior iterator. Exact skipping detects a source
that now ends before the saved position. Positional continuation assumes a
re-iterable source but does not claim that a stochastic suffix equals the suffix
from the interrupted traversal. Trusted version-2 root-output EOF progress needs
no new source cursor. Cursor closure is not EOF: failed acquisition/read/Map,
consumer validation, interruption, and strict replay shortfall cannot authorize
exhausted Fold progress, even when cleanup also fails. Compatible version-1 active
progress still replays position; ambiguous version-1 exhausted progress requires
an explicit fresh managed rerun before carry/role/source effects. Legacy snapshots
are not rewritten or silently relabeled as trustworthy EOF.
CachedDataset applies the same positional meaning to retained cache progress: the
completed prefix remains exact, while a resumed stochastic source may provide a
fresh suffix after the saved yield count.

## Structural Operations

Structural dataset nodes change iteration structure rather than individual values.

- `Batch`
- `Unbatch`
- `Take`
- `Skip`
- `Shuffle`
- `Repeat`

For bounded custom synchronous behavior, subclass `StreamDataset` and declare a
class-level `dryml.methods.StreamNode`. The declaration names ordered
`IteratorPort` inputs/output, pure element-spec and cardinality transforms,
deterministic pull policy, and an exact maximum buffered-item count. Its
implementation receives borrowed input iterators and returns an iterator; a list
returned by an element `Map` Method remains one element and is not implicitly
expanded into stream outputs. Custom declarations are process-local trusted code,
not a generator serializer, async scheduler, key join, whole-stream collector,
or JIT interface.

Example:

```python
from dryml.data import Batch, Take

small_batches = Take(Batch(dataset, batch_size=32), 10)
```

## Combining Datasets

`Zip` combines datasets elementwise. `Chain` concatenates datasets sequentially.

```python
from dryml.data import Zip, Chain

pairs = Zip(features, labels)
combined = Chain(train_a, train_b)
```

## Working With `(x, y)` Data

Utility functions help with common supervised-learning structures:

- `iter_xy(dataset)`
- `collect_xy(dataset)`
- `collate_xy(dataset)`
- `materialize_supervised(dataset, *, strict_numpy=False)`
- `Collect`

`as_supervised(dataset, inputs, targets)` builds a persisted ordinary Dataset
projection yielding `(inputs, targets)`, rather than asking trainers to select
fields. A live Dataset returns a live `Map`. A soft Definition,
ConcreteDefinition, ObjectRef, StateRef, symbolic `Expr` (including `Par`), or explicit `Ref(...)`/`Mat(...)`
assertion returns an inert `Map` Definition without resolving or materializing
the source; an assertion contributes its retained target to that new materializing
Dataset graph. A scalar path selects one branch (and can retain a nested
dictionary); named/nested selections use dictionaries of
`Select.from_path(...)` leaves. Tuple/list selection trees require explicit
`Select.from_path` leaves so path and output-tree intent cannot be guessed.
`input_as_target=True` produces an autoencoder-style pair without copies or
implicit batching.

Dataset placeholders can be supplied before choosing the actual datasets:

```python
from dryml.core import Par
from dryml.data import as_supervised
from dryml.models import Experiment

experiment_template = Experiment.defn(
    model=model_template,
    train_fn=training_template,
    train_data=as_supervised(Par("train_ds"), "cart", input_as_target=True),
    test_data=as_supervised(Par("test_ds"), "cart", input_as_target=True),
)
experiment = experiment_template.sub(train_ds=train_ds, test_ds=test_ds)
```

This authors an inert graph; it does not inspect a placeholder's specification or
iterate data. Binding accepts the same source definitions/references as direct
authoring, including a concrete training definition and saved test `StateRef`.
The bound source must materialize a Dataset at the normal build boundary. Literal
selection errors still fail when `as_supervised` is called. The helper uses core
`authoring_helper` with an explicit dataset-source predicate and a shared live/inert
projection recipe; it does not opt into concrete signature normalization.

Native trainers consume this authored Dataset as-is. Put `Batch`, `Shuffle`,
`Take`, and related controls in the Dataset graph before constructing an
Experiment; trainers do not add hidden batching, selection, or shuffling.

These utilities assume an element structure where `x` and `y` can be selected by path.

### Strict NumPy Materialization

`materialize_supervised` collects one canonical `(inputs, targets)` Dataset in
authored order and returns `(inputs, targets, actual_examples)`. It flattens
authored batches, including a short final batch, and owns/closes its traversal.
The default remains permissive, including unknown-cardinality sources used by
sklearn. Source and consumer failures or interruption remain primary when cleanup
also fails; a cleanup-only failure propagates instead of returning success.

Opt in with `materialize_supervised(dataset, strict_numpy=True)` for finite,
runtime-validated dense CPU NumPy collection:

- Before opening any source, require finite **logical yield** cardinality and
  exactly two dense `TensorSpec` branches with `backend="numpy"`. Unknown or
  infinite yield counts require an authored finite bound such as `Take`.
- Features have sample shape `(width,)` and targets sample shape `()`. Both
  branches must be unbatched or coherently batched. Batched runtime features are
  matrices and targets are vectors. A fixed batch size is enforced; `Dynamic`
  permits differing positive sizes, including the final short batch.
- Each branch declares `int8/16/32/64`, `uint8/16/32/64`, `float32`, or `float64`.
  Boolean, object, string, complex, float16, sparse, ragged, Python-layout and
  foreign/unspecified backend declarations are rejected before acquisition.
- Every runtime value must be a pair tuple of dense NumPy ndarrays or NumPy
  scalars with the exact declared dtype, rank, shape and batch facts. Python
  scalars/lists, foreign arrays and masked arrays are rejected. Dense ndarray
  subclasses (such as memory-mapped arrays) are normalized to ordinary arrays
  only after their native facts have been validated, without changing dtype.
  Validated values are copied before advancing the source, so a producer reusing
  a mutable NumPy buffer cannot overwrite already collected yields.
  Feature width must be positive and consistent; a declared `Dynamic` width
  still cannot drift between yields. Validation precedes NumPy stacking, so
  mixed runtime dtypes cannot be silently promoted.
- Actual aligned nonempty rows determine the returned example count. Unknown
  example-count metadata is accepted when the logical yield bound is finite.
  Short logical traversal raises `DatasetExhaustedError` with requested and
  observed counts. The first excess logical yield raises `ValueError` immediately,
  before its value is validated or collated. Authored cursor EOF remains
  authoritative: the collector never reads underneath `Take` or a source cutoff.

Strict results are arrays shaped `(actual_examples, width)` and
`(actual_examples,)`, preserving their respective dtypes. Invalid declarations,
runtime structures, layouts, backends or dtypes raise `TypeError`; invalid counts,
shapes, batches, width or empty collections raise `ValueError`. Source acquisition,
read and cleanup errors propagate. `strict_numpy` must be an exact boolean.
The collector uses declared-backend NumPy collation without importing optional
tensor or model frameworks or adding conversion edges. Caller-authored NumPy
conversion in the Dataset graph is allowed. NaN/Inf handling and backend-specific
missing-value or numeric-conversion policies belong to the consuming trainer.

The maintained [CPU XGBoost](xgboost.md) and [OLS/binary Logit](statsmodels.md)
trainers opt into this strict collector. Their recipes and pure Method selection
remain usable without native packages; fitting requires the exact optional reader
stack. Neither trainer inserts a foreign-backend conversion. This does not change
the permissive default used by sklearn or require optional model dependencies for
ordinary Data operations.

## Specs And Data

Dataset specs are important because models and methods use them to infer outputs and verify structure. A dataset yielding `(x, y)` pairs should usually expose a matching spec tree.

```python
from dryml.core import TensorSpec

pair_spec = (
    TensorSpec("float32", shape=(128,)),
    TensorSpec("int64", shape=()),
)
```

## Common Pitfalls

- Dataset objects should be re-iterable unless clearly documented otherwise.
- Keep specs aligned with actual yielded values.
- Avoid embedding large data directly in definitions when a file-backed source is more appropriate.
- Use `Batch` and `Unbatch` consistently with tensor specs.

## Related Docs

- [Methods](methods.md)
- [Tensor Specs](tensor_specs.md)
- [Models API](models.md)
- [Repos and Stores](repos.md)
