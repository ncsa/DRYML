# ADR 0009: Managed Operation Lifecycle

Status: accepted.

## Context

Cached datasets, trained model state, and classification metrics need one
durable lifecycle without putting large or mutable results into Object
definitions. Analysts need callable `compute` and `train` methods, while Stores
must preserve exact lineage, prior readable results, resumable partial work, and
safe collaboration across ordinary process interruption and concurrent trusted
use.

## Decision

DRYML uses `dryml.managed` for the shared `compute` and `train` lifecycle.
Objects remain portable logical identities. A selected Store owns mutable
operation control and immutable completed realizations. Managed declarations
remain distinct from dispatch `OperationSpec` values, execution outcomes, typed
records, and physical representations.

## Identity Hierarchy

The stable identities are, from logical to physical:

1. An Object `ConcreteDefinition` identifies the producer recipe.
2. A managed declaration identifies a named method and deterministic output set.
3. `ManagedOutputRef` identifies producer definition, method, and output slot;
   it never embeds Store, realization, record, or representation IDs.
4. An operation namespace identifies producer CDef and method in one Store.
5. A declaration fingerprint identifies one compatible generation beneath that
   namespace.
6. A realization ID identifies one independent outcome across attempts and
   representations.
7. Typed record IDs identify immutable output, execution, and realization
   envelopes.
8. Representation IDs identify physical format contracts; adapter-derived forms
   retain the source realization ID and output slot.

A rerun creates a realization. An adapter creates another representation of the
same realization. These operations are never interchangeable.

## Authority Hierarchy

Authority is deliberately split:

- Object files and definitions remain authority for ordinary Object identity and
  ordinary saved state.
- The lifetime ownership lock, generation control, checkpoint head, and
  activation events are authority for the mutable managed lifecycle.
- Attempt workspaces and committed checkpoint manifests are authority for valid
  resumable partial bytes.
- Product manifests, typed output records, `ExecutionRecord`, and
  `RealizationRecord` are authority for completed bytes and exact lineage.
- The activation event is authoritative selection; the direct active pointer is
  a rebuildable acceleration path.
- Record-reference and SQLite query indexes are derived and rebuildable. They
  cannot override Object files, control state, records, or products.

Every managed read follows one chain and fails closed at a broken link:

`ManagedOutputRef -> selected Store -> generation/activation -> RealizationRecord -> typed output record -> representation -> verified product`

## State Machine

One current declaration generation has at most one pending realization.

- A first call starts work.
- A normal call resumes compatible pending work.
- A non-resumable pending result requires explicit rerun.
- Successful completion publishes all immutable records before activation.
- A normal call reuses an active result only when its exact consumed vector still
  matches a concurrency-stable current resolution.
- Changed inputs make the result stale; explicit rerun is required.
- Explicit rerun creates an independent realization and retains prior history.
- Failed or interrupted reruns do not replace the prior active result.
- Explicit activation may select a completed realization from the current
  compatible declaration generation.

Status, progress, checkpoint capability, and active selection are separate
facts. Incomplete output is never consumable as completed output.

## Fencing And Recovery

`DirStore` serializes one writer per producer/method namespace with a
lifetime-held OS lock and monotonic fence epoch. The coordinator is the only
writer of control, checkpoint heads, final records, and activation. Workers may
write only within fence-isolated attempt workspaces and return bounded intents.

Publication is pointer-last: verify output products, write typed output and
execution records, write `RealizationRecord`, mark completion, append the
activation event, then atomically replace the active pointer. A stale fence
cannot mutate or activate state. Exact publication of the immutable activation
event is the selection commit point. A failure before that event exists leaves
the prior active selection unchanged. Once the exact event exists, the
coordinator validates it and reconciles pointer publication under the same
fence; it does not report the realization as a failed rerun that a later pointer
rebuild would activate. Post-replace durability errors are retried
idempotently. Transient validation-read failures cannot reclassify an exact
durably written event or pointer as a failed rerun, while a successfully read
mismatch still fails closed. An activation whose commit status cannot be
established is reported as indeterminate and requires explicit reconciliation.
Recovery may adopt only immutable bytes justified by a durable
finalization intent and exact manifest validation; it never deletes partial or
authoritative data automatically.

## Store Capabilities

`DirStore` is the v1 live writable managed Store. It supplies control, locking,
fencing, durable products, records, activation, transfer, and cleanup.
`ZipStore` may read a verified transferred snapshot. `ZipExportStore` is an
export destination. Zip Stores cannot start, resume, rerun, lease, activate, or
clean live work. Multi-rank cooperative publication and network/distributed
filesystem coordination are not supported.

An operation resolves one explicit Store, one unambiguous bound Store, or one
default Store before work. Repo Store order is not a conflict-resolution rule.

## Schema And Compatibility

Managed control JSON, activation events, checkpoint descriptors, output records,
realization records, and representation specs are versioned machine-readable
contracts. Unknown schemas, malformed IDs, declaration-fingerprint mismatch,
partial ownership vectors, missing specs, or product-integrity mismatch fail
closed.

Existing Object/CDef, record, representation, and `OperationSpec` schemas remain
their own contracts. The unshipped draft Artifact directory-payload lifecycle is
not read or migrated. `Accuracy` is replaced by `CategoricalAccuracy`; no alias
or dual-read compatibility is provided. `Scalar` remains immediate definition
data, while computed scalar aggregates use managed `DataRecord` products.

Optional PyArrow, TensorFlow, and Torch modules remain behind representation or
backend execution boundaries. Loading managed definitions, status, caches, or
metric results does not import those frameworks.

## Exact Input Resolution

Before consumer execution, logical inputs are collected twice as one stable
vector containing declaration generation, activation generation, realization
ID, output slot, and record ID. Bounded instability fails explicitly rather than
mixing generations. Execution records the exact consumed vector. Missing or
incomplete dependencies fail before consumer mutation and are never computed
implicitly.

Experiment execution hydrates the selected `StoredStateRecord` into a fresh model
instance. Metrics likewise hydrate only during execution and iterate the exact
pinned cache `DataRecord`. Reading a completed metric loads neither input.

## Classification Metrics

`CategoricalAccuracy` and `ConfusionMatrix` are lightweight Artifact identities
whose `compute` method publishes a managed `DataRecord`. Both require stable
declared labels and accept sparse or one-hot categorical values only when shape
and winner are unambiguous. Unknown categories, out-of-range vector widths,
tied predictions, malformed one-hot labels, and empty input fail without
publication.

Confusion-matrix rows are true labels and columns are predicted labels in the
exact declared label order. Labels and orientation are included in the
representation identity and result document.

## Export Closure

Recipe export copies only the complete definition graph, including referenced
definitions, and creates no realization. Exact-result transfer recursively
copies producer definitions, selected and consumed realization records, typed
outputs, execution/adapter lineage required by the closure, representation and
operation specs, products, checkpoints referenced by completed realizations,
and sanitized compatible activation state. It never copies leases, owners,
fence epochs, live attempts, or uncommitted scratch.

Destination content is validated before activation and byte-identical content is
adopted idempotently. Existing destination selections are not overwritten.
Cleanup is separate, explicit, fenced, dry-runnable, and resumable; it refuses
active, leased, checkpoint-referenced, or externally consumed state.

## Consequences

Analysts use one lifecycle vocabulary across cache, train, scalar aggregation,
and metrics while collaborators can choose recipe recomputation or exact-result
transfer. The cost is additional Store-local control and immutable sidecars.
Retained history and partial work require explicit user cleanup. Managed workers
and trusted serialized inputs remain correctness boundaries, not security
sandboxes for hostile code or data.
