# Stage 5+7 Verification

Status: persistent-fixture, local, worker-routing, cross-framework, and GPU
qualification harnesses are implemented.
Routine fake-matrix, Core subprocess transport, coordinator sentinel, authority,
and recovery-schema tests pass; all numerical qualification gates, including the
24-cell CPU matrix, TFDS TensorFlow-to-Torch delivery, step-64 Torch/W3
recovery, TensorFlow/W1 GPU, and Torch/W3 GPU gates remain unrun.

The Stage 5+7 release matrix is not part of routine test collection. Real W1-W3
training, TFDS access, subprocess/Ray execution, and GPU evidence require an
explicit caller-prepared fixture manifest and explicit pytest opt-in. A skipped
or unavailable case is `unrun`, never numerical qualification evidence.

## Persistent Fixtures

`tests/qualification/stage5_7_baseline.json` is the checked-in fixed baseline.
Generated TFDS directories and Store snapshots are deliberately not tracked.
The caller selects a persistent manifest path, fixture Store, and TFDS data root.
The local manifest binds the configuration digest, version evidence, exact W3
NumPy/Parquet `StateRef` records, and machine-readable MNIST builder/config/
version/content identity. Its portable case/evidence encoding deliberately omits
the local TFDS root; execution supplies that same resolved root and rejects an
absent or mismatched prepared builder without downloading. Every W3 matrix case
and repeat must use the recorded NumPy reference unchanged; Parquet is used only
to prove sample codec equivalence before the matrix.

Prepare through the explicit callable:

```python
from tests.qualification.stage5_7_fixtures import prepare_manifest

manifest = prepare_manifest(
    "/persistent/stage5_7/manifest.json",
    fixture_store="/persistent/stage5_7/store",
    tfds_data_dir="/persistent/tfds",
    build=prepare_my_tfds_and_w3_caches,
    environment={
        "python": "3.12.13", "dryml": "...", "pandas": "...", "pyarrow": "...",
        "tensorflow": "...", "tensorflow_datasets": "...", "torch": "...",
    },
    allow_download=True,
)
```

`prepare_manifest` requires installed `pyarrow>=25.0.1` before it creates a
directory, locks authority, or invokes the builder. It prepares MNIST only with
`allow_download=True`, and only in `tfds_data_dir`; the trusted
`build(store, baseline) -> FixtureReferences` callback returns distinct completed
exact W3 NumPy and Parquet cache references. `prepare_manifest` holds a native advisory lock scoped to manifest
and Store authority across preflight, building, validation, and publication. It
refuses existing manifests and nonempty Stores; it writes/fsyncs a private
temporary manifest and uses non-replacing publication. Omitting
`allow_download=True` reports `QualificationUnrun` and does not call the builder.

## Case Evidence

Each real case must load its manifest with the same selected Store and complete
environment map, construct a machine-readable `QualificationCase`, then preflight
the persisted codec states before work. A closed `QualificationEvidence` record
retains case/manifest identity, final Experiment StateRef, exact model/test
bindings, history StateRef/rows/Artifact receipt, Artifact metric and named
independent formula provenance, Python/pandas/framework/TFDS versions, seeds and
baseline digest, runtime/backend/device/worker/process identity, elapsed time,
peak RSS, and output size. Core worker facts are derived from the returned
`CoreExecutionFuture.snapshot()`: qualified backend identity, actual worker
PID/identity, admission/allocation, submission ID, and Core publication evidence.
They are not copied from a requested matrix label. A Core snapshot with no
allocation records `worker-process-no-session-allocation`, not managed resources.
Local and managed-local cells return a closed observation from the fresh harness
child: PID, semantic mode, session/runtime mode, and actual allocation. The
coordinator rejects a mismatch; managed-local establishes its one-CPU/no-GPU
session inside that child.

Torch/W3/subprocess recovery evidence is a closed exact-facts record. It retains
the step-64 model and optimizer state digests, separately recorded tensor
placements, optimizer iterations, TrainState epoch/next-batch/step/examples/loss
numerator/loss denominator, checkpoint digest, history occurrence, and durable
pending Artifact status captured live at the interruption boundary, then compared
with a fresh-Repo exact Store reload. State digests use a bounded canonical
traversal, not pickle: mappings sort tagged scalar keys, sequences retain order,
and finite tensor/array values retain dtype, shape, and CPU-contiguous bytes.
Unsupported/object-dtype/non-finite or oversized state fails recovery evidence;
device placement is checked separately from value identity.
The retained initial Artifact receiver plus the request-selected managed control
Store locates the completed result before row repair. A bounded test diagnostic sidecar may be
retained before interruption, but it is not success evidence. Ordered events must start with
`artifact_repaired`, then `optimizer_update:65`; booleans are not accepted as
recovery proof. Missing, extra, non-finite, negative, or inconsistent fields fail
validation.

`cpu_matrix(manifest)` emits exactly the fixed 24 NumPy-delivery requests: W1-W3,
TensorFlow/Torch, and local/managed-local/subprocess/existing-Ray execution. It
does not start a worker. Every W3 request contains the same exact manifest NumPy
reference. `supplemental_tfds_torch_case(manifest)` is a separately counted W1
TensorFlow-mode TFDS-to-Torch interoperability gate; it is not a twenty-fifth
matrix cell and does not change primary Torch delivery.

`test_stage5_7_accelerated.py` owns exactly three opt-in real gates: the
TensorFlow-mode TFDS-to-Torch W1 interoperability gate plus TensorFlow/W1 and
Torch/W3 GPU gates. `accelerated_cases(manifest)` emits the latter two separately
identified subprocess requests, each with one caller-selected GPU visibility
entry. Their case identities and `<case_id>--gpu-tf-w1` or
`<case_id>--gpu-torch-w3` paths cannot collide with local, CPU-matrix, or recovery
evidence. The TFDS supplemental W1 source remains in TensorFlow mode through
graph-visible `Select`, `ImageNormalize`, `Flatten`, `Cast`, and `Project` Methods
until the `TrainingPreparation`-retained `tf_to_torch` training boundary; callers
do not add NumPy, DLPack, or conversion-lambda glue. Its saved accuracy Artifact
uses the same W1 `>= 0.80` contract as the primary cases.

`tests/qualification/stage5_7_workers.py` builds one selected primary case into a closed
JSON worker request. The request carries only case/manifest identities, exact W3
references encoded by the case, selected authority paths, execution backend,
session-allocation facts, and optional fixed recovery control. It never carries a live
Repo, Store, model/tensor, Dataset cursor, or framework handle. The coordinator
preflights selected manifest/fixture/control authority before Core Execute
submission; workers reconstruct authority from the request. `local` and
`managed-local` execute their semantic local paths inline in a fresh
case-isolation child, with ordinary and one-CPU/no-GPU session allocation
respectively. Every managed operation, status lookup, and recovery lookup opens
and uses the request's explicit control Store rather than the output Store
default. Both retain the mandatory managed Experiment/Artifact lifecycle.
`subprocess` and `ray` requests explicitly carry
`worker-process-no-session-allocation`; this records session allocation only and
does not relax the mandatory managed Experiment/Artifact lifecycle. They use Core
Execute's SubProcess and existing-address Ray backends respectively. A missing Ray address makes only the selected six
Ray cells `QualificationUnrun`; the other 18 primary requests remain
constructible. The harness never starts or provisions Ray.

Before a real case creates its case child, spool, or submission, the coordinator
runs under the orchestrator/materialization restriction, loads and checks
manifest/environment identity, validates manifest-bound W3 StateRef closure,
receipt, definition, and snapshot metadata without restoring or iterating fixture
data, validates TFDS content bytes, opens the existing control Store, and rejects
any output/work/evidence/control/fixture/TFDS overlap. The coordinator runs under
`RuntimeMode.ORCHESTRATOR` with strict materialization scope and does not import
TF/Torch/TFDS or restore model/Dataset payloads. It validates every final
reference through exact StateRef closure, projection, definition, and selected
Store metadata; the final Experiment remains that metadata receipt because its
normal restore can materialize a model edge. The existing narrow result-inspection
scope first admits only ExperimentData-compatible history and Value-compatible
scalar result root definitions, then loads only those exact references. The worker
repeats the TFDS builder/payload validation and may materialize workload data, but
returns only provisional closed data. The coordinator attaches observed
backend/future/request facts, atomically publishes non-replacing final evidence,
fsyncs its parent directory, and reopens it for exact comparison. Worker
diagnostics are never final evidence.

Routine fake-executor tests enumerate all 24 unique W1/W2/W3 x TF/Torch x
local/managed-local/subprocess/Ray requests, inspect their transport, and prove
fresh-process reconstruction does not import TensorFlow, Torch, TFDS, or pandas.
The opt-in real CPU and recovery test entries are intentionally `UNRUN` until
release automation provides prepared U10 authority, isolated output/control roots,
and, for Ray cells, an existing same-host address. The fixed recovery request
interrupts Torch/W3 subprocess work after the step-64 Artifact result is durable
while its row remains pending/failed. It exact-loads that checkpoint before
resuming and requires row repair before the observed optimizer update 65, without
a restart, duplicate row/result, or update. No real worker result is recorded by
this change.

Acceptance gates are fixed: W1 accuracy >= 0.80, W2 normalized-pixel MSE <=
0.12, and W3 noisy-observation MSE <= 0.05. Artifact values must agree with
independent formula checks; histories must be nonempty and completed. The CPU
budget is 300 seconds and 8 GiB peak RSS per isolated case. RSS is each child's
absolute normalized `ru_maxrss`, including framework setup, not a delta from a
preloaded baseline. No real case was run
for this change.

The two GPU gates retain the same W1/W3 numerical gates, 300-second deadline,
8 GiB peak-RSS budget, and 4 GiB case-Store budget. Before a worker case path or
submission exists, a disposable process applies the caller-selected
`DRYML_STAGE5_7_GPU_DEVICE` visibility control and verifies exactly one usable
framework GPU. Missing hardware is `QualificationUnrun`; after launch, CPU,
mixed, unknown, or contradictory placement is a failure. Accepted accelerated
evidence records the selected visibility entry plus normalized `gpu:0` claim,
native model-parameter placement, actual training tensor placement, forward
execution result placement, runtime/backend/PID, and U11 worker allocation and
submission facts. Availability APIs alone cannot satisfy this evidence contract.
TensorFlow captures its actual Keras accounting `train_step` inputs, prediction,
and parameters through an unset-by-default process-local observation scope, not a
model-wrapper call. Torch formula validation prepares the exact-loaded final model
with its public runtime device API before inference and rejects a non-GPU result.
Cross-framework GPU transfer and differentiable mixed-framework model composition
remain explicitly unsupported; same-framework native composition retains normal
gradients.

## Opt-In Runs

Use targeted test paths and add `--stage5-7-qualification` only after preparing
the persistent fixtures. The local W1/W2/W3 entry point requires both paths and
does not create authority itself:

```bash
DRYML_STAGE5_7_MANIFEST=/persistent/stage5_7/manifest.json \
DRYML_STAGE5_7_FIXTURE_STORE=/persistent/stage5_7/store \
DRYML_STAGE5_7_TFDS_DATA_DIR=/persistent/tfds \
DRYML_STAGE5_7_OUTPUT_STORE=/persistent/stage5_7/case-output \
DRYML_STAGE5_7_WORK_DIR=/persistent/stage5_7/case-work \
DRYML_STAGE5_7_EVIDENCE_DIR=/persistent/stage5_7/case-evidence \
./tests.sh tests/qualification/test_stage5_7_local.py \
  --stage5-7-qualification --no-cov -x
```

Without the switch collection labels real work `UNRUN`. With the switch but
missing paths, packages, TFDS fixture, or device prerequisites, it reports a
truthful `QualificationUnrun` skip. It does not download MNIST, provision Ray,
or substitute synthetic data for release evidence. This harness is not numerical
qualification evidence and does not complete U10's real gates.

The worker matrix additionally requires an existing
`DRYML_STAGE5_7_CONTROL_STORE`. Its six Ray cells require an existing
same-host `DRYML_STAGE5_7_RAY_ADDRESS`; no address leaves those cells unrun while
the 18 non-Ray cells remain eligible.

The accelerated gates additionally require a caller-selected decimal
`DRYML_STAGE5_7_GPU_DEVICE`; the harness neither provisions a GPU nor falls back
to CPU. Run only their explicit path with `--stage5-7-qualification`. Collection
without that option, or collection-only with it, does not inspect hardware or
initialize TensorFlow, Torch, or TFDS. No accelerated qualification evidence has
been recorded by this change.

Configured output Store, work, and evidence paths are distinct existing roots,
not case destinations. They must be non-symlink paths with no equality or
ancestor/descendant overlap with one another, the control Store, the read-only
fixture Store, or the prepared TFDS root. The standalone U10 local gates use
`<case_id>--local-qualification`; the U11 CPU matrix uses
`<case_id>--cpu-matrix`, preserving a common case identity while allowing both
gates under one root. The separate recovery gate retains the W3/Torch/subprocess
case identity but uses `<case_id>--recovery-step64`, so a full configured run
cannot collide. Children are created once without replacement only after
coordinator preflight. Each selected real subprocess/Ray case launches
in a fresh worker process
with a 300-second deadline; exceeding that budget is a failed qualification gate,
not an unrun prerequisite. The runner preserves partial diagnostic outputs after
timeout, uses the fixture Store only as read authority, and measures bytes from
only that case's child Store, never shared fixtures or prior cases. Process-local
CPU controls and framework seeds are set before model materialization; a
preinitialized incompatible framework is reported unrun.
