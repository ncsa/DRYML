# Stage 5+7 Verification

Status: persistent-fixture and opt-in local harness implemented; all numerical
qualification gates remain unrun.

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
peak RSS, and output size. Missing, extra, non-finite, negative, or inconsistent
fields fail validation.

`cpu_matrix(manifest)` emits exactly the fixed 24 NumPy-delivery requests: W1-W3,
TensorFlow/Torch, and local/managed-local/subprocess/existing-Ray execution. It
does not start a worker. Every W3 request contains the same exact manifest NumPy
reference. `supplemental_tfds_torch_case(manifest)` is a separately counted W1
TensorFlow-mode TFDS-to-Torch interoperability gate; it is not a twenty-fifth
matrix cell and does not change primary Torch delivery.

Acceptance gates are fixed: W1 accuracy >= 0.80, W2 normalized-pixel MSE <=
0.12, and W3 noisy-observation MSE <= 0.05. Artifact values must agree with
independent formula checks; histories must be nonempty and completed. The CPU
budget is 300 seconds and 8 GiB peak RSS per isolated case. No real case was run
for this change.

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

Configured output Store, work, and evidence paths are distinct existing roots,
not case destinations. They must be non-symlink paths with no equality or
ancestor/descendant overlap with one another, the read-only fixture Store, or the
prepared TFDS root. The harness derives deterministic `case.case_id` children
below each root, creates them once without replacement, and fails closed if any
child already exists. Each selected real case launches in a fresh child process
with a 300-second deadline; exceeding that budget is a failed qualification gate,
not an unrun prerequisite. The runner preserves partial diagnostic outputs after
timeout, uses the fixture Store only as read authority, and measures bytes from
only that case's child Store, never shared fixtures or prior cases. Process-local
CPU controls and framework seeds are set before model materialization; a
preinitialized incompatible framework is reported unrun.
