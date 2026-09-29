# ML Workflow Qualification Harness

The qualification harness keeps reusable preprocessing in Dataset/Method graphs
and separates fixture preparation from execution. Build MNIST data from a
caller-supplied `TFDSAdapter` only after the persistent fixture manifest has been
validated:

```python
from tests.qualification.ml_workflow_workloads import mnist_pipeline, w1_label_methods

prepared = mnist_pipeline(raw_mnist_dataset)
graph = prepared.method_graph()
graph.learn(strategy="local")
labels = w1_label_methods()
```

The graph exposes `Project`, `Select`, `ImageNormalize`, `Flatten`, and `Cast`.
W1 label decoding exposes `ArgMax` for predictions and `Select` for targets.
Callers do not insert ad hoc TensorFlow-to-Torch conversion functions: supported
U4-U6 prepared graph handoffs select the direct data-boundary conversion. The
canonical `MethodGraph` exposes the nested normalization/project/layout/cast
occurrences and selected conversion edges after `learn(strategy="local")`; callers
must inspect this public graph rather than composition-private fields.

For W2, call `mnist_pipeline(raw_mnist_dataset, autoencode=True)` so the same
normalized, flattened image Method supplies both `x` and `y`. For W3, construct
the case only from the loaded manifest:

```python
from tests.qualification.ml_workflow_fixtures import load_manifest
from tests.qualification.ml_workflow_workloads import case_from_manifest

manifest = load_manifest(
    "/persistent/ml-qualification/manifest.json",
    fixture_store="/persistent/ml-qualification/store",
    tfds_data_dir="/persistent/tfds",
)
case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
assert case.w3_test_ref == manifest.references.numpy
```

This does not open the Store, TFDS, or regenerate a seed-derived dataset. The real
runner supplies that exact reference as `Experiment.test_data`, then verifies the
same reference survives checkpoint binding and every repeat. See
[ML Workflow Qualification Verification](ml_workflow_qualification_verification.md)
for fixture preparation,
fixed gates, and evidence requirements.

An opted-in local runner uses
`run_local_case(manifest, case, opted_in=True, runner=...)`. Its callback receives
only the immutable machine-readable case and returns a closed
`QualificationEvidence` record. It must include final Experiment/model/test refs,
completed history/Artifact receipts, Artifact value plus independently calculated
formula provenance, complete version/runtime/worker evidence, and bounded elapsed,
RSS, and output measurements. The harness rejects calls before opt-in and
preflights the caller-selected Store before runner invocation. The harness does
not claim an unrun case has qualified numerically.

All 24 primary CPU matrix requests use `TFDSAdapter(..., as_numpy=True)` for
NumPy delivery, including Torch requests. The separately counted
`supplemental_tfds_torch_case(manifest)` is the only local harness request with
`tensorflow_mode=True`; it proves TFDS TensorFlow delivery reaches Torch through
the prepared handoff without caller conversion code. Real MNIST execution always
constructs `TFDSAdapter` with the manifest's selected `data_dir` and
`download=False`. An absent, stale, or unrelated local MNIST builder is a
`QualificationUnrun`, never an implicit download. Qualification Experiments use
`checkpoint_every_steps=32`; `None` on the public Experiment API means
terminal-only checkpointing, while terminal evaluation always occurs.
