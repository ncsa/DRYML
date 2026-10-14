"""Routine U12 contracts and explicitly opted-in cross-framework/GPU gates.

Importing this module does not inspect GPUs or import TensorFlow, Torch, or TFDS.
Its TFDS interoperability and two GPU real gates remain opt-in after callers
select persistent authority and, for GPU cases, exactly one visible GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.cardinality import Cardinality
from dryml.core import TensorSpec
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset, Batch, GeneratorDataset
from tests.qualification.ml_workflow_fixtures import (
    FixtureManifest, FixtureManifestError, FixtureReferences, QualificationUnrun,
    REQUIRED_ENVIRONMENT_KEYS, TFDSAuthority, load_baseline,
)
from tests.qualification.ml_workflow_workloads import (
    QualificationEvidence, accelerated_cases, cpu_matrix, initialize_gpu_framework, mnist_pipeline,
    native_device_observations, preflight_gpu_framework, qualification_case_paths, selected_gpu_device,
)
from tests.qualification.ml_workflow_workers import (
    QualificationWorkerRequest, inspect_worker_transport, worker_request,
)


_TEST_ENVIRONMENT = {key: "test" for key in REQUIRED_ENVIRONMENT_KEYS}


def _tiny_tensorflow_pairs():
    """Yield fixed native TensorFlow values for the routine handoff contract."""
    import tensorflow as tf

    yield tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)
    yield tf.constant([1.0], dtype=tf.float32), tf.constant([2.0], dtype=tf.float32)


def _authority(tmp_path):
    """Create framework-free persistent authority for accelerated contracts."""

    store = tmp_path / "fixture-store"
    repo = Repo(DirStore(store))
    numpy_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[2.0]], dtype=np.float32)}))
    parquet_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[3.0]], dtype=np.float32)}))
    tfds = tmp_path / "tfds"
    tfds.mkdir()
    manifest = FixtureManifest(
        store.resolve(), load_baseline(), FixtureReferences(numpy_ref, parquet_ref),
        _TEST_ENVIRONMENT, TFDSAuthority(tfds, "mnist", "default", "1.0.0", "0" * 64),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_data()), encoding="ascii")
    roots = {name: tmp_path / name for name in ("output", "work", "evidence", "control")}
    for root in roots.values():
        root.mkdir()
    return manifest, manifest_path, roots


def _gpu_evidence(manifest, case, control):
    """Return otherwise-closed synthetic GPU evidence for rejection-boundary tests."""

    final, model = manifest.references.numpy, manifest.references.parquet
    refs = {
        "experiment": final.digest(), "model": model.digest(), "test": final.digest(),
        "history": model.digest(), "artifact": model.digest(),
    }
    return QualificationEvidence(
        case=case, final_experiment_ref=final, model_ref=model, test_ref=final,
        history_ref=model,
        history_rows=({"state_ref": final, "evaluation_status": "completed", "eval_artifacts": {"accuracy": model}},),
        artifact_ref=model, artifact_value=0.8,
        formula={"name": "categorical_accuracy", "value": 0.8, "provenance": "synthetic_boundary"},
        environment=manifest.environment,
        runtime={"backend": "tf", "device": "gpu:0", "worker_id": "test", "process_id": 1},
        elapsed_seconds=0.0, peak_rss_bytes=0, output_bytes=0,
        worker={
            "execution_backend": "worker-provisional", "semantic_backend": "subprocess",
            "isolation": {"kind": "worker-provisional", "parent_pid": 0},
            "worker_pid": 1, "worker_identity": "test",
            "runtime_allocation": {"resource_mode": "worker", "allocation": "worker-local", "admission": "worker"},
            "shared_authority": {"fixture_store": os.fspath(manifest.fixture_store), "control_store": os.fspath(control)},
            "submitted": {"request_digest": "test", "submission_id": "test", "result_ref": final.digest()},
            "core_outcome": {"kind": "worker", "publication_refs": (), "update_refs": ()},
            "final_refs": refs, "coordinator_validated_refs": None,
        },
        device_evidence={
            "allocation": {"visible_device": "7", "claimed_device": "gpu:0"},
            "native_parameters": ("gpu:0",), "training_tensors": ("gpu:0",),
            "execution_tensors": ("gpu:0",), "observed_device": "gpu:0",
        },
    )


def test_accelerated_cases_are_outside_the_matrix_and_use_noncolliding_paths(tmp_path):
    """U12 has exactly two isolated GPU cases outside the 36-cell CPU matrix."""

    manifest, _, roots = _authority(tmp_path)
    accelerated = accelerated_cases(manifest)
    assert len(cpu_matrix(manifest)) == 36
    assert {case.case_id for case in accelerated}.isdisjoint(case.case_id for case in cpu_matrix(manifest))
    paths = [
        qualification_case_paths(
            case, output_store_root=roots["output"], work_root=roots["work"],
            evidence_root=roots["evidence"], tfds_root=manifest.tfds.data_dir,
            gate_id=f"gpu-{case.framework}-{case.workload.lower()}", create=True,
        )
        for case in accelerated
    ]
    assert len({path.output_store for path in paths}) == len(paths)
    assert all("--gpu-" in path.output_store.name for path in paths)


def test_gpu_request_is_closed_requires_one_selected_device_and_keeps_frameworks_unloaded(tmp_path, monkeypatch):
    """Accelerated transport carries a caller GPU choice without probing hardware."""

    manifest, manifest_path, roots = _authority(tmp_path)
    case = accelerated_cases(manifest)[0]
    monkeypatch.delenv("DRYML_ML_QUALIFICATION_GPU_DEVICE", raising=False)
    with pytest.raises(QualificationUnrun, match="GPU_DEVICE"):
        worker_request(
            manifest, case, manifest_path=manifest_path, tfds_data_dir=manifest.tfds.data_dir,
            output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
            control_store=roots["control"],
        )
    request = worker_request(
        manifest, case, manifest_path=manifest_path, tfds_data_dir=manifest.tfds.data_dir,
        output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
        control_store=roots["control"], gpu_device="7",
    )
    assert request.resource_mode == "worker-process-one-gpu"
    assert request.gate_id == "gpu-tf-w1"
    assert QualificationWorkerRequest.from_data(inspect_worker_transport(request)) == request
    monkeypatch.setenv("DRYML_ML_QUALIFICATION_GPU_DEVICE", "3")
    assert selected_gpu_device() == "3"


def test_missing_gpu_prerequisite_is_unrun_before_worker_launch(monkeypatch):
    """A failed disposable availability probe is truthful unrun, not CPU fallback."""

    monkeypatch.setattr(
        "tests.qualification.ml_workflow_workloads.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 3})(),
    )
    with pytest.raises(QualificationUnrun, match="no single usable"):
        preflight_gpu_framework("tf", visible_device="0")


def test_jax_gpu_helpers_reject_before_probe_import_or_environment_mutation(monkeypatch):
    """CPU-only JAX qualification cannot fall through to Torch GPU handling."""

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_workloads.subprocess.run",
        lambda *args, **kwargs: pytest.fail("unsupported JAX must not launch a GPU probe"),
    )

    with pytest.raises(FixtureManifestError, match="accelerated"):
        preflight_gpu_framework("jax", visible_device="0")
    with pytest.raises(FixtureManifestError, match="accelerated"):
        initialize_gpu_framework("jax", 1, visible_device="0")

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_accelerated_evidence_rejects_missing_mixed_or_contradictory_native_device_facts(tmp_path):
    """A GPU claim needs observed model, train, execution, and allocation evidence."""

    manifest, _, roots = _authority(tmp_path)
    evidence = _gpu_evidence(manifest, accelerated_cases(manifest)[0], roots["control"])
    assert QualificationEvidence.from_data(evidence.to_data()) == evidence
    with pytest.raises(FixtureManifestError, match="incomplete"):
        replace(evidence, device_evidence={})
    mixed = dict(evidence.device_evidence)
    mixed["training_tensors"] = ("cpu",)
    with pytest.raises(FixtureManifestError, match="mixed"):
        replace(evidence, device_evidence=mixed)
    contradictory = dict(evidence.device_evidence)
    contradictory["observed_device"] = "cpu"
    with pytest.raises(FixtureManifestError, match="disagrees"):
        replace(evidence, device_evidence=contradictory)


def test_accelerated_request_evidence_rejects_selector_identity_pid_and_allocation_substitution(tmp_path):
    """Publication binds GPU facts to the exact selected request and Core worker."""
    from tests.qualification.ml_workflow_workers import validate_request_evidence

    manifest, manifest_path, roots = _authority(tmp_path)
    case = accelerated_cases(manifest)[0]
    request = worker_request(
        manifest, case, manifest_path=manifest_path, tfds_data_dir=manifest.tfds.data_dir,
        output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
        control_store=roots["control"], gpu_device="7",
    )
    evidence = _gpu_evidence(manifest, case, roots["control"])
    worker = dict(evidence.worker)
    worker.update({
        "execution_backend": "subprocess", "semantic_backend": "subprocess",
        "worker_pid": 17, "worker_identity": "subprocess:17",
        "runtime_allocation": {
            "resource_mode": "worker-process-one-gpu", "allocation": "worker-process-one-gpu",
            "admission": "admitted-with-one-gpu-visibility",
        },
        "shared_authority": {
            "fixture_store": os.fspath(manifest.fixture_store), "control_store": os.fspath(roots["control"].resolve()),
        },
        "submitted": {"request_digest": request.request_id, "submission_id": "submission", "result_ref": evidence.final_experiment_ref.digest()},
    })
    runtime = {**evidence.runtime, "process_id": 17, "worker_id": "subprocess:17"}
    accepted = replace(evidence, worker=worker, runtime=runtime)
    validate_request_evidence(request, accepted)

    with pytest.raises(FixtureManifestError, match="selector"):
        validate_request_evidence(request, replace(accepted, device_evidence={
            **accepted.device_evidence,
            "allocation": {"visible_device": "3", "claimed_device": "gpu:0"},
        }))
    with pytest.raises(FixtureManifestError, match="PID"):
        validate_request_evidence(request, replace(accepted, runtime={**accepted.runtime, "process_id": 18}))
    with pytest.raises(FixtureManifestError, match="identity"):
        validate_request_evidence(request, replace(accepted, runtime={**accepted.runtime, "worker_id": "subprocess:18"}))
    with pytest.raises(FixtureManifestError, match="allocation"):
        validate_request_evidence(request, replace(accepted, worker={
            **accepted.worker,
            "runtime_allocation": {"resource_mode": "worker-process-one-gpu", "allocation": "other", "admission": "admitted-with-one-gpu-visibility"},
        }))


def test_native_device_observations_require_one_actual_parameter_training_and_execution_placement():
    """Synthetic native values prove the GPU evidence collector rejects mixed facts."""

    class Tensor:
        def __init__(self, device):
            self.device = device

    class Model:
        def __init__(self, *devices):
            self.variables = tuple(Tensor(device) for device in devices)

    observed = native_device_observations(
        Model("/device:GPU:0"), training_tensors=(Tensor("cuda:0"),),
        execution_tensors=(Tensor("gpu:0"),),
    )
    assert observed["observed_device"] == "gpu:0"
    with pytest.raises(FixtureManifestError, match="mixed"):
        native_device_observations(
            Model("/device:GPU:0"), training_tensors=(Tensor("cpu"),),
            execution_tensors=(Tensor("gpu:0"),),
        )


def test_torch_exact_loaded_formula_model_uses_public_gpu_preparation_before_inference(tmp_path, monkeypatch):
    """GPU formula validation prepares the exact-loaded wrapper without a GPU host."""
    from tests.qualification.ml_workflow_workloads import prepare_exact_model_for_formula

    manifest, _, roots = _authority(tmp_path)
    case = accelerated_cases(manifest)[1]
    prepared = []

    class ExactLoadedModel:
        def to_device(self, device):
            prepared.append(str(device))

    monkeypatch.setenv("DRYML_ML_QUALIFICATION_GPU_DEVICE", "7")
    prepare_exact_model_for_formula(ExactLoadedModel(), case)

    assert prepared == ["cuda:0"]


def test_tensorflow_to_torch_handoff_runs_through_training_preparation_and_is_graph_visible():
    """Actual TF values reach the real Torch training boundary without caller glue."""
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")
    import dryml.tf
    from dryml.models import Experiment
    from dryml.models.torch import Model, Optimizer, Training

    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=(model := Model(torch.nn.Linear, 1, 1)), lr=0.0),
        loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
    )
    trainer(Experiment(
        model, trainer,
        train_data=Batch(
            GeneratorDataset(
                _tiny_tensorflow_pairs, cardinality=Cardinality.finite(2),
                spec=(TensorSpec("float32", shape=(1,), backend="tf"), TensorSpec("float32", shape=(1,), backend="tf")),
            ),
            1,
        ),
    ))

    assert [edge.adapter for edge in trainer.method_graph().conversion_edges] == ["tf_to_torch", "tf_to_torch"]
    assert all(parameter.grad is not None for parameter in model.obj.parameters())


@pytest.mark.ml_workflow_qualification
@pytest.mark.parametrize("gate", ("interoperability", "tensorflow-w1-gpu", "torch-w3-gpu"))
def test_ml_workflow_real_accelerated_gates(gate):
    """Run the one TFDS handoff and two caller-selected GPU gates, or report unrun."""

    from tests.qualification.test_ml_workflow_local import _real_manifest_or_unrun, _real_runner_in_child
    from tests.qualification.ml_workflow_workers import execute_real_worker_request

    manifest = _real_manifest_or_unrun()
    roots = {name: os.environ.get(variable) for name, variable in {
        "output": "DRYML_ML_QUALIFICATION_OUTPUT_STORE", "work": "DRYML_ML_QUALIFICATION_WORK_DIR",
        "evidence": "DRYML_ML_QUALIFICATION_EVIDENCE_DIR", "control": "DRYML_ML_QUALIFICATION_CONTROL_STORE",
    }.items()}
    if not all(roots.values()):
        pytest.skip("QualificationUnrun: prepared authority and output/control roots are required")
    try:
        if gate == "interoperability":
            from tests.qualification.ml_workflow_workloads import run_local_case, supplemental_tfds_torch_case

            case = supplemental_tfds_torch_case(manifest)
            run_local_case(
                manifest, case, opted_in=True,
                runner=lambda request: _real_runner_in_child(manifest, request),
                output_store=roots["output"],
            )
            return
        gpu_device = os.environ.get("DRYML_ML_QUALIFICATION_GPU_DEVICE")
        if gpu_device is None:
            raise QualificationUnrun("set one caller-selected DRYML_ML_QUALIFICATION_GPU_DEVICE")
        case = accelerated_cases(manifest)[0 if gate == "tensorflow-w1-gpu" else 1]
        request = worker_request(
            manifest, case, manifest_path=os.environ["DRYML_ML_QUALIFICATION_MANIFEST"],
            tfds_data_dir=os.environ["DRYML_ML_QUALIFICATION_TFDS_DATA_DIR"], output_store=roots["output"],
            work_dir=roots["work"], evidence_dir=roots["evidence"], control_store=roots["control"],
            gpu_device=gpu_device,
        )
        evidence = execute_real_worker_request(request)
        assert evidence.case == case and evidence.runtime["device"] == "gpu:0"
    except QualificationUnrun as error:
        pytest.skip(f"QualificationUnrun: {error}")
