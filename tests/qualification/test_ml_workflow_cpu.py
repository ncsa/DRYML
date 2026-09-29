"""Worker matrix contracts and opt-in real CPU/recovery qualification gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset
from tests.qualification.ml_workflow_fixtures import (
    FixtureManifest, FixtureManifestError, FixtureReferences, QualificationUnrun, REQUIRED_ENVIRONMENT_KEYS,
    TFDSAuthority, load_baseline,
)
from tests.qualification.ml_workflow_workers import (
    QualificationWorkerRequest, inspect_worker_transport,
    execute_real_worker_request, preflight_coordinator_request, recovery_request, run_worker_request, submit_worker_request, validate_recovery_report,
    worker_request, worker_requests, _publish_final_evidence,
)
from tests.qualification.ml_workflow_workloads import cpu_matrix, supplemental_tfds_torch_case


_TEST_ENVIRONMENT = {key: "test" for key in REQUIRED_ENVIRONMENT_KEYS}


def _authority(tmp_path):
    """Create tiny authority paths without framework, codec, or network work."""

    fixture_store = tmp_path / "fixture-store"
    repo = Repo(DirStore(fixture_store))
    numpy_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[2.0]], dtype=np.float32)}))
    parquet_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[3.0]], dtype=np.float32)}))
    tfds = tmp_path / "tfds"
    tfds.mkdir()
    manifest = FixtureManifest(
        fixture_store.resolve(), load_baseline(), FixtureReferences(numpy_ref, parquet_ref),
        _TEST_ENVIRONMENT, TFDSAuthority(tfds, "mnist", "default", "1.0.0", "0" * 64),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_data()), encoding="ascii")
    roots = {name: tmp_path / name for name in ("output", "work", "evidence", "control")}
    for root in roots.values():
        root.mkdir()
    return manifest, manifest_path, tfds, roots


class _FakeExecutor:
    """Capture Core Execute submission arguments without running a worker."""

    def __init__(self):
        self.calls = []

    def submit(self, target, payload):
        """Record one inert submission and return its routine fake result."""

        self.calls.append((target, payload))
        return {"submitted": payload["case"]["workload"]}


def _requests(tmp_path, *, ray_address="127.0.0.1:6379"):
    """Return fresh matrix request data and their selected test authority."""

    manifest, manifest_path, tfds, roots = _authority(tmp_path)
    return worker_requests(
        manifest, manifest_path=manifest_path, tfds_data_dir=tfds, output_store=roots["output"],
        work_dir=roots["work"], evidence_dir=roots["evidence"], control_store=roots["control"],
        ray_address=ray_address,
    ), manifest, manifest_path, tfds, roots


def test_worker_matrix_enumerates_all_24_primary_cells_once_and_excludes_supplement(tmp_path):
    """The U11 matrix is complete, stable, unique, and not multiplied by TFDS mode."""

    requests, manifest, *_ = _requests(tmp_path)
    assert len(requests) == len({request.case.case_id for request in requests}) == len({request.request_id for request in requests}) == 24
    assert {(request.case.workload, request.case.framework, request.case.execution) for request in requests} == {
        (workload, framework, execution)
        for workload in ("W1", "W2", "W3") for framework in ("tf", "torch")
        for execution in ("local", "managed-local", "subprocess", "ray")
    }
    assert all(request.case.w3_test_ref == manifest.references.numpy for request in requests if request.case.workload == "W3")
    assert supplemental_tfds_torch_case(manifest).case_id not in {request.case.case_id for request in requests}


def test_fake_execute_routes_all_24_requests_without_live_transport_or_framework_import(tmp_path):
    """Routine routing covers every worker request while remaining framework-free."""

    requests, *_ = _requests(tmp_path)
    executor = _FakeExecutor()
    for request in requests:
        submit_worker_request(executor, request)
    assert len(executor.calls) == 24
    assert all(target is run_worker_request for target, _ in executor.calls)
    for request, (_, payload) in zip(requests, executor.calls):
        assert QualificationWorkerRequest.from_data(payload) == request
        assert json.loads(json.dumps(payload, ensure_ascii=True)) == payload
        optional_before = {
            name for name in ("pandas", "tensorflow", "tensorflow_datasets", "torch")
            if name in sys.modules
        }
        result = run_worker_request(payload)
        # Routing must not add optional imports, even when an earlier routine
        # history test has already loaded pandas in this shared pytest process.
        assert set(result["optional_modules"]) <= optional_before
        if request.case.execution in {"subprocess", "ray"}:
            assert request.resource_mode == "worker-process-no-session-allocation"


def test_worker_authority_and_ray_target_reject_before_submission(tmp_path):
    """Missing fixture/control authority and Ray address cannot mutate a workload."""

    requests, _, _, _, roots = _requests(tmp_path)
    executor = _FakeExecutor()
    missing_control = next(request for request in requests if request.case.execution == "local")
    Path(missing_control.control_store).rmdir()
    with pytest.raises(QualificationUnrun, match="control store"):
        submit_worker_request(executor, missing_control)
    assert executor.calls == []
    ray = next(request for request in requests if request.case.execution == "ray")
    with pytest.raises(QualificationUnrun, match="RAY_ADDRESS"):
        QualificationWorkerRequest(
            ray.case, ray.manifest_path, ray.tfds_data_dir, ray.output_store, ray.work_dir,
            ray.evidence_dir, ray.control_store, ray.execution_backend, ray.resource_mode,
        )
    assert roots["output"].is_dir()


def test_missing_ray_address_classifies_only_six_selected_ray_cells_as_unrun(tmp_path):
    """Matrix enumeration remains complete while only selected Ray requests are unrun."""

    _, manifest, manifest_path, tfds, roots = _requests(tmp_path)
    requests = worker_requests(
        manifest, manifest_path=manifest_path, tfds_data_dir=tfds, output_store=roots["output"],
        work_dir=roots["work"], evidence_dir=roots["evidence"], control_store=roots["control"],
    )
    assert len(requests) == 18
    outcomes = []
    for case in cpu_matrix(manifest):
        try:
            worker_request(
                manifest, case, manifest_path=manifest_path, tfds_data_dir=tfds,
                output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
                control_store=roots["control"],
            )
        except QualificationUnrun:
            outcomes.append((case.execution, "unrun"))
        else:
            outcomes.append((case.execution, "constructible"))
    assert outcomes.count(("ray", "unrun")) == 6
    assert all(status == "constructible" for execution, status in outcomes if execution != "ray")


def test_coordinator_preflight_completes_before_case_children_or_spool(tmp_path, monkeypatch):
    """Coordinator authority checks happen before a request can create case state."""

    requests, manifest, _, _, roots = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    control = DirStore(roots["control"])
    control.close()
    import tests.qualification.ml_workflow_fixtures as fixtures
    import tests.qualification.ml_workflow_workers as workers

    monkeypatch.setattr(workers, "installed_environment", lambda: _TEST_ENVIRONMENT)
    monkeypatch.setattr(workers, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(fixtures, "validate_tfds_content_authority", lambda authority: None)
    preflight_coordinator_request(request)
    assert not list(roots["output"].iterdir())
    assert not list(roots["work"].iterdir())
    assert not list(roots["evidence"].iterdir())


def test_worker_transport_and_fresh_process_reconstruct_only_closed_request_data(tmp_path):
    """A fresh process decodes only portable request data and imports no ML payload."""

    requests, *_ = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    payload = inspect_worker_transport(request)
    assert "Repo" not in json.dumps(payload) and "DirStore" not in json.dumps(payload)
    code = """
import json, sys
from tests.qualification.ml_workflow_workers import run_worker_request
print(json.dumps(dict(run_worker_request(json.loads(sys.argv[1]))), sort_keys=True))
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path.cwd() / "src"), str(Path.cwd()), os.environ.get("PYTHONPATH")]))}
    observed = json.loads(subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)], check=True, capture_output=True,
        text=True, env=environment,
    ).stdout)
    assert observed["request_id"] == request.request_id
    assert observed["optional_modules"] == []
    assert observed["worker_pid"] != os.getpid()


@pytest.mark.parametrize("execution, allocated", (("local", False), ("managed-local", True)))
def test_fresh_local_isolation_preserves_semantic_session_mode(tmp_path, execution, allocated):
    """Local cells use a fresh child without relabeling semantic local execution."""

    requests, manifest, manifest_path, tfds, roots = _requests(tmp_path)
    case = next(item.case for item in requests if item.case.execution == execution)
    request = worker_request(
        manifest, case, manifest_path=manifest_path, tfds_data_dir=tfds,
        output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
        control_store=roots["control"],
    )
    code = """
import json, sys
from tests.qualification.ml_workflow_workers import run_local_isolation_probe
print(json.dumps(run_local_isolation_probe(json.loads(sys.argv[1]))))
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path.cwd() / "src"), str(Path.cwd()), os.environ.get("PYTHONPATH")]))}
    observed = json.loads(subprocess.run(
        [sys.executable, "-c", code, json.dumps(request.to_data())], check=True,
        capture_output=True, text=True, env=environment,
    ).stdout)
    assert observed["pid"] != os.getpid()
    assert observed["session_allocation"] is allocated
    assert observed["semantic"]["execution_backend"] == "core-local"
    assert observed["semantic"]["resource_mode"] == request.resource_mode


def test_final_evidence_file_is_non_replacing_and_reopens_exact_bytes(tmp_path):
    """Only the coordinator final publisher can create one durable final file."""

    class Evidence:
        def to_data(self):
            return {"accepted": True, "refs": ["one", "two"]}

    import tests.qualification.ml_workflow_workers as workers

    path = tmp_path / "qualification-evidence.json"
    synchronized = []
    original = workers._fsync_directory
    workers._fsync_directory = synchronized.append
    try:
        _publish_final_evidence(path, Evidence())
    finally:
        workers._fsync_directory = original
    assert json.loads(path.read_text(encoding="ascii")) == Evidence().to_data()
    assert synchronized == [tmp_path]
    with pytest.raises(FixtureManifestError, match="replace"):
        _publish_final_evidence(path, Evidence())


def test_coordinator_classifies_absent_prerequisite_but_rejects_corrupt_authority(tmp_path, monkeypatch):
    """Only absent pre-launch authority is unrun; malformed authority is hard failure."""

    requests, manifest, manifest_path, tfds, roots = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    Path(request.control_store).rmdir()
    with pytest.raises(QualificationUnrun):
        preflight_coordinator_request(request)
    roots["control"].mkdir()
    manifest_path.write_text("{", encoding="ascii")
    monkeypatch.setattr("tests.qualification.ml_workflow_workers.installed_environment", lambda: _TEST_ENVIRONMENT)
    with pytest.raises(FixtureManifestError, match="malformed"):
        preflight_coordinator_request(request)


def test_recovery_request_requires_step_64_restore_and_artifact_repair_order(tmp_path):
    """Recovery controls cannot accept restart, duplicate, or update-before-repair evidence."""

    _, manifest, manifest_path, tfds, roots = _requests(tmp_path)
    request = recovery_request(
        manifest, manifest_path=manifest_path, tfds_data_dir=tfds, output_store=roots["output"],
        work_dir=roots["work"], evidence_dir=roots["evidence"], control_store=roots["control"],
    )
    assert request.recovery is not None and request.recovery.interrupt_after_step == 64
    facts = {
        "model_digest": "m", "model_placement": ["cpu"], "optimizer_digest": "o", "optimizer_placement": ["cpu"], "optimizer_iterations": [64],
        "epoch": 1, "next_batch": 0, "step": 64, "examples_seen": 4096,
        "loss_numerator": 1.0, "loss_denominator": 64, "checkpoint_ref": "c",
        "history_occurrence": "v1:attempt:2", "artifact_status": "pending:a",
    }
    report = {
        "checkpoint_step": 64, "before": facts, "restored": dict(facts),
        "events": ("artifact_repaired", "optimizer_update:65"),
        "final_refs": {"experiment": "e", "model": "m", "test": "t", "history": "h", "artifact": "a"},
    }
    validate_recovery_report(report)
    with pytest.raises(Exception, match="optimizer update"):
        validate_recovery_report({**report, "events": ("optimizer_update:65", "artifact_repaired")})


def test_tiny_core_subprocess_uses_the_same_worker_submission_transport(tmp_path):
    """A production Core Execute subprocess accepts the top-level worker callable."""

    requests, _, _, _, roots = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    from dryml.core import Repo
    from dryml.core.execute import CoreOptions, Executor
    from dryml.core.store.dir import DirStore
    from dryml.execute.subprocess import SubProcessConfig

    repo = Repo(DirStore(roots["output"]))
    control = DirStore(roots["control"])
    spool = roots["work"] / "core-spool"
    spool.mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, control_store=control, return_objects=False),
    )
    try:
        observed = submit_worker_request(executor, request).result(timeout=30)
        assert observed["request_id"] == request.request_id
        assert observed["worker_pid"] != os.getpid()
    finally:
        executor.close(cancel=True, timeout=10)
        repo.close(flush=False)
        control.close()


def test_coordinator_scope_keeps_import_and_materialization_restrictions_after_success_and_error(tmp_path):
    """Coordinator-only routing uses the product orchestrator floor and restores it."""

    import dryml
    from dryml import session
    from dryml.runtime import RuntimeMode, active_runtime, materialization_action
    from dryml.runtime.errors import RuntimeTransitionError
    from dryml.artifacts import Value
    from dryml.data import Dataset
    from dryml.models import ExperimentData, Model
    from tests.qualification.ml_workflow_workers import _coordinator_result_scope, _coordinator_scope, _load_selected_coordinator_results

    requests, *_ = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    blocked = {"tensorflow", "tensorflow_datasets", "torch", "pandas"}
    before = set(sys.modules) & blocked
    with _coordinator_scope():
        assert active_runtime().mode is RuntimeMode.ORCHESTRATOR
        assert run_worker_request(inspect_worker_transport(request))["request_id"] == request.request_id
        with pytest.raises(RuntimeTransitionError, match="object-mode floor"):
            dryml.configure(object_mode="fresh")
        loaded = []

        class ResultRepo:
            def load_state_ref(self, reference, **kwargs):
                loaded.append((reference, kwargs))
                return reference

        def reference(cls):
            return SimpleNamespace(definition=SimpleNamespace(cls=cls))

        history, artifact = reference(ExperimentData), reference(Value)
        assert _load_selected_coordinator_results(
            ResultRepo(), history_ref=history, artifact_ref=artifact,
        ) == (history, artifact)
        assert [reference for reference, _ in loaded] == [history, artifact]
        for rejected in (Model, Dataset):
            for slot in ("history_ref", "artifact_ref"):
                loaded.clear()
                selected = {"history_ref": history, "artifact_ref": artifact}
                selected[slot] = reference(rejected)
                with pytest.raises(FixtureManifestError, match="not .*compatible"):
                    _load_selected_coordinator_results(ResultRepo(), **selected)
                assert loaded == []
        assert materialization_action().value == "strict"
        with pytest.raises(RuntimeError, match="selected load failure"):
            with _coordinator_result_scope():
                assert materialization_action().value == "warn"
                raise RuntimeError("selected load failure")
        assert materialization_action().value == "strict"
    assert active_runtime().mode is RuntimeMode.NONE
    assert set(sys.modules) & blocked == before
    with pytest.raises(RuntimeError, match="scope failure"):
        with _coordinator_scope():
            raise RuntimeError("scope failure")
    assert active_runtime().mode is RuntimeMode.NONE
    session.reset()


def test_local_child_observation_rejects_request_or_allocation_mismatch(tmp_path):
    """Coordinator accepts only the child-observed local session/allocation facts."""

    from tests.qualification.ml_workflow_workers import _validate_local_observation

    requests, *_ = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "managed-local")
    observed = {
        "pid": 123, "semantic_mode": "managed-local", "session_mode": "managed",
        "runtime_mode": "inline", "has_allocation": True,
        "allocation": "RuntimeAllocationView",
    }
    assert _validate_local_observation(observed, request, 123) == observed
    with pytest.raises(FixtureManifestError, match="actual session mode"):
        _validate_local_observation({**observed, "session_mode": "python"}, request, 123)
    with pytest.raises(FixtureManifestError, match="allocation observation"):
        _validate_local_observation({**observed, "allocation": "no-session-allocation"}, request, 123)


def test_worker_environment_threads_only_the_selected_control_store(tmp_path):
    """Local workers receive the request control authority independently of output state."""

    from tests.qualification.ml_workflow_workers import _request_environment

    requests, *_ = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "local")
    original = os.environ.get("DRYML_ML_QUALIFICATION_CASE_CONTROL_STORE")
    with _request_environment(request):
        assert os.environ["DRYML_ML_QUALIFICATION_CASE_CONTROL_STORE"] == request.control_store
        assert os.environ["DRYML_ML_QUALIFICATION_CASE_CONTROL_STORE"] != request.output_store
    assert os.environ.get("DRYML_ML_QUALIFICATION_CASE_CONTROL_STORE") == original


def test_core_snapshot_without_allocation_reports_worker_process_truthfully(tmp_path):
    """Core worker evidence does not relabel a missing session allocation as managed."""

    from tests.qualification.ml_workflow_workers import _observed_core_evidence

    @dataclass(frozen=True)
    class Reference:
        value: str

        def digest(self):
            return self.value

    @dataclass(frozen=True)
    class Evidence:
        worker: dict
        final_experiment_ref: Reference
        runtime: dict

    requests, *_ = _requests(tmp_path)
    request = next(item for item in requests if item.case.execution == "subprocess")
    evidence = Evidence({}, Reference("result"), {"process_id": 1, "worker_id": "pid:1"})
    snapshot = SimpleNamespace(
        backend=SimpleNamespace(
            worker_id="subprocess:17", pid=17, allocation=None, submission_id="submission",
            report=object(),
        ),
        evidence=SimpleNamespace(publications=(), updates=()),
    )
    observed = _observed_core_evidence(evidence, request, snapshot)
    assert observed.worker["runtime_allocation"] == {
        "resource_mode": "worker-process-no-session-allocation",
        "allocation": "worker-process-no-session-allocation",
        "admission": "admitted-without-session-allocation",
    }
    assert observed.runtime == {"process_id": 17, "worker_id": "subprocess:17"}


def test_canonical_state_digest_is_stable_bounded_and_device_independent_for_arrays_and_scalars():
    """Recovery state digests retain typed values, not pickle or array layout accidents."""

    from tests.qualification.ml_workflow_workers import _canonical_state_digest

    first = {"tensor": np.asarray([[1.0, 2.0]], dtype=np.float32), "state": [None, True, 3, -0.0, "ok"]}
    second = {"state": [None, True, 3, -0.0, "ok"], "tensor": np.asfortranarray(first["tensor"])}
    assert _canonical_state_digest(first) == _canonical_state_digest(second)
    assert _canonical_state_digest(first) != _canonical_state_digest({**first, "state": [None, True, 4, -0.0, "ok"]})
    changed = np.array(first["tensor"], copy=True)
    changed[0, 0] = 9.0
    assert _canonical_state_digest(first) != _canonical_state_digest({**first, "tensor": changed})
    with pytest.raises(FixtureManifestError, match="non-finite"):
        _canonical_state_digest(np.asarray([float("nan")], dtype=np.float32))
    with pytest.raises(FixtureManifestError, match="dtype"):
        _canonical_state_digest(np.asarray([object()], dtype=object))
    nested = 0
    for _ in range(33):
        nested = [nested]
    with pytest.raises(FixtureManifestError, match="structural bounds"):
        _canonical_state_digest(nested)


def test_canonical_state_digest_matches_tiny_torch_state_dict_and_changes_with_state():
    """A fresh tiny Torch model/optimizer state has deterministic recovery digests."""

    torch = pytest.importorskip("torch")
    from tests.qualification.ml_workflow_workers import _canonical_state_digest, _state_placement

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones((1, 2))).sum().backward()
    optimizer.step()
    model_state, optimizer_state = model.state_dict(), optimizer.state_dict()
    assert _canonical_state_digest(model_state) == _canonical_state_digest(copy.deepcopy(model_state))
    assert _canonical_state_digest(optimizer_state) == _canonical_state_digest(copy.deepcopy(optimizer_state))
    changed_model = copy.deepcopy(model_state)
    next(iter(changed_model.values())).add_(1)
    assert _canonical_state_digest(model_state) != _canonical_state_digest(changed_model)
    changed_optimizer = copy.deepcopy(optimizer_state)
    state = next(iter(changed_optimizer["state"].values()))
    if hasattr(state["step"], "add_"):
        state["step"].add_(1)
    else:
        state["step"] += 1
    assert _canonical_state_digest(optimizer_state) != _canonical_state_digest(changed_optimizer)
    assert _state_placement(model_state) == _state_placement(optimizer_state) == ("cpu",)


@pytest.mark.ml_workflow_qualification
@pytest.mark.parametrize("case_index", range(24), ids=lambda index: f"cpu-{index:02d}")
def test_ml_workflow_real_cpu_worker_matrix(case_index):
    """Explicit real worker matrix gate; collection remains unrun without opt-in authority.

    The executed runner is intentionally supplied by release automation after it
    selects persistent fixture/output/control roots and, for Ray, an existing
    ``DRYML_ML_QUALIFICATION_RAY_ADDRESS``.  This test never creates a Ray target.
    """

    manifest_path = os.environ.get("DRYML_ML_QUALIFICATION_MANIFEST")
    fixture_store = os.environ.get("DRYML_ML_QUALIFICATION_FIXTURE_STORE")
    tfds_root = os.environ.get("DRYML_ML_QUALIFICATION_TFDS_DATA_DIR")
    roots = {name: os.environ.get(variable) for name, variable in {
        "output": "DRYML_ML_QUALIFICATION_OUTPUT_STORE", "work": "DRYML_ML_QUALIFICATION_WORK_DIR",
        "evidence": "DRYML_ML_QUALIFICATION_EVIDENCE_DIR", "control": "DRYML_ML_QUALIFICATION_CONTROL_STORE",
    }.items()}
    if not manifest_path or not fixture_store or not tfds_root or not all(roots.values()):
        pytest.skip("QualificationUnrun: prepared U10 authority and output/control roots are required")
    from tests.qualification.ml_workflow_fixtures import installed_environment, load_manifest
    try:
        manifest = load_manifest(manifest_path, fixture_store=fixture_store, tfds_data_dir=tfds_root, environment=installed_environment())
        case = cpu_matrix(manifest)[case_index]
        request = worker_request(
            manifest, case, manifest_path=manifest_path, tfds_data_dir=tfds_root,
            output_store=roots["output"], work_dir=roots["work"],
            evidence_dir=roots["evidence"], control_store=roots["control"],
            ray_address=os.environ.get("DRYML_ML_QUALIFICATION_RAY_ADDRESS"),
        )
        assert execute_real_worker_request(request).case == request.case
    except QualificationUnrun as error:
        pytest.skip(f"QualificationUnrun: {error}")


@pytest.mark.ml_workflow_qualification
def test_ml_workflow_real_torch_w3_subprocess_recovery():
    """Explicit step-64 recovery gate; it is never represented by fake evidence."""

    manifest_path = os.environ.get("DRYML_ML_QUALIFICATION_MANIFEST")
    fixture_store = os.environ.get("DRYML_ML_QUALIFICATION_FIXTURE_STORE")
    tfds_root = os.environ.get("DRYML_ML_QUALIFICATION_TFDS_DATA_DIR")
    roots = {name: os.environ.get(variable) for name, variable in {
        "output": "DRYML_ML_QUALIFICATION_OUTPUT_STORE", "work": "DRYML_ML_QUALIFICATION_WORK_DIR",
        "evidence": "DRYML_ML_QUALIFICATION_EVIDENCE_DIR", "control": "DRYML_ML_QUALIFICATION_CONTROL_STORE",
    }.items()}
    if not manifest_path or not fixture_store or not tfds_root or not all(roots.values()):
        pytest.skip("QualificationUnrun: real recovery requires prepared U10 authority and output/control roots")
    from tests.qualification.ml_workflow_fixtures import installed_environment, load_manifest
    try:
        manifest = load_manifest(manifest_path, fixture_store=fixture_store, tfds_data_dir=tfds_root, environment=installed_environment())
        request = recovery_request(
            manifest, manifest_path=manifest_path, tfds_data_dir=tfds_root, output_store=roots["output"],
            work_dir=roots["work"], evidence_dir=roots["evidence"], control_store=roots["control"],
        )
        evidence = execute_real_worker_request(request)
        assert evidence.recovery is not None
        validate_recovery_report(evidence.recovery)
    except QualificationUnrun as error:
        pytest.skip(f"QualificationUnrun: {error}")
