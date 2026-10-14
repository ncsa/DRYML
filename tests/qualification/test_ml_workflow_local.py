"""Routine contract tests plus explicitly marked real ML workflow gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import replace

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by Windows CI collection.
    _resource = None

import numpy as np
import pytest

from dryml.core import Definition, Repo
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset, Map
from dryml.artifacts import CachedDataset
from dryml.managed import ManagedConfig
from tests.qualification.ml_workflow_fixtures import (
    FixtureManifest, FixtureManifestError, FixtureReferences, QualificationUnrun,
    REQUIRED_ENVIRONMENT_KEYS, TFDSAuthority, _tfds_content_digest, config_digest, load_baseline, load_manifest,
    prepare_manifest, verify_codec_equivalence,
)
from tests.qualification.ml_workflow_workloads import (
    QualificationCase, QualificationEvidence, _QualificationMetric, accuracy_formula, build_workload, case_from_manifest, cpu_matrix,
    mnist_pipeline, mse_formula, native_device_evidence, observe_jax_training_tensors,
    qualification_case_paths, require_real_qualification, run_local_case, supplemental_tfds_torch_case,
    w1_label_methods,
)


_TEST_ENVIRONMENT = {key: "test" for key in REQUIRED_ENVIRONMENT_KEYS}
_QUALIFICATION_CASE_TIMEOUT_SECONDS = 300


def _peak_rss_bytes() -> int:
    """Return this fresh process's absolute maximum RSS in normalized bytes."""

    if _resource is None:
        raise QualificationUnrun(
            "Peak RSS measurement requires the platform resource module."
        )
    value = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _authority(tmp_path):
    """Create reference-only manifest authority without codec or framework work."""

    store = tmp_path / "store"
    repo = Repo(DirStore(store))
    numpy_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[2.0]], dtype=np.float32)}))
    parquet_ref = repo.save_object(ArrayDataset({"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[3.0]], dtype=np.float32)}))
    manifest = FixtureManifest(
        store.resolve(), load_baseline(), FixtureReferences(numpy_ref, parquet_ref), _TEST_ENVIRONMENT,
        TFDSAuthority(tmp_path / "tfds", "mnist", "default", "1.0.0", "0" * 64),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_data()), encoding="ascii")
    return path, store, manifest


def test_version_one_baseline_is_explicitly_unsupported(tmp_path):
    """The JAX-expanded v2 baseline never reinterprets persistent v1 authority."""

    value = load_baseline()
    value["version"] = 1
    path = tmp_path / "baseline-v1.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(FixtureManifestError, match="unsupported"):
        load_baseline(path)


@pytest.mark.parametrize("mutation", ("missing", "malformed", "version", "config", "environment", "extra"))
def test_manifest_rejects_missing_malformed_version_config_environment_and_extra_fields(tmp_path, mutation):
    """Manifest authority and mandatory environment compatibility fail before Store work."""

    path, store, manifest = _authority(tmp_path)
    if mutation == "missing":
        path.unlink()
    elif mutation == "malformed":
        path.write_text("{", encoding="ascii")
    else:
        value = manifest.to_data()
        if mutation == "version":
            value["version"] = 1
        elif mutation == "config":
            value["baseline"] = {"changed": True}
        elif mutation == "environment":
            value["environment"]["python"] = "wrong"
        else:
            value["extra"] = True
        path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(FixtureManifestError):
        load_manifest(path, fixture_store=store, tfds_data_dir=tmp_path / "tfds", environment=_TEST_ENVIRONMENT)


def test_manifest_uses_fixed_digest_selected_store_and_complete_environment(tmp_path):
    """A manifest cannot drift its KTD11 config, Store, or version-key schema."""

    path, store, manifest = _authority(tmp_path)
    value = manifest.to_data()
    assert value["config_digest"] == config_digest(load_baseline())
    value["config_digest"] = "0" * 64
    path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(FixtureManifestError, match="digest"):
        load_manifest(path, fixture_store=store, tfds_data_dir=tmp_path / "tfds", environment=_TEST_ENVIRONMENT)
    path.write_text(json.dumps(manifest.to_data()), encoding="ascii")
    with pytest.raises(FixtureManifestError, match="Store"):
        load_manifest(path, fixture_store=tmp_path / "other-store", tfds_data_dir=tmp_path / "tfds", environment=_TEST_ENVIRONMENT)
    with pytest.raises(FixtureManifestError, match="keys"):
        load_manifest(path, fixture_store=store, tfds_data_dir=tmp_path / "tfds", environment={"python": "test"})


def test_preparation_requires_opt_in_and_never_overwrites_authority(tmp_path, monkeypatch):
    """Preparation rejects before a builder can download, replace, or reuse a Store."""

    calls = []
    def build(store, baseline):
        calls.append((store, baseline))
        raise AssertionError("builder must not run without explicit permission")

    with pytest.raises(QualificationUnrun):
        prepare_manifest(tmp_path / "manifest.json", fixture_store=tmp_path / "store", build=build, environment=_TEST_ENVIRONMENT, tfds_data_dir=tmp_path / "tfds")
    assert calls == []
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_fixtures.require_supported_pyarrow",
        lambda: "25.0.1",
    )
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="ascii")
    with pytest.raises(FixtureManifestError, match="replace"):
        prepare_manifest(existing, fixture_store=tmp_path / "store", build=build, environment=_TEST_ENVIRONMENT, tfds_data_dir=tmp_path / "tfds", allow_download=True)
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "user-data").write_text("keep", encoding="ascii")
    with pytest.raises(FixtureManifestError, match="nonempty"):
        prepare_manifest(tmp_path / "other.json", fixture_store=populated, build=build, environment=_TEST_ENVIRONMENT, tfds_data_dir=tmp_path / "tfds", allow_download=True)
    assert calls == []


def test_unsupported_pyarrow_fails_before_manifest_or_store_mutation(tmp_path, monkeypatch):
    """The Parquet prerequisite rejects before locks, directories, or builders run."""

    calls = []
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_fixtures.require_supported_pyarrow",
        lambda: (_ for _ in ()).throw(QualificationUnrun("pyarrow unsupported")),
    )

    with pytest.raises(QualificationUnrun, match="pyarrow"):
        prepare_manifest(
            tmp_path / "new" / "manifest.json", fixture_store=tmp_path / "store",
            build=lambda *_: calls.append(True), environment=_TEST_ENVIRONMENT,
            tfds_data_dir=tmp_path / "tfds", allow_download=True,
        )

    assert calls == []
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "store").exists()


def test_w3_codec_equivalence_requires_distinct_refs_order_shapes_dtypes_and_values(tmp_path):
    """Float32/float64, order, shape, and value drift fail exact retained-cache checks."""

    _, _, manifest = _authority(tmp_path)
    calls = []
    values = ({"x": np.asarray([1.0], dtype=np.float32), "y": np.asarray([2.0], dtype=np.float32)},)
    def load(reference):
        calls.append(reference)
        return values
    verify_codec_equivalence(manifest.references, load=load)
    assert calls == [manifest.references.numpy, manifest.references.parquet]
    with pytest.raises(FixtureManifestError, match="dtype"):
        verify_codec_equivalence(manifest.references, load=lambda reference: values if reference == manifest.references.numpy else ({"x": np.asarray([1.0], dtype=np.float64), "y": np.asarray([2.0], dtype=np.float64)},))
    with pytest.raises(FixtureManifestError, match="distinct"):
        FixtureReferences(manifest.references.numpy, manifest.references.numpy)


def test_tiny_local_cache_refs_are_genuinely_distinct_without_network(tmp_path):
    """Routine fixture setup creates separate completed local cache StateRefs."""

    repo = Repo(DirStore(tmp_path / "cache-store"))
    arrays = {"x": np.asarray([[1.0]], dtype=np.float32), "y": np.asarray([[2.0]], dtype=np.float32)}
    first = CachedDataset(ArrayDataset(arrays, validate_lengths=True)).compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    second = CachedDataset(ArrayDataset(arrays, validate_lengths=False)).compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    assert first != second
    assert repo.load_state_ref(first, reuse_live="never").ready
    assert repo.load_state_ref(second, reuse_live="never").ready


def test_workload_graph_is_publicly_inspectable_prepared_and_executable(tmp_path):
    """Canonical MethodGraph proves preprocessing and synthetic-consumer handoff path."""

    source = ArrayDataset((np.zeros((2, 2, 2, 1), dtype=np.uint8), np.asarray([0, 1], dtype=np.int64)))
    pipeline = mnist_pipeline(source)
    graph = pipeline.method_graph()
    graph.learn(strategy="local")
    names = {node.method_type.__name__ for node in graph.method_nodes if node.method_type is not None}
    assert {"Project", "Select", "ImageNormalize", "Flatten", "Cast"} <= names
    # This is the public prepared iterator-to-consumer boundary, not private Pipe fields.
    first = next(graph.iterator())
    assert isinstance(first, tuple) and len(first) == 2
    assert first[0].dtype == np.dtype("float32") and first[0].shape == (4,)
    assert w1_label_methods()["prediction"].__class__.__name__ == "ArgMax"
    torch = pytest.importorskip("torch")
    from dryml import F
    from dryml.models.torch import Sequential
    consumer = Map(ArrayDataset(np.zeros((1, 4), dtype=np.float32)), Sequential(layer_defs=(F("Linear", 4, 1),)))
    consumer_graph = consumer.method_graph()
    consumer_graph.learn(strategy="local")
    assert [edge.adapter for edge in consumer_graph.conversion_edges] == ["numpy_to_torch"]
    assert next(consumer_graph.iterator()).device.type == "cpu"
    _, _, manifest = _authority(tmp_path)
    matrix = cpu_matrix(manifest)
    assert len(matrix) == 36
    assert all(case.case_kind == "matrix" and not case.tensorflow_mode for case in matrix)
    supplemental = supplemental_tfds_torch_case(manifest)
    assert supplemental.case_id not in {case.case_id for case in matrix}
    assert supplemental.case_kind == "tfds-tensorflow-to-torch"
    assert supplemental.tensorflow_mode
    assert all(case.w3_test_ref == manifest.references.numpy for case in matrix if case.workload == "W3")


@pytest.mark.parametrize(
    ("workload", "artifact_name"),
    (("W1", "accuracy"), ("W2", "reconstruction_mse"), ("W3", "test_mse")),
)
def test_all_workload_artifact_recipes_construct_from_public_dataset_shapes(
    tmp_path, monkeypatch, workload, artifact_name,
):
    """W1-W3 preflight exact Artifact graphs without mutating fixture authority."""

    def snapshot(root):
        return {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    if workload == "W3":
        fixture_store = tmp_path / "w3-fixtures"
        fixture_repo = Repo(DirStore(fixture_store))
        values = {
            "x": np.asarray([[1.0]], dtype=np.float32),
            "y": np.asarray([[2.0]], dtype=np.float32),
        }
        first = CachedDataset(
            ArrayDataset(values, validate_lengths=True), repo=fixture_repo,
        ).compute(codec="numpy", managed=ManagedConfig(state_repo=fixture_repo))
        second = CachedDataset(
            ArrayDataset(values, validate_lengths=False), repo=fixture_repo,
        ).compute(codec="numpy", managed=ManagedConfig(state_repo=fixture_repo))
        assert fixture_repo.load_state_ref(first, reuse_live="never").ready
        manifest = FixtureManifest(
            fixture_store.resolve(),
            load_baseline(),
            FixtureReferences(first, second),
            _TEST_ENVIRONMENT,
            TFDSAuthority(tmp_path / "tfds", "mnist", "default", "1.0.0", "0" * 64),
        )
    else:
        _, fixture_store, manifest = _authority(tmp_path)
    output_store = DirStore(tmp_path / "output")
    repo = Repo((
        output_store,
        DirStore.open_existing(fixture_store, query_index="none"),
    ))
    control_store = DirStore(tmp_path / "control", query_index="none")
    managed = ManagedConfig(state_repo=repo, control_store=control_store)
    case = case_from_manifest(
        manifest, workload=workload, framework="torch", execution="local",
    )
    source = ArrayDataset((
        np.zeros((4, 28, 28, 1), dtype=np.uint8),
        np.arange(4, dtype=np.int64),
    ))
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_workloads.initialize_cpu_framework",
        lambda framework, seed: None,
    )
    fixture_before = snapshot(fixture_store)

    experiment = build_workload(
        repo,
        case,
        managed=managed,
        mnist_source=lambda split, tensorflow_mode: source,
    )
    if workload == "W3":
        assert experiment.test_data.object_ref == first.object
    experiment._preflight_artifacts()
    model_ref = repo.save(experiment.model, deep_capture=True)
    x_path, y_path = ((0,), (1,)) if workload in {"W1", "W2"} else ("x", "y")
    artifact = _QualificationMetric(
        experiment.test_data.last_state_ref,
        model_ref,
        metric="accuracy" if workload == "W1" else "mse",
        x=x_path,
        y=y_path,
        fixture_store=os.fspath(fixture_store) if workload == "W3" else None,
        repo=repo,
    )
    result_ref = artifact.compute(
        store=output_store,
        managed=managed,
    )
    result = repo.load_state_ref(result_ref, reuse_live="never").value()

    assert tuple(experiment.artifacts) == (artifact_name,)
    assert experiment.train_data.example_cardinality().require_finite() in {4, 4096}
    assert np.asarray(result).shape == () and np.isfinite(result)
    assert snapshot(fixture_store) == fixture_before
    if workload in {"W1", "W2"}:
        assert output_store.read_state_ref_record(
            experiment.test_data.last_state_ref.digest()
        ) is not None
        fixture_handle = DirStore.open_existing(fixture_store, query_index="none")
        try:
            assert fixture_handle.read_state_ref_record(
                experiment.test_data.last_state_ref.digest()
            ) is None
        finally:
            fixture_handle.close()


@pytest.mark.parametrize(
    ("workload", "artifact_name"),
    (("W1", "accuracy"), ("W2", "reconstruction_mse"), ("W3", "test_mse")),
)
def test_all_workload_artifact_definitions_preflight_without_framework_imports(
    workload, artifact_name,
):
    """Every recipe has a concrete Artifact root before native setup begins."""

    from tests.models.test_dataset_training_integration import CountingModel

    if workload in {"W1", "W2"}:
        x_path, y_path = (0,), (1,)
    else:
        x_path, y_path = "x", "y"
    recipe = Definition(
        _QualificationMetric,
        CountingModel(),
        CountingModel(),
        metric="accuracy" if workload == "W1" else "mse",
        x=x_path,
        y=y_path,
        fixture_store="/fixture" if workload == "W3" else None,
    )

    definition = recipe.concretize()

    assert definition.cls.resolve() is _QualificationMetric
    assert definition.parameters["metric"] == (
        "accuracy" if artifact_name == "accuracy" else "mse"
    )
    assert definition.parameters["fixture_store"] == (
        "/fixture" if workload == "W3" else None
    )


def test_workload_cache_uses_selected_control_store(tmp_path, monkeypatch):
    """W1/W2 cache creation retains the request's distinct control authority."""

    from tests.models.test_dataset_training_integration import CountingModel, CountingTrainer
    import dryml.artifacts

    _, _, manifest = _authority(tmp_path)
    repo = Repo(DirStore(tmp_path / "output"))
    control = DirStore(tmp_path / "control", query_index="none")
    managed = ManagedConfig(state_repo=repo, control_store=control)
    captured = {}

    class ObservedCache:
        def __init__(self, source, *, repo):
            captured.update(source=source, repo=repo)

        def compute(self, **kwargs):
            captured.update(kwargs)
            return None

    monkeypatch.setattr(dryml.artifacts, "CachedDataset", ObservedCache)
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_workloads.initialize_cpu_framework",
        lambda framework, seed: None,
    )
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_workloads._model_and_training",
        lambda framework, workload, seed: (CountingModel(), CountingTrainer()),
    )
    source = ArrayDataset((
        np.zeros((1, 28, 28, 1), dtype=np.uint8),
        np.zeros((1,), dtype=np.int64),
    ))
    case = case_from_manifest(
        manifest, workload="W1", framework="torch", execution="managed-local",
    )

    build_workload(
        repo,
        case,
        managed=managed,
        mnist_source=lambda split, tensorflow_mode: source,
    )

    assert captured["repo"] is repo
    assert captured["store"] is repo.stores[0]
    assert captured["managed"] is managed
    assert captured["managed"].control_store is control


def test_case_is_closed_and_carries_manifest_environment_seed_and_fixture_identity(tmp_path):
    """Fresh request decoding rejects missing/extra fields and retains exact W3 identity."""

    _, _, manifest = _authority(tmp_path)
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    assert QualificationCase.from_data(case.to_data()) == case
    malformed = case.to_data()
    malformed.pop("seed")
    with pytest.raises(FixtureManifestError):
        QualificationCase.from_data(malformed)
    assert case.config_digest == config_digest(manifest.baseline)
    assert dict(case.environment) == _TEST_ENVIRONMENT


def test_case_paths_isolate_sequential_fake_cases_and_measure_only_each_store(tmp_path):
    """Three case identities use non-replacing child paths and no cumulative Store size."""

    _, _, manifest = _authority(tmp_path)
    roots = tuple(tmp_path / name for name in ("output", "work", "evidence"))
    for root in roots:
        root.mkdir()
    manifest.tfds.data_dir.mkdir()
    cases = (
        case_from_manifest(manifest, workload="W1", framework="tf", execution="local"),
        case_from_manifest(manifest, workload="W2", framework="torch", execution="local"),
        case_from_manifest(manifest, workload="W3", framework="tf", execution="local"),
    )
    paths = [qualification_case_paths(
        case, output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
        tfds_root=manifest.tfds.data_dir, create=True,
    ) for case in cases]
    supplemental_paths = qualification_case_paths(
        supplemental_tfds_torch_case(manifest), output_store_root=roots[0], work_root=roots[1],
        evidence_root=roots[2], tfds_root=manifest.tfds.data_dir, create=True,
    )
    for index, path in enumerate(paths, start=1):
        (path.output_store / "case.bin").write_bytes(b"x" * index)
        (path.evidence_dir / "qualification-evidence.json").write_text("{}", encoding="ascii")
    all_paths = (*paths, supplemental_paths)
    assert len({path.output_store for path in all_paths}) == len(all_paths)
    assert len({path.work_dir for path in all_paths}) == len(all_paths)
    assert len({path.evidence_dir for path in all_paths}) == len(all_paths)
    assert [sum(file.stat().st_size for file in path.output_store.rglob("*") if file.is_file()) for path in paths] == [1, 2, 3]
    with pytest.raises(FixtureManifestError, match="replace"):
        qualification_case_paths(
            cases[0], output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
            tfds_root=manifest.tfds.data_dir,
        )


@pytest.mark.parametrize("root_name", ("evidence", "output", "work"))
def test_case_paths_reject_fixture_or_tfds_containment_before_creating_children(tmp_path, root_name):
    """Output roots cannot equal, contain, or sit below fixture/TFDS authority."""

    _, store, manifest = _authority(tmp_path)
    tfds_root = tmp_path / "tfds"
    tfds_root.mkdir()
    case = case_from_manifest(manifest, workload="W1", framework="torch", execution="local")
    roots = {name: tmp_path / name for name in ("output", "work", "evidence")}
    for root in roots.values():
        root.mkdir()
    if root_name == "evidence":
        roots[root_name] = store
    elif root_name == "output":
        roots[root_name] = tmp_path
    else:
        roots[root_name] = tfds_root / "child"
        roots[root_name].mkdir()
    with pytest.raises(QualificationUnrun, match="disjoint"):
        qualification_case_paths(
            case, output_store_root=roots["output"], work_root=roots["work"],
            evidence_root=roots["evidence"], tfds_root=tfds_root, create=True,
        )
    assert not any((root / case.case_id).exists() for root in roots.values() if root.exists())


def test_case_paths_accept_disjoint_fixture_tfds_and_output_roots(tmp_path):
    """Five separately rooted authorities create only their isolated case children."""

    _, _, manifest = _authority(tmp_path)
    tfds_root = tmp_path / "tfds"
    tfds_root.mkdir()
    roots = tuple(tmp_path / name for name in ("output", "work", "evidence"))
    for root in roots:
        root.mkdir()
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    paths = qualification_case_paths(
        case, output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
        tfds_root=tfds_root, create=True,
    )
    assert all(path.is_dir() for path in (paths.output_store, paths.work_dir, paths.evidence_dir))


def test_local_cpu_and_recovery_gate_paths_are_disjoint_for_one_underlying_case(tmp_path):
    """U10 local, U11 CPU, and recovery gates cannot collide for one case identity."""

    _, _, manifest = _authority(tmp_path)
    tfds_root = tmp_path / "tfds"
    tfds_root.mkdir()
    roots = tuple(tmp_path / name for name in ("output", "work", "evidence"))
    for root in roots:
        root.mkdir()
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="subprocess")
    primary = qualification_case_paths(
        case, output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
        tfds_root=tfds_root, gate_id="local-qualification", create=True,
    )
    cpu = qualification_case_paths(
        case, output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
        tfds_root=tfds_root, gate_id="cpu-matrix", create=True,
    )
    recovery = qualification_case_paths(
        case, output_store_root=roots[0], work_root=roots[1], evidence_root=roots[2],
        tfds_root=tfds_root, gate_id="recovery-step64", create=True,
    )
    assert len({primary.output_store, cpu.output_store, recovery.output_store}) == 3
    assert case.case_id in primary.output_store.name and case.case_id in recovery.output_store.name


def test_qualification_child_timeout_preserves_diagnostic_evidence_and_never_accepts_it(tmp_path, monkeypatch):
    """A timed-out child is reaped by subprocess.run and cannot yield a success record."""

    _, _, manifest = _authority(tmp_path)
    manifest.tfds.data_dir.mkdir()
    roots = tuple(tmp_path / name for name in ("output", "work", "evidence"))
    for root in roots:
        root.mkdir()
    monkeypatch.setenv("DRYML_ML_QUALIFICATION_OUTPUT_STORE", os.fspath(roots[0]))
    monkeypatch.setenv("DRYML_ML_QUALIFICATION_WORK_DIR", os.fspath(roots[1]))
    monkeypatch.setenv("DRYML_ML_QUALIFICATION_EVIDENCE_DIR", os.fspath(roots[2]))
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    child_code = """
import os
import time
from pathlib import Path
Path(os.environ['DRYML_ML_QUALIFICATION_CASE_EVIDENCE_DIR'], 'worker-provisional.json').write_text('{}', encoding='ascii')
time.sleep(10)
"""
    started = time.monotonic()
    with pytest.raises(FixtureManifestError, match=case.case_id):
        _real_runner_in_child(
            manifest, case, timeout_seconds=0.1, child_code=child_code,
        )
    assert time.monotonic() - started < 3
    evidence_path = roots[2] / f"{case.case_id}--local-qualification" / "worker-provisional.json"
    assert evidence_path.read_text(encoding="ascii") == "{}"


def test_tfds_content_digest_detects_bytes_but_ignores_metadata(tmp_path):
    """Prepared TFDS authority hashes file content, not only names, sizes, or mtimes."""

    root = tmp_path / "mnist"
    root.mkdir()
    data = root / "split.tfrecord"
    data.write_bytes(b"abcd")
    builder = type("Builder", (), {"data_path": root})()
    original = _tfds_content_digest(builder)
    os.utime(data, None)
    assert _tfds_content_digest(builder) == original
    data.write_bytes(b"wxyz")
    assert _tfds_content_digest(builder) != original
    (root / "linked").symlink_to(data)
    with pytest.raises(QualificationUnrun, match="symlink"):
        _tfds_content_digest(builder)


def test_native_device_evidence_requires_matching_parameter_and_training_tensors():
    """Synthetic TF-like and Torch-like devices prove closed CPU/GPU/mismatch gates."""

    class Tensor:
        def __init__(self, device):
            self._device = device
            self.device_reads = 0

        @property
        def device(self):
            self.device_reads += 1
            return self._device

    class TorchModel:
        def __init__(self, *devices):
            self._parameters = tuple(Tensor(device) for device in devices)

        def parameters(self):
            return iter(self._parameters)

    class KerasModel:
        def __init__(self, *devices):
            self.variables = tuple(Tensor(device) for device in devices)

    assert native_device_evidence(TorchModel("cpu"), training_tensors=(Tensor("cpu"),)) == "cpu"
    assert native_device_evidence(KerasModel("/device:GPU:0"), training_tensors=(Tensor("cuda:0"),)) == "gpu:0"
    shared = Tensor("cpu")
    shared_model = TorchModel()
    shared_model._parameters = (shared, shared)
    assert native_device_evidence(shared_model, training_tensors=(Tensor("cpu"),)) == "cpu"
    assert shared.device_reads == 1
    with pytest.raises(FixtureManifestError, match="mixed"):
        native_device_evidence(TorchModel("cuda:0"), training_tensors=(Tensor("cpu"),))
    with pytest.raises(FixtureManifestError, match="missing"):
        native_device_evidence(KerasModel("/device:CPU:0"), training_tensors=(object(),))


def test_peak_rss_is_absolute_and_cannot_be_reduced_by_a_preloaded_baseline(monkeypatch):
    """Each fresh child reports its lifetime peak, never a delta from setup RSS."""

    usage = type("Usage", (), {"ru_maxrss": 4096})()
    resource = type("Resource", (), {})()
    resource.RUSAGE_SELF = object()
    resource.getrusage = lambda _: usage
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_resource", resource)
    monkeypatch.setattr(sys, "platform", "linux")
    preloaded_baseline = 8192 * 1024
    assert preloaded_baseline > _peak_rss_bytes()
    assert _peak_rss_bytes() == 4096 * 1024
    monkeypatch.setattr(module, "_resource", None)
    with pytest.raises(QualificationUnrun, match="resource module"):
        _peak_rss_bytes()


def test_native_device_evidence_traverses_tiny_torch_autoencoder_on_cpu():
    """Composite Torch evidence observes both graph children without private field walks."""

    torch = pytest.importorskip("torch")
    from dryml import F
    from dryml.models import AutoEncoder
    from dryml.models.torch import Sequential

    model = AutoEncoder(
        Sequential(layer_defs=(F("Linear", 2, 1),)),
        Sequential(layer_defs=(F("Linear", 1, 2),)),
    )
    assert native_device_evidence(model, training_tensors=(torch.zeros((1, 2)),)) == "cpu"


def test_native_device_evidence_traverses_tiny_tf_autoencoder_on_cpu():
    """Composite TensorFlow evidence observes both built graph children on CPU."""

    tf = pytest.importorskip("tensorflow")
    from dryml import F
    from dryml.models import AutoEncoder
    from dryml.models.tf import Sequential

    model = AutoEncoder(
        Sequential(layer_defs=(F("Dense", units=1),)),
        Sequential(layer_defs=(F("Dense", units=2),)),
    )
    tensor = tf.zeros((1, 2))
    model(tensor)
    assert native_device_evidence(model, training_tensors=(tensor,)) == "cpu"


def test_formula_threshold_and_history_contracts_reject_unrun_or_fake_runner(tmp_path):
    """Routine tests do not replace real numerical qualification with fake evidence."""

    _, _, manifest = _authority(tmp_path)
    accuracy = accuracy_formula([1, 0, 1, 1, 1], [1, 0, 1, 1, 0])
    assert accuracy == 0.8
    assert mse_formula([1.0, 2.0], [1.0, 2.1]) > 0
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    with pytest.raises(QualificationUnrun):
        run_local_case(manifest, case, opted_in=False, runner=lambda _: pytest.fail("runner was called"))
    with pytest.raises(QualificationUnrun, match="explicit opt-in"):
        require_real_qualification(False)


def test_evidence_record_is_closed_round_trippable_and_rejects_nonfinite_or_extra_fields(tmp_path):
    """Closed evidence rejects malformed records before association validation or publication."""

    manifest_path, _, manifest = _authority(tmp_path)
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    roots = {name: tmp_path / name for name in ("output", "work", "evidence", "control")}
    for root in roots.values():
        root.mkdir()
    from tests.qualification.ml_workflow_workers import validate_request_evidence, worker_request
    request = worker_request(
        manifest, case, manifest_path=manifest_path, tfds_data_dir=tmp_path / "tfds",
        output_store=roots["output"], work_dir=roots["work"], evidence_dir=roots["evidence"],
        control_store=roots["control"],
    )
    evidence = QualificationEvidence(
        case=case, final_experiment_ref=manifest.references.numpy, model_ref=manifest.references.parquet,
        test_ref=manifest.references.numpy, history_ref=manifest.references.parquet,
        history_rows=({"state_ref": manifest.references.numpy, "evaluation_status": "completed", "eval_artifacts": {"test_mse": manifest.references.parquet}},),
        artifact_ref=manifest.references.parquet, artifact_value=0.01,
        formula={"name": "noisy_observation_mse", "value": 0.01, "provenance": "independent_test"},
        environment=manifest.environment,
        runtime={"backend": "torch", "device": "cpu", "worker_id": "test", "process_id": 1},
        elapsed_seconds=0.0, peak_rss_bytes=0, output_bytes=0,
        worker={
            "execution_backend": "worker-provisional", "semantic_backend": "local",
            "isolation": {"kind": "worker-provisional", "parent_pid": 0},
            "worker_pid": 1, "worker_identity": "test",
            "runtime_allocation": {"resource_mode": "python", "allocation": "none", "admission": "in-process"},
            "shared_authority": {"fixture_store": os.fspath(manifest.fixture_store), "control_store": os.fspath(roots["control"].resolve())},
            "submitted": {"request_digest": request.request_id, "submission_id": "in-process:1", "result_ref": manifest.references.numpy.digest()},
            "core_outcome": {"kind": "in-process", "publication_refs": (), "update_refs": ()},
            "final_refs": {
                "experiment": manifest.references.numpy.digest(), "model": manifest.references.parquet.digest(),
                "test": manifest.references.numpy.digest(), "history": manifest.references.parquet.digest(),
                "artifact": manifest.references.parquet.digest(),
            },
            "coordinator_validated_refs": None,
        },
    )
    assert QualificationEvidence.from_data(evidence.to_data()) == evidence
    validate_request_evidence(request, evidence)
    with pytest.raises(FixtureManifestError, match="substituted"):
        validate_request_evidence(request, replace(evidence, worker={**evidence.worker, "submitted": {**evidence.worker["submitted"], "request_digest": "other"}}))
    malformed = evidence.to_data()
    malformed["extra"] = True
    with pytest.raises(FixtureManifestError, match="extra"):
        QualificationEvidence.from_data(malformed)
    with pytest.raises(FixtureManifestError, match="finite"):
        QualificationEvidence(
            case=case, final_experiment_ref=manifest.references.numpy, model_ref=manifest.references.parquet,
            test_ref=manifest.references.numpy, history_ref=manifest.references.parquet,
            history_rows=evidence.history_rows, artifact_ref=manifest.references.parquet, artifact_value=float("nan"),
            formula=evidence.formula, environment=manifest.environment, runtime=evidence.runtime,
            elapsed_seconds=0.0, peak_rss_bytes=0, output_bytes=0, worker=evidence.worker,
        )


def _real_manifest_or_unrun():
    """Load explicitly selected persistent authority for a marked numerical gate."""

    manifest_path = os.environ.get("DRYML_ML_QUALIFICATION_MANIFEST")
    fixture_store = os.environ.get("DRYML_ML_QUALIFICATION_FIXTURE_STORE")
    if not manifest_path or not fixture_store:
        pytest.skip("QualificationUnrun: set DRYML_ML_QUALIFICATION_MANIFEST and DRYML_ML_QUALIFICATION_FIXTURE_STORE")
    from tests.qualification.ml_workflow_fixtures import installed_environment
    try:
        tfds_data_dir = os.environ.get("DRYML_ML_QUALIFICATION_TFDS_DATA_DIR")
        if not tfds_data_dir:
            raise QualificationUnrun("set DRYML_ML_QUALIFICATION_TFDS_DATA_DIR")
        return load_manifest(manifest_path, fixture_store=fixture_store, tfds_data_dir=tfds_data_dir, environment=installed_environment())
    except QualificationUnrun as error:
        pytest.skip(f"QualificationUnrun: {error}")

def _real_runner(manifest, case, *, recovery=None, worker_request_id=None, control_store=None):
    """Run a real local Experiment/Artifact workflow after explicit test opt-in."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.data import TFDSAdapter
    from dryml.data import Map
    from dryml.managed import ManagedConfig
    from dryml.models import ExperimentData
    from tests.qualification.ml_workflow_workloads import (
        QualificationEvidence, _FORMULAS, _METRIC_NAMES, native_device_evidence,
        prepare_exact_model_for_formula,
    )

    def source(split, tensorflow_mode):
        return TFDSAdapter(
            "mnist", split=split, as_supervised=True, as_numpy=not tensorflow_mode,
            data_dir=os.fspath(manifest.tfds.data_dir), download=False,
        )

    output_store = os.environ.get("DRYML_ML_QUALIFICATION_CASE_OUTPUT_STORE")
    work_dir = os.environ.get("DRYML_ML_QUALIFICATION_CASE_WORK_DIR")
    if not output_store or not work_dir:
        raise QualificationUnrun("qualification child lacks isolated output Store/work paths")
    output_path = Path(output_store).expanduser().resolve(strict=False)
    work_path = Path(work_dir).expanduser().resolve(strict=False)
    if output_path == manifest.fixture_store or work_path == manifest.fixture_store:
        raise FixtureManifestError("Real case output Store/work directory must be distinct from fixture authority.")
    if not work_path.is_dir():
        raise QualificationUnrun("Selected real-case work directory is unavailable.")
    control_path = Path(
        control_store or os.environ.get("DRYML_ML_QUALIFICATION_CASE_CONTROL_STORE") or output_path
    ).expanduser().resolve(strict=False)
    if control_path == manifest.fixture_store or not control_path.is_dir():
        raise FixtureManifestError("Real case control Store must be an existing non-fixture authority.")
    repo = Repo((
        DirStore(output_path),
        DirStore.open_existing(manifest.fixture_store, query_index="none"),
    ))
    control = DirStore.open_existing(control_path)
    managed = ManagedConfig(state_repo=repo, control_store=control)
    started = time.monotonic()
    experiment = build_workload(repo, case, managed=managed, mnist_source=source)
    training_tensors, execution_tensors = [], []
    native_models = tuple(
        model for model in repo.iter_graph(experiment.model, missing="raise", order="post")
        if getattr(model, "native_backend", None) == case.framework
    )
    if not native_models:
        raise FixtureManifestError("Real runner cannot observe a native training model boundary.")
    def collect(destination, value):
        if isinstance(value, dict):
            for item in value.values():
                collect(destination, item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(destination, item)
        else:
            destination.append(value)

    def observe_tf_train_step(*, x, y, prediction, parameters):
        del parameters
        if not training_tensors:
            collect(training_tensors, (x, y))
        if not execution_tensors:
            collect(execution_tensors, prediction)

    original_torch_forwards = []
    if case.framework == "torch":
        for training_model in native_models:
            original_forward = training_model.obj.forward

            def capture_torch_forward(x, *args, _forward=original_forward, **kwargs):
                if not training_tensors:
                    collect(training_tensors, x)
                result = _forward(x, *args, **kwargs)
                if not execution_tensors:
                    collect(execution_tensors, result)
                return result

            original_torch_forwards.append((training_model, original_forward))
            training_model.obj.forward = capture_torch_forward
    observer_scope = nullcontext()
    if case.framework == "tf":
        from dryml.models.tf.base import observe_keras_train_step

        observer_scope = observe_keras_train_step(observe_tf_train_step)
    elif case.framework == "jax":
        observer_scope = observe_jax_training_tensors(
            experiment.train_fn,
            training_tensors=training_tensors,
            execution_tensors=execution_tensors,
        )
    recovery_evidence = None
    try:
        if recovery is None:
            with observer_scope:
                final = experiment.train(managed=managed)
        else:
            import dryml.models.experiment as experiment_module
            from dryml.artifacts import Artifact
            from tests.qualification.ml_workflow_workers import _canonical_state_digest, _state_placement

            original_boundary = experiment_module._experiment_boundary
            triggered = []

            def retained_facts(value, checkpoint, state_repo):
                """Capture live or freshly restored step-64 recovery authority."""

                model_state = value.model.obj.state_dict()
                optimizer_state = value.train_fn.optimizer.obj.state_dict()
                iterations = tuple(
                    int(state["step"])
                    for state in optimizer_state["state"].values()
                    if "step" in state
                )
                history = ExperimentData.find(checkpoint.object_projection(), repo=state_repo)
                if history is None:
                    raise FixtureManifestError("Recovery checkpoint lacks a durable history row.")
                rows = history.data[history.data["state_ref"] == checkpoint]
                if len(rows) != 1:
                    raise FixtureManifestError("Recovery checkpoint has no unique durable history occurrence.")
                row = rows.iloc[0]
                initial = row.artifact_inputs.get("test_mse") if isinstance(row.artifact_inputs, dict) else None
                if initial is None:
                    raise FixtureManifestError("Recovery interruption lacks the retained Artifact receiver.")
                # The initial receiver plus managed completion authority is the
                # only valid way to discover the durable completed result here.
                completed = Artifact.recover(
                    initial, repo=state_repo, control_store=control, reuse_live="never",
                )
                if not completed.ready or completed.last_state_ref is None:
                    raise FixtureManifestError("Recovery interruption lacks a durable completed Artifact result.")
                state = value.state
                return {
                    "model_digest": _canonical_state_digest(model_state),
                    "model_placement": _state_placement(model_state),
                    "optimizer_digest": _canonical_state_digest(optimizer_state),
                    "optimizer_placement": _state_placement(optimizer_state),
                    "optimizer_iterations": iterations, "epoch": state.epoch,
                    "next_batch": state.next_batch, "step": state.step,
                    "examples_seen": state.examples_seen, "loss_numerator": state.loss_numerator,
                    "loss_denominator": state.loss_denominator, "checkpoint_ref": checkpoint.digest(),
                    "history_occurrence": str(row.row_key),
                    "artifact_status": f"{row.evaluation_status}:{completed.last_state_ref.digest()}",
                }

            live_facts = None

            def interrupt_after_retained_step(boundary):
                if (boundary == recovery.fail_boundary and not triggered
                        and experiment.state.step == recovery.interrupt_after_step):
                    triggered.append(boundary)
                    checkpoint = experiment.train.status(
                        state_repo=repo, control_store=control,
                    ).checkpoint_state_ref
                    if checkpoint is None:
                        raise FixtureManifestError("Recovery interruption lacks a checkpoint receipt.")
                    nonlocal live_facts
                    live_facts = retained_facts(experiment, checkpoint, repo)
                    probe = work_path / "recovery-live-probe.json"
                    encoded = json.dumps(live_facts, sort_keys=True, default=list)
                    if len(encoded.encode("utf-8")) > 1024 * 1024:
                        raise FixtureManifestError("Recovery diagnostic probe exceeds its bounded size.")
                    probe.write_text(encoded, encoding="utf-8")
                    raise RuntimeError("ML workflow deterministic recovery interruption")
                original_boundary(boundary)

            experiment_module._experiment_boundary = interrupt_after_retained_step
            try:
                with pytest.raises(RuntimeError, match="deterministic recovery interruption"):
                    experiment.train(managed=managed)
            finally:
                experiment_module._experiment_boundary = original_boundary
            checkpoint = experiment.train.status(
                state_repo=repo, control_store=control,
            ).checkpoint_state_ref
            if checkpoint is None or experiment.state.step != recovery.interrupt_after_step:
                raise FixtureManifestError("Recovery did not retain the required step-64 checkpoint.")
            if live_facts is None:
                raise FixtureManifestError("Recovery interruption did not capture live step-64 facts.")
            # A new Repo guarantees this is a restoration comparison, not a live
            # cache comparison. The outer worker process is already isolated.
            repo.close(flush=False)
            repo = Repo(DirStore(output_path))
            managed = ManagedConfig(state_repo=repo, control_store=control)
            retained = repo.load_state_ref(checkpoint, reuse_live="never", cache="none")
            before = live_facts
            restored = retained_facts(retained, checkpoint, repo)
            if restored != before:
                raise FixtureManifestError("Fresh Repo restore does not equal live step-64 facts.")
            if before["step"] != recovery.interrupt_after_step or before["artifact_status"].split(":", 1)[0] not in {"pending", "failed"}:
                raise FixtureManifestError("Recovery did not retain the required pending step-64 authority.")
            repair_events = []

            def record_repair_order(boundary):
                if boundary == "completed_status_published" and retained.state.step == recovery.interrupt_after_step:
                    repair_events.append("artifact_repaired")
                original_boundary(boundary)

            experiment_module._experiment_boundary = record_repair_order
            import dryml.models.torch.base as torch_training
            original_update = torch_training.record_train_update

            def record_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                if retained.state.step == recovery.interrupt_after_step + 1:
                    repair_events.append(f"optimizer_update:{retained.state.step}")
                return result

            torch_training.record_train_update = record_update
            try:
                final = retained.train(managed=managed)
            finally:
                experiment_module._experiment_boundary = original_boundary
                torch_training.record_train_update = original_update
            if tuple(repair_events)[:2] != ("artifact_repaired", "optimizer_update:65"):
                raise FixtureManifestError("Recovery did not prove Artifact repair before optimizer update 65.")
            repaired_history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
            repaired_rows = repaired_history.data[repaired_history.data["state_ref"] == checkpoint] if repaired_history is not None else ()
            if len(repaired_rows) != 1 or str(repaired_rows.iloc[0].row_key) != before["history_occurrence"]:
                raise FixtureManifestError("Recovery created a duplicate history row instead of repairing the retained occurrence.")
            repaired_artifact = repaired_rows.iloc[0].eval_artifacts.get("test_mse")
            if repaired_artifact is None or repaired_artifact.digest() != before["artifact_status"].split(":", 1)[1]:
                raise FixtureManifestError("Recovery created a duplicate Artifact result instead of repairing the retained result.")
            recovery_evidence = {
                "checkpoint_step": recovery.interrupt_after_step,
                "before": before,
                "restored": restored,
                "events": tuple(repair_events),
                "final_refs": None,
            }
    finally:
        for training_model, original_forward in original_torch_forwards:
            training_model.obj.forward = original_forward
    if not training_tensors or not execution_tensors:
        raise FixtureManifestError("Real runner did not observe native training and execution tensors.")
    history = ExperimentData.find(final.object_projection(), repo=repo)
    if history is None or history.data.empty:
        raise FixtureManifestError("Real runner did not publish ExperimentData history.")
    row = history.data.iloc[-1]
    artifact_ref = row.eval_artifacts[_METRIC_NAMES[case.workload]]
    artifact_value = float(repo.load_state_ref(artifact_ref, reuse_live="never").value())
    model_ref = final.at("model")
    test_ref = final.at("test_data")
    model = repo.load_state_ref(model_ref, reuse_live="never")
    test_data = repo.load_state_ref(test_ref, reuse_live="never")
    prepare_exact_model_for_formula(model, case)
    predictions, observations = [], []
    for sample in test_data:
        if isinstance(sample, dict):
            inputs, targets = sample["x"], sample["y"]
        else:
            inputs, targets = sample
        prediction = model(inputs)
        if case.accelerator == "gpu":
            if native_device_evidence(model, training_tensors=(prediction,)) != "gpu:0":
                raise FixtureManifestError("Exact-loaded formula model did not remain on the selected GPU.")
        prediction = prediction.detach().cpu() if hasattr(prediction, "detach") else prediction
        observation = targets.detach().cpu() if hasattr(targets, "detach") else targets
        predictions.append(np.asarray(prediction))
        observations.append(np.asarray(observation))
    if case.workload == "W1":
        formula_value = accuracy_formula(np.argmax(np.asarray(predictions), axis=-1), np.asarray(observations))
    else:
        formula_value = mse_formula(np.asarray(predictions), np.asarray(observations))
    output_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
    observed_device = native_device_evidence(model, training_tensors=tuple(training_tensors))
    device_evidence = None
    if case.accelerator == "gpu":
        from tests.qualification.ml_workflow_workloads import native_device_observations, selected_gpu_device

        observations = native_device_observations(
            model, training_tensors=tuple(training_tensors), execution_tensors=tuple(execution_tensors),
        )
        device_evidence = {
            "allocation": {"visible_device": selected_gpu_device(), "claimed_device": "gpu:0"},
            **dict(observations),
        }
    final_refs = {
        "experiment": final.digest(), "model": model_ref.digest(), "test": test_ref.digest(),
        "history": history.last_state_ref.digest(), "artifact": artifact_ref.digest(),
    }
    if recovery_evidence is not None:
        recovery_evidence["final_refs"] = final_refs
    return QualificationEvidence(
        case=case, final_experiment_ref=final, model_ref=model_ref, test_ref=test_ref,
        history_ref=history.last_state_ref,
        history_rows=({"state_ref": row.state_ref, "evaluation_status": row.evaluation_status, "eval_artifacts": dict(row.eval_artifacts)},),
        artifact_ref=artifact_ref, artifact_value=artifact_value,
        formula={"name": _FORMULAS[case.workload], "value": formula_value, "provenance": "direct_saved_model_prediction_v1"},
        environment=manifest.environment,
        runtime={"backend": case.framework, "device": observed_device, "worker_id": f"pid:{os.getpid()}", "process_id": os.getpid()},
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        output_bytes=output_bytes,
        worker={
            "execution_backend": "worker-provisional",
            "semantic_backend": "local" if case.execution in {"local", "managed-local"} else case.execution,
            "isolation": {"kind": "worker-provisional", "parent_pid": os.getppid()},
            "worker_pid": os.getpid(), "worker_identity": f"pid:{os.getpid()}",
            "runtime_allocation": {"resource_mode": "worker", "allocation": "worker-local", "admission": "worker"},
            "shared_authority": {"fixture_store": os.fspath(manifest.fixture_store), "control_store": os.fspath(control_path)},
            "submitted": {"request_digest": case.case_id if worker_request_id is None else worker_request_id, "submission_id": f"worker:{os.getpid()}", "result_ref": final.digest()},
            "core_outcome": {"kind": "worker-initial", "publication_refs": (), "update_refs": ()},
            "final_refs": final_refs,
            "coordinator_validated_refs": None,
        },
        recovery=recovery_evidence,
        device_evidence=device_evidence,
    )


def _real_runner_in_child(manifest, case, *, timeout_seconds=None, child_code=None):
    """Run one real case in a fresh process and return only closed evidence.

    The child receives its manifest path/root/store selection through the explicit
    opt-in environment, creates no shared fixture output, and writes its closed
    evidence under the caller-selected case work directory for the parent to
    validate against authoritative Stores.
    """

    paths = qualification_case_paths(
        case,
        output_store_root=os.environ["DRYML_ML_QUALIFICATION_OUTPUT_STORE"],
        work_root=os.environ["DRYML_ML_QUALIFICATION_WORK_DIR"],
        evidence_root=os.environ["DRYML_ML_QUALIFICATION_EVIDENCE_DIR"],
        tfds_root=manifest.tfds.data_dir,
        gate_id="local-qualification",
        create=True,
    )
    provisional_path = paths.evidence_dir / "worker-provisional.json"
    if provisional_path.exists():
        raise FixtureManifestError("Refusing to replace existing case provisional evidence.")
    code = """
import json
import os
import sys
from pathlib import Path
from tests.qualification.test_ml_workflow_local import _real_manifest_or_unrun, _real_runner
from tests.qualification.ml_workflow_workloads import QualificationCase
manifest = _real_manifest_or_unrun()
case = QualificationCase.from_data(json.loads(sys.argv[1]))
evidence = _real_runner(manifest, case)
Path(os.environ['DRYML_ML_QUALIFICATION_CASE_EVIDENCE_DIR'], 'worker-provisional.json').write_text(json.dumps(evidence.to_data()), encoding='ascii')
""" if child_code is None else child_code
    timeout = _QUALIFICATION_CASE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    if type(timeout) not in (int, float) or timeout <= 0:
        raise ValueError("Qualification child timeout must be a positive number of seconds.")
    environment = {
        **os.environ,
        "DRYML_ML_QUALIFICATION_CASE_OUTPUT_STORE": os.fspath(paths.output_store),
        "DRYML_ML_QUALIFICATION_CASE_WORK_DIR": os.fspath(paths.work_dir),
        "DRYML_ML_QUALIFICATION_CASE_EVIDENCE_DIR": os.fspath(paths.evidence_dir),
        "PYTHONPATH": os.pathsep.join(filter(None, [
            str(Path.cwd() / "src"), str(Path.cwd()), os.environ.get("PYTHONPATH"),
        ])),
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, json.dumps(case.to_data())],
            capture_output=True, text=True, env=environment, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise FixtureManifestError(
            f"Qualification case {case.case_id} timed out after {timeout} seconds."
        ) from error
    if result.returncode:
        raise FixtureManifestError("Isolated qualification child failed without evidence.")
    try:
        provisional = QualificationEvidence.from_data(json.loads(provisional_path.read_text(encoding="ascii")))
    except (OSError, json.JSONDecodeError, FixtureManifestError) as error:
        raise FixtureManifestError("Isolated qualification child returned malformed provisional evidence.") from error
    from tests.qualification.ml_workflow_workers import _publish_final_evidence
    from tests.qualification.ml_workflow_workloads import validate_evidence

    worker = dict(provisional.worker)
    worker.update({
        "execution_backend": "isolation-process", "semantic_backend": "local",
        "isolation": {"kind": "fresh-os-process", "parent_pid": os.getpid()},
        "worker_pid": provisional.runtime["process_id"],
        "worker_identity": f"pid:{provisional.runtime['process_id']}",
        "runtime_allocation": {
            "resource_mode": "unmanaged-session", "allocation": "no-session-allocation",
            "admission": "semantic-local-in-isolation-child",
        },
        "coordinator_validated_refs": {
            "output_store": os.fspath(paths.output_store.resolve()),
            "refs": dict(provisional.worker["final_refs"]),
        },
    })
    accepted = replace(provisional, worker=worker)
    validate_evidence(manifest, accepted, output_store=paths.output_store)
    final_path = paths.evidence_dir / "qualification-evidence.json"
    _publish_final_evidence(final_path, accepted)
    try:
        reopened = QualificationEvidence.from_data(json.loads(final_path.read_text(encoding="ascii")))
    except (OSError, json.JSONDecodeError, FixtureManifestError) as error:
        raise FixtureManifestError("Isolated qualification final evidence cannot be reopened.") from error
    if reopened != accepted:
        raise FixtureManifestError("Isolated qualification final evidence differs from accepted evidence.")
    return accepted


@pytest.mark.ml_workflow_qualification
@pytest.mark.parametrize("workload", ("W1", "W2", "W3"))
def test_ml_workflow_real_local_workloads(workload):
    """Explicit W1/W2/W3 gate; missing prerequisites are unrun, never a pass."""

    manifest = _real_manifest_or_unrun()
    case = case_from_manifest(manifest, workload=workload, framework="torch", execution="local")
    try:
        output_store = os.environ.get("DRYML_ML_QUALIFICATION_OUTPUT_STORE")
        if not output_store or not os.environ.get("DRYML_ML_QUALIFICATION_WORK_DIR") or not os.environ.get("DRYML_ML_QUALIFICATION_EVIDENCE_DIR"):
            raise QualificationUnrun("set qualification output Store, work, and evidence roots")
        run_local_case(manifest, case, opted_in=True, runner=lambda request: _real_runner_in_child(manifest, request), output_store=output_store)
    except QualificationUnrun as error:
        pytest.skip(f"QualificationUnrun: {error}")
