"""Lightweight import and evidence contracts for ML workflow qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset

from tests.qualification.ml_workflow_fixtures import (
    FixtureManifest, FixtureReferences, REQUIRED_ENVIRONMENT_KEYS, TFDSAuthority,
    installed_environment, load_baseline,
)
from tests.qualification.ml_workflow_workloads import (
    QualificationCase, accelerated_cases, case_from_manifest, cpu_matrix, supplemental_tfds_torch_case,
)
from tests.qualification.ml_workflow_workers import QualificationWorkerRequest, inspect_worker_transport


def _manifest(tmp_path):
    """Create minimal local authority used only to exercise manifest decoding."""

    store_path = tmp_path / "store"
    repo = Repo(DirStore(store_path))
    numpy_reference = repo.save_object(ArrayDataset({
        "x": np.asarray([[1.0]], dtype=np.float32),
        "y": np.asarray([[2.0]], dtype=np.float32),
    }))
    parquet_reference = repo.save_object(ArrayDataset({
        "x": np.asarray([[1.0]], dtype=np.float32),
        "y": np.asarray([[3.0]], dtype=np.float32),
    }))
    manifest = FixtureManifest(
        store_path.resolve(), load_baseline(), FixtureReferences(numpy_reference, parquet_reference),
        {key: "test" for key in REQUIRED_ENVIRONMENT_KEYS},
        TFDSAuthority(tmp_path / "tfds", "mnist", "default", "1.0.0", "0" * 64),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_data()), encoding="ascii")
    return path, store_path, manifest


def test_qualification_case_round_trip_retains_exact_w3_reference_without_live_handles(tmp_path):
    """Fresh process case construction consumes only manifest JSON and an exact StateRef."""

    path, store_path, manifest = _manifest(tmp_path)
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    assert case.w3_test_ref == manifest.references.numpy
    assert QualificationCase.from_data(case.to_data()) == case
    matrix = cpu_matrix(manifest)
    supplemental = supplemental_tfds_torch_case(manifest)
    assert len(matrix) == 24
    assert all(case.case_kind == "matrix" and not case.tensorflow_mode for case in matrix)
    assert supplemental.case_kind == "tfds-tensorflow-to-torch" and supplemental.tensorflow_mode
    assert supplemental.case_id not in {case.case_id for case in matrix}
    accelerated = accelerated_cases(manifest)
    assert {(case.workload, case.framework, case.execution, case.accelerator) for case in accelerated} == {
        ("W1", "tf", "subprocess", "gpu"), ("W3", "torch", "subprocess", "gpu"),
    }
    assert not {case.case_id for case in accelerated} & {case.case_id for case in (*matrix, supplemental)}

    code = """
import json
from tests.qualification.ml_workflow_fixtures import load_manifest
from tests.qualification.ml_workflow_workloads import case_from_manifest
m = load_manifest(__import__('sys').argv[1], fixture_store=__import__('sys').argv[2], tfds_data_dir=__import__('sys').argv[3], environment={key: 'test' for key in __import__('tests.qualification.ml_workflow_fixtures', fromlist=['REQUIRED_ENVIRONMENT_KEYS']).REQUIRED_ENVIRONMENT_KEYS})
c = case_from_manifest(m, workload='W3', framework='tf', execution='managed-local')
print(json.dumps({'digest': c.w3_test_ref.digest(), 'loaded': sorted(n for n in __import__('sys').modules if n in {'tensorflow', 'torch', 'pandas', 'tensorflow_datasets'})}))
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path.cwd() / "src"), str(Path.cwd()), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code, str(path), str(store_path), str(tmp_path / "tfds")], check=True, capture_output=True, text=True, env=environment)
    observed = json.loads(result.stdout)
    assert observed == {"digest": manifest.references.numpy.digest(), "loaded": []}


def test_fixture_manifest_portable_tfds_identity_excludes_local_root(tmp_path):
    """The local manifest binds its root while portable cases do not repeat it."""

    _, _, manifest = _manifest(tmp_path)
    data = manifest.to_data()

    assert set(data["tfds"]) == {"data_dir", "builder", "config", "version", "content_digest"}
    case = case_from_manifest(manifest, workload="W3", framework="torch", execution="local")
    assert str(tmp_path / "tfds") not in json.dumps(case.to_data())


def test_worker_request_transport_is_closed_and_does_not_capture_core_authority(tmp_path):
    """Worker routing carries paths/refs/config only, never a live Repo or Store."""

    path, store_path, manifest = _manifest(tmp_path)
    roots = {name: tmp_path / name for name in ("output", "work", "evidence", "control")}
    for root in roots.values():
        root.mkdir()
    request = QualificationWorkerRequest(
        case_from_manifest(manifest, workload="W3", framework="torch", execution="subprocess"),
        str(path), str(tmp_path / "tfds"), str(roots["output"]), str(roots["work"]),
        str(roots["evidence"]), str(roots["control"]), "core-subprocess", "worker-process-no-session-allocation",
    )
    transport = inspect_worker_transport(request)
    assert QualificationWorkerRequest.from_data(transport) == request
    encoded = json.dumps(transport, sort_keys=True)
    assert str(store_path) in encoded
    # Framework version keys and serialized reference class names are evidence,
    # not live handles; JSON round-trip is the transport boundary.
    assert "object at 0x" not in encoded


def test_qualification_preflight_records_tfds_version_without_importing_tfds(monkeypatch):
    """Distribution preflight records TFDS evidence without authorizing data access."""

    missing = object()
    tfds_before = sys.modules.get("tensorflow_datasets", missing)
    versions = {
        "dryml": "0.3.0.dev2",
        "pandas": "3.0.0",
        "pyarrow": "25.0.1",
        "tensorflow": "2.19.0",
        "tensorflow-datasets": "4.9.9",
        "torch": "2.8.0",
    }
    monkeypatch.setattr(
        "tests.qualification.ml_workflow_fixtures.importlib.metadata.version",
        versions.__getitem__,
    )

    observed = installed_environment()

    assert observed["tensorflow_datasets"] == "4.9.9"
    assert sys.modules.get("tensorflow_datasets", missing) is tfds_before
