"""Closed worker routing and recovery controls for Stage 5+7 qualification.

This module is deliberately framework-free.  Coordinators construct JSON-only
requests, preflight selected shared authority, and submit those requests through
Core Execute.  Workers reconstruct the manifest and selected stores themselves;
no coordinator Repo, Store, dataset cursor, or native framework value is part of
the transport record.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import MappingProxyType

from .stage5_7_fixtures import (
    FixtureManifestError, QualificationUnrun, installed_environment, load_manifest,
    preflight_manifest, validate_fixture_reference_metadata,
)
from .stage5_7_workloads import QualificationCase, cpu_matrix


_ROUTES = {
    "local": ("core-local", "unmanaged"),
    "managed-local": ("core-local", "managed"),
    "subprocess": ("core-subprocess", "worker-process-no-session-allocation"),
    "ray": ("core-ray", "worker-process-no-session-allocation"),
}
_REQUEST_FIELDS = frozenset({
    "case", "manifest_path", "tfds_data_dir", "output_store", "work_dir",
    "evidence_dir", "control_store", "execution_backend", "resource_mode",
    "ray_address", "recovery", "gate_id",
})
_STATE_DIGEST_MAX_DEPTH = 32
_STATE_DIGEST_MAX_NODES = 100_000
_STATE_DIGEST_MAX_SEQUENCE = 100_000
_STATE_DIGEST_MAX_STRING_BYTES = 1 << 20
_STATE_DIGEST_MAX_RAW_BYTES = 64 << 20


def _closed_json(value: object) -> object:
    """Copy one JSON-compatible value or reject a live transport handle."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise FixtureManifestError("Qualification worker transport must contain only closed JSON data.") from error


def _canonical_state_digest(value: object) -> str:
    """Return a bounded, device-independent digest for native recovery state.

    Mappings are sorted by tagged scalar keys, sequences retain order, and
    arrays/tensors retain dtype, shape, and CPU-contiguous bytes. Unsupported,
    non-finite, cyclic, or oversized state is rejected instead of falling back
    to pickle serialization.
    """

    import numpy as np

    digest = hashlib.sha256()
    nodes = raw_bytes = 0
    active: set[int] = set()

    def emit(tag: bytes, payload: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def scalar_key(item: object) -> bytes:
        if item is None:
            return b"null"
        if type(item) is bool:
            return b"bool\x01" if item else b"bool\x00"
        if type(item) is int:
            sign = b"+" if item >= 0 else b"-"
            magnitude = abs(item).to_bytes(max(1, (abs(item).bit_length() + 7) // 8), "big")
            return b"int" + sign + len(magnitude).to_bytes(4, "big") + magnitude
        if type(item) is float:
            if not math.isfinite(item):
                raise FixtureManifestError("Recovery state contains a non-finite scalar.")
            return b"float" + struct.pack(">d", item)
        if type(item) is str:
            encoded = item.encode("utf-8")
            if len(encoded) > _STATE_DIGEST_MAX_STRING_BYTES:
                raise FixtureManifestError("Recovery state contains an oversized string.")
            return b"string" + len(encoded).to_bytes(8, "big") + encoded
        raise FixtureManifestError("Recovery state mapping keys must be exact scalar values.")

    def array_value(item: object):
        if isinstance(item, np.ndarray):
            return b"array", np.ascontiguousarray(item)
        if type(item).__module__.split(".", 1)[0] == "torch":
            try:
                return b"tensor", np.ascontiguousarray(item.detach().cpu().contiguous().numpy())
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise FixtureManifestError("Recovery state contains an unsupported tensor.") from error
        return None

    def visit(item: object, depth: int) -> None:
        nonlocal nodes, raw_bytes
        nodes += 1
        if nodes > _STATE_DIGEST_MAX_NODES or depth > _STATE_DIGEST_MAX_DEPTH:
            raise FixtureManifestError("Recovery state traversal exceeds its structural bounds.")
        if item is None:
            emit(b"null")
            return
        if type(item) is bool:
            emit(b"bool", b"\x01" if item else b"\x00")
            return
        if type(item) is int:
            emit(b"int", scalar_key(item))
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise FixtureManifestError("Recovery state contains a non-finite scalar.")
            emit(b"float", struct.pack(">d", item))
            return
        if type(item) is str:
            encoded = item.encode("utf-8")
            if len(encoded) > _STATE_DIGEST_MAX_STRING_BYTES:
                raise FixtureManifestError("Recovery state contains an oversized string.")
            emit(b"string", encoded)
            return
        array = array_value(item)
        if array is not None:
            domain, normalized = array
            if normalized.dtype.hasobject or normalized.dtype.kind not in "biuf":
                raise FixtureManifestError("Recovery state contains an unsupported array dtype.")
            if not bool(np.isfinite(normalized).all()):
                raise FixtureManifestError("Recovery state contains a non-finite array value.")
            raw = normalized.tobytes(order="C")
            raw_bytes += len(raw)
            if raw_bytes > _STATE_DIGEST_MAX_RAW_BYTES or normalized.ndim > _STATE_DIGEST_MAX_DEPTH:
                raise FixtureManifestError("Recovery state array exceeds its bounds.")
            emit(domain, normalized.dtype.str.encode("ascii"))
            emit(b"shape", b"".join(int(size).to_bytes(8, "big") for size in normalized.shape))
            emit(b"raw", raw)
            return
        if isinstance(item, Mapping):
            if len(item) > _STATE_DIGEST_MAX_SEQUENCE:
                raise FixtureManifestError("Recovery state mapping exceeds its bounds.")
            identifier = id(item)
            if identifier in active:
                raise FixtureManifestError("Recovery state contains a cycle.")
            active.add(identifier)
            try:
                entries = sorted((scalar_key(key), entry) for key, entry in item.items())
                emit(b"mapping", len(entries).to_bytes(8, "big"))
                for key, entry in entries:
                    emit(b"key", key)
                    visit(entry, depth + 1)
            finally:
                active.remove(identifier)
            return
        if type(item) in (list, tuple):
            if len(item) > _STATE_DIGEST_MAX_SEQUENCE:
                raise FixtureManifestError("Recovery state sequence exceeds its bounds.")
            identifier = id(item)
            if identifier in active:
                raise FixtureManifestError("Recovery state contains a cycle.")
            active.add(identifier)
            try:
                emit(b"list" if type(item) is list else b"tuple", len(item).to_bytes(8, "big"))
                for entry in item:
                    visit(entry, depth + 1)
            finally:
                active.remove(identifier)
            return
        raise FixtureManifestError("Recovery state contains an unsupported value.")

    visit(value, 0)
    return digest.hexdigest()


def _state_placement(value: object) -> tuple[str, ...]:
    """Return sorted native tensor placements without including them in state identity."""

    placements: set[str] = set()

    def visit(item: object) -> None:
        if type(item).__module__.split(".", 1)[0] == "torch":
            try:
                placements.add(str(item.device).lower())
                return
            except AttributeError:
                pass
        if isinstance(item, Mapping):
            for entry in item.values():
                visit(entry)
        elif type(item) in (list, tuple):
            for entry in item:
                visit(entry)

    visit(value)
    return tuple(sorted(placements))


def _path(value: str, name: str) -> str:
    """Validate one nonempty authority path without opening or creating it."""

    if type(value) is not str or not value:
        raise FixtureManifestError(f"Qualification worker request lacks {name} authority.")
    return value


@dataclass(frozen=True, slots=True)
class RecoveryControl:
    """Deterministic opt-in W3 interruption/recovery control.

    Args:
        interrupt_after_step: Exact retained optimizer step where the worker stops.
        fail_boundary: Experiment publication boundary that fails after a durable
            Artifact result while its history row remains pending.  Recovery must
            repair that row before a resumed optimizer update.

    The control is request data, not a callback or framework handle.  Real worker
    implementations install it only after reconstructing their selected workload.
    """

    interrupt_after_step: int = 64
    fail_boundary: str = "artifact_completed"

    def __post_init__(self) -> None:
        """Reject recovery controls outside the fixed KTD11 case."""

        if self.interrupt_after_step != 64 or self.fail_boundary != "artifact_completed":
            raise FixtureManifestError("Recovery qualification must interrupt W3 at step 64 after a durable Artifact result.")

    def to_data(self) -> dict[str, object]:
        """Encode the fixed interruption control without executable state."""

        return {"interrupt_after_step": self.interrupt_after_step, "fail_boundary": self.fail_boundary}

    @classmethod
    def from_data(cls, value: object) -> "RecoveryControl":
        """Decode the closed fixed recovery control."""

        if not isinstance(value, Mapping) or set(value) != {"interrupt_after_step", "fail_boundary"}:
            raise FixtureManifestError("Qualification recovery control is malformed.")
        return cls(value["interrupt_after_step"], value["fail_boundary"])


@dataclass(frozen=True, slots=True)
class QualificationWorkerRequest:
    """Immutable JSON-only routing request for one qualification worker.

    ``resource_mode`` describes selected runtime allocation/session controls only.
    It never disables the managed Experiment/Artifact operation lifecycle.
    """

    case: QualificationCase
    manifest_path: str
    tfds_data_dir: str
    output_store: str
    work_dir: str
    evidence_dir: str
    control_store: str
    execution_backend: str
    resource_mode: str
    ray_address: str | None = None
    recovery: RecoveryControl | None = None
    gate_id: str = "cpu-matrix"

    def __post_init__(self) -> None:
        """Validate route identity and detach every request field from callers."""

        if not isinstance(self.case, QualificationCase):
            raise FixtureManifestError("Qualification worker request lacks a closed case.")
        for name in ("manifest_path", "tfds_data_dir", "output_store", "work_dir", "evidence_dir", "control_store"):
            object.__setattr__(self, name, _path(getattr(self, name), name.replace("_", " ")))
        expected_backend, expected_mode = _ROUTES[self.case.execution]
        if (self.execution_backend, self.resource_mode) != (expected_backend, expected_mode):
            raise FixtureManifestError("Qualification worker route disagrees with its matrix execution mode.")
        if self.case.execution == "ray":
            if type(self.ray_address) is not str or not self.ray_address:
                raise QualificationUnrun("Same-host Ray qualification requires DRYML_STAGE5_7_RAY_ADDRESS.")
        elif self.ray_address is not None:
            raise FixtureManifestError("Only Ray matrix requests may carry a Ray address.")
        if self.recovery is not None:
            if not isinstance(self.recovery, RecoveryControl) or (
                    self.case.workload, self.case.framework, self.case.execution) != ("W3", "torch", "subprocess"):
                raise FixtureManifestError("Only the Torch W3 subprocess case may carry recovery controls.")
        expected_gate = "recovery-step64" if self.recovery is not None else "cpu-matrix"
        if self.gate_id != expected_gate:
            raise FixtureManifestError("Qualification request gate identity disagrees with its recovery control.")

    @property
    def request_id(self) -> str:
        """Return the stable identity of this complete portable worker request."""

        return hashlib.sha256(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()

    def to_data(self) -> dict[str, object]:
        """Encode this request using only definitions, references, paths, and facts."""

        return {
            "case": self.case.to_data(), "manifest_path": self.manifest_path,
            "tfds_data_dir": self.tfds_data_dir, "output_store": self.output_store,
            "work_dir": self.work_dir, "evidence_dir": self.evidence_dir,
            "control_store": self.control_store, "execution_backend": self.execution_backend,
            "resource_mode": self.resource_mode, "ray_address": self.ray_address,
            "recovery": None if self.recovery is None else self.recovery.to_data(),
            "gate_id": self.gate_id,
        }

    @classmethod
    def from_data(cls, value: object) -> "QualificationWorkerRequest":
        """Decode a request before any Store/framework reconstruction occurs."""

        if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
            raise FixtureManifestError("Qualification worker request is malformed.")
        return cls(
            QualificationCase.from_data(value["case"]), value["manifest_path"], value["tfds_data_dir"],
            value["output_store"], value["work_dir"], value["evidence_dir"], value["control_store"],
            value["execution_backend"], value["resource_mode"], value["ray_address"],
            None if value["recovery"] is None else RecoveryControl.from_data(value["recovery"]), value["gate_id"],
        )


def worker_request(manifest, case: QualificationCase, *, manifest_path, tfds_data_dir,
                   output_store, work_dir, evidence_dir, control_store,
                   ray_address: str | None = None,
                   recovery: RecoveryControl | None = None) -> QualificationWorkerRequest:
    """Build one selected primary or recovery request without probing other cells.

    Missing Ray configuration therefore classifies only the selected Ray cell as
    :class:`QualificationUnrun`; local and subprocess requests remain constructible.
    """

    if case not in cpu_matrix(manifest):
        raise FixtureManifestError("Qualification worker request must select a primary matrix case.")
    return QualificationWorkerRequest(
        case, os.fspath(manifest_path), os.fspath(tfds_data_dir), os.fspath(output_store),
        os.fspath(work_dir), os.fspath(evidence_dir), os.fspath(control_store),
        *_ROUTES[case.execution], ray_address if case.execution == "ray" else None,
        recovery=recovery, gate_id="recovery-step64" if recovery is not None else "cpu-matrix",
    )


def worker_requests(manifest, *, manifest_path, tfds_data_dir, output_store, work_dir,
                    evidence_dir, control_store, ray_address: str | None = None) -> tuple[QualificationWorkerRequest, ...]:
    """Return every constructible primary request without starting workers or loading data.

    Callers that omit a Ray address receive all 18 non-Ray requests.  They can use
    :func:`worker_request` to truthfully classify each of the six Ray cells.
    """

    requests = tuple(
        worker_request(
            manifest, case, manifest_path=manifest_path, tfds_data_dir=tfds_data_dir,
            output_store=output_store, work_dir=work_dir, evidence_dir=evidence_dir,
            control_store=control_store, ray_address=ray_address,
        )
        for case in cpu_matrix(manifest)
        if case.execution != "ray" or ray_address is not None
    )
    if len({request.case.case_id for request in requests}) != len(requests):
        raise AssertionError("Worker matrix request identities must be unique.")
    if len({request.request_id for request in requests}) != len(requests):
        raise AssertionError("Worker matrix request identities must be unique.")
    return requests


def recovery_request(manifest, *, manifest_path, tfds_data_dir, output_store, work_dir,
                     evidence_dir, control_store) -> QualificationWorkerRequest:
    """Build the separate fixed Torch W3 subprocess recovery request."""

    case = next(case for case in cpu_matrix(manifest) if (
        case.workload, case.framework, case.execution) == ("W3", "torch", "subprocess"))
    return worker_request(
        manifest, case, manifest_path=manifest_path, tfds_data_dir=tfds_data_dir,
        output_store=output_store, work_dir=work_dir, evidence_dir=evidence_dir,
        control_store=control_store, recovery=RecoveryControl(),
    )


def preflight_worker_request(request: QualificationWorkerRequest) -> None:
    """Reject unavailable shared authority before Core Execute submits a workload.

    This intentionally checks only selected filesystem authority.  Manifest/TFDS
    content validation remains worker-local immediately before real construction.
    """

    if not isinstance(request, QualificationWorkerRequest):
        raise TypeError("request must be QualificationWorkerRequest")
    for name in ("manifest_path", "tfds_data_dir", "control_store"):
        if not Path(getattr(request, name)).is_dir() and name != "manifest_path":
            raise QualificationUnrun(f"Qualification worker {name.replace('_', ' ')} is unavailable.")
    if not Path(request.manifest_path).is_file() or not Path(request.case.fixture_store).is_dir():
        raise QualificationUnrun("Qualification worker shared manifest or fixture Store is unavailable.")


def preflight_coordinator_request(request: QualificationWorkerRequest) -> None:
    """Run non-materializing coordinator preflight under the orchestrator floor."""

    with _coordinator_scope():
        _preflight_coordinator_request(request)


def _preflight_coordinator_request(request: QualificationWorkerRequest) -> None:
    """Complete all coordinator checks before creating a case child or submission.

    Manifest/environment, TFDS content, W3 cache metadata authority, shared Store opening,
    and all six authority roots are checked as prerequisites.  A prerequisite
    failure remains retryable ``QualificationUnrun`` because no case directory,
    spool, or backend submission has been created at this point.
    """

    preflight_worker_request(request)
    try:
        manifest = load_manifest(
            request.manifest_path, fixture_store=request.case.fixture_store,
            tfds_data_dir=request.tfds_data_dir, environment=installed_environment(),
        )
        if manifest.digest != request.case.manifest_digest:
            raise FixtureManifestError("Qualification worker request manifest digest drifted.")
        # The orchestrator validates manifest-bound bytes and Store receipts but
        # deliberately does not import TFDS.  The worker reconstructs TFDS before
        # opening payloads through preflight_manifest().
        from dryml.core.store.dir import DirStore
        from .stage5_7_fixtures import validate_tfds_content_authority
        validate_tfds_content_authority(manifest.tfds)
        fixture_store = DirStore.open_existing(manifest.fixture_store)
        try:
            validate_fixture_reference_metadata(manifest.references, store=fixture_store)
        finally:
            fixture_store.close()
        from .stage5_7_workloads import _paths_overlap, _safe_qualification_root
        roots = tuple(_safe_qualification_root(value, name) for value, name in (
            (request.output_store, "output Store"), (request.work_dir, "work"),
            (request.evidence_dir, "evidence"), (request.control_store, "control Store"),
            (request.case.fixture_store, "fixture Store"), (request.tfds_data_dir, "TFDS"),
        ))
        if any(_paths_overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1:]):
            raise FixtureManifestError("Qualification control, output, fixture Store, and TFDS roots must be disjoint.")
        for root, name in zip(roots[:4], ("output Store", "work", "evidence", "control Store")):
            if not os.access(root, os.W_OK | os.X_OK):
                raise QualificationUnrun(f"Qualification {name} is not writable.")
        control = DirStore.open_existing(request.control_store)
        try:
            # Opening the existing Store validates its format without creating or
            # repairing an authority path during coordinator preflight.
            pass
        finally:
            control.close()
    except QualificationUnrun:
        raise
    except FixtureManifestError:
        raise
    except (OSError, ValueError) as error:
        raise FixtureManifestError(f"Qualification coordinator authority is corrupt or incomplete: {error}") from error


def inspect_worker_transport(request: QualificationWorkerRequest) -> Mapping[str, object]:
    """Return closed request facts after proving no live worker input was retained."""

    data = _closed_json(request.to_data())
    assert isinstance(data, dict)
    decoded = QualificationWorkerRequest.from_data(data)
    if decoded != request:
        raise FixtureManifestError("Qualification worker request did not survive closed transport.")
    return data


def submit_worker_request(executor, request: QualificationWorkerRequest):
    """Preflight and submit one JSON-only request through a Core Execute facade.

    The narrow executor seam makes routine tests inspect all 24 submissions without
    importing ML frameworks.  Production callers pass an existing ``CoreExecutor``;
    no Ray service is created here.
    """

    preflight_worker_request(request)
    payload = inspect_worker_transport(request)
    submit = getattr(executor, "submit", None)
    if not callable(submit):
        raise TypeError("executor must expose Core Execute submit().")
    return submit(run_worker_request, dict(payload))


def run_real_worker_request(data: Mapping[str, object]) -> Mapping[str, object]:
    """Run one opted-in workload after reconstructing all selected authority locally.

    This is intentionally reached only through an isolated child or Core Execute
    worker. It imports real ML/test payload code after manifest authority is
    validated and returns provisional closed evidence. It never publishes final
    qualification evidence; that is the coordinator's non-replacing authority.
    """

    request = QualificationWorkerRequest.from_data(data)
    preflight_worker_request(request)
    from .stage5_7_fixtures import installed_environment, load_manifest, preflight_manifest
    from .stage5_7_workloads import QualificationEvidence, validate_evidence
    from tests.qualification.test_stage5_7_local import _real_runner

    try:
        manifest = load_manifest(
            request.manifest_path, fixture_store=request.case.fixture_store,
            tfds_data_dir=request.tfds_data_dir, environment=installed_environment(),
        )
        preflight_manifest(manifest)
    except QualificationUnrun as error:
        raise FixtureManifestError(
            f"Qualification worker prerequisite drifted after coordinator preflight: {error}"
        ) from error
    previous = {
        name: os.environ.get(name)
        for name in (
            "DRYML_STAGE5_7_CASE_OUTPUT_STORE", "DRYML_STAGE5_7_CASE_WORK_DIR",
            "DRYML_STAGE5_7_CASE_CONTROL_STORE",
        )
    }
    try:
        os.environ["DRYML_STAGE5_7_CASE_OUTPUT_STORE"] = request.output_store
        os.environ["DRYML_STAGE5_7_CASE_WORK_DIR"] = request.work_dir
        os.environ["DRYML_STAGE5_7_CASE_CONTROL_STORE"] = request.control_store
        evidence = _real_runner(
            manifest, request.case, recovery=request.recovery,
            worker_request_id=request.request_id,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not isinstance(evidence, QualificationEvidence):
        raise FixtureManifestError("Qualification worker did not return closed evidence.")
    return evidence.to_data()


def execute_real_worker_request(request: QualificationWorkerRequest):
    """Submit one real request through existing Core Execute without provisioning Ray.

    The coordinator owns only temporary reconstructed handles needed by Core
    Execute.  The submitted callable receives ``request.to_data()`` exclusively;
    shared state/control Stores are reopened in the worker.  Ray configuration is
    an attachment to the caller-provided address and never uses a local startup.
    """

    from dryml.core import Repo
    from dryml.core.execute import CoreOptions, Executor
    from dryml.core.store.dir import DirStore
    from dryml.execute.ray import RayBackendConfig
    from dryml.execute.subprocess import SubProcessConfig
    from .stage5_7_workloads import QualificationEvidence, qualification_case_paths

    with _coordinator_scope():
        preflight_coordinator_request(request)
        paths = qualification_case_paths(
            request.case, output_store_root=request.output_store, work_root=request.work_dir,
            evidence_root=request.evidence_dir, tfds_root=request.tfds_data_dir,
            gate_id=request.gate_id, create=True,
        )
        request = replace(
            request, output_store=os.fspath(paths.output_store), work_dir=os.fspath(paths.work_dir),
            evidence_dir=os.fspath(paths.evidence_dir),
        )
        if request.case.execution in {"local", "managed-local"}:
            evidence, observation, child_pid = _run_in_process_request(request)
            observed = _observed_local_evidence(evidence, request, observation, child_pid)
            return _finalize_coordinator_evidence(manifest=None, evidence=observed, request=request)

        spool = Path(request.work_dir) / "core-execute-spool"
        spool.mkdir(exist_ok=False)
        repo = Repo((DirStore(request.output_store), DirStore(request.case.fixture_store)))
        control = DirStore.open_existing(request.control_store)
        backend = (
            RayBackendConfig(address=request.ray_address, spool_directory=spool)
            if request.case.execution == "ray"
            else SubProcessConfig(spool_directory=spool)
        )
        executor = Executor(backend, core=CoreOptions(repo=repo, control_store=control, return_objects=False))
        try:
            future = executor.submit(run_real_worker_request, request.to_data())
            evidence = QualificationEvidence.from_data(future.result(timeout=300))
            observed = _observed_core_evidence(evidence, request, future.snapshot())
            future.cleanup(timeout=30)
            return _finalize_coordinator_evidence(manifest=None, evidence=observed, request=request)
        finally:
            executor.close(cancel=True, timeout=30)
            repo.close(flush=False)
            control.close()


@contextmanager
def _coordinator_scope():
    """Apply the product orchestrator materialization floor around coordination."""

    from dryml import session
    from dryml.runtime import RuntimeMode, active_runtime, materialization_scope

    entered = False
    if active_runtime().mode is RuntimeMode.NONE:
        session.set_mode("orchestrator")
        entered = True
    elif active_runtime().mode is not RuntimeMode.ORCHESTRATOR:
        raise QualificationUnrun("Qualification coordinator requires an orchestrator runtime.")
    try:
        with materialization_scope("strict"):
            yield
    finally:
        if entered:
            session.reset()


@contextmanager
def _coordinator_result_scope():
    """Allow only the selected final receipt/history/scalar result inspections.

    The enclosing coordinator remains in ``RuntimeMode.ORCHESTRATOR``.  Callers
    must first complete metadata-only exact-reference validation, then use this
    scope solely for the final Experiment receipt, ExperimentData history, and
    scalar Artifact result.  It restores the strict coordinator floor on every
    exit and is not an authority to load model or Dataset payloads.
    """

    from dryml.runtime import materialization_scope

    with materialization_scope("warn"):
        yield


@contextmanager
def _request_environment(request: QualificationWorkerRequest):
    """Install only case-local worker paths while preserving the caller environment."""

    names = {
        "DRYML_STAGE5_7_CASE_OUTPUT_STORE": request.output_store,
        "DRYML_STAGE5_7_CASE_WORK_DIR": request.work_dir,
        "DRYML_STAGE5_7_CASE_CONTROL_STORE": request.control_store,
        "DRYML_STAGE5_7_CASE_RESOURCE_MODE": request.resource_mode,
    }
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(names)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_in_process_request(request: QualificationWorkerRequest):
    """Run local semantics in a fresh isolation process, not the coordinator.

    ``local`` and ``managed-local`` retain their direct local execution semantics;
    the extra process is solely a harness isolation boundary.  Subprocess and Ray
    remain Core Execute backends and never take this path.
    """

    code = (
        "import json, sys\n"
        "from tests.qualification.stage5_7_workers import _run_local_child_request\n"
        "print(json.dumps(_run_local_child_request(json.loads(sys.argv[1])), sort_keys=True))\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [
            str(Path.cwd() / "src"), str(Path.cwd()), os.environ.get("PYTHONPATH"),
        ])),
    }
    subprocess = __import__("subprocess")
    process = subprocess.Popen(
        [sys.executable, "-c", code, json.dumps(request.to_data())], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise FixtureManifestError("Isolated local qualification child timed out.") from error
    if process.returncode:
        raise FixtureManifestError(f"Isolated local qualification child failed: {stderr.strip()}")
    try:
        result = json.loads(stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FixtureManifestError("Isolated local qualification child returned malformed provisional evidence.") from error
    if not isinstance(result, Mapping) or set(result) != {"evidence", "observation"}:
        raise FixtureManifestError("Isolated local qualification child omitted its execution observation.")
    from .stage5_7_workloads import QualificationEvidence

    try:
        evidence = QualificationEvidence.from_data(result["evidence"])
        observation = _validate_local_observation(result["observation"], request, process.pid)
    except (TypeError, ValueError, FixtureManifestError) as error:
        raise FixtureManifestError("Isolated local qualification child returned invalid execution observation.") from error
    return evidence, observation, process.pid


def _run_local_child_request(data: Mapping[str, object]) -> Mapping[str, object]:
    """Execute one semantic-local request inside its fresh harness child."""

    from dryml import session

    request = QualificationWorkerRequest.from_data(data)
    if request.case.execution not in {"local", "managed-local"}:
        raise FixtureManifestError("Only local semantic requests may enter the isolation child.")
    session.reset()
    try:
        if request.case.execution == "managed-local":
            session.manage(cpus=1, gpus=0)
        with _request_environment(request):
            evidence = run_real_worker_request(request.to_data())
            snapshot = session.current()
            from dryml.runtime import active_runtime

            observation = {
                "pid": os.getpid(), "semantic_mode": request.case.execution,
                "session_mode": snapshot.mode,
                "runtime_mode": active_runtime().mode.value,
                "has_allocation": snapshot.allocation is not None,
                "allocation": (
                    type(snapshot.allocation).__name__
                    if snapshot.allocation is not None else "no-session-allocation"
                ),
            }
            return {"evidence": evidence, "observation": observation}
    finally:
        session.reset()


def run_local_isolation_probe(data: Mapping[str, object]) -> Mapping[str, object]:
    """Return framework-free proof of one fresh local semantic child.

    Routine tests use this probe instead of a real workload.  It exercises the
    same session/allocation branch as :func:`_run_local_child_request` while
    keeping optional packages and fixture payloads out of the test.
    """

    from dryml import session

    request = QualificationWorkerRequest.from_data(data)
    if request.case.execution not in {"local", "managed-local"}:
        raise FixtureManifestError("Only local semantic requests may enter the isolation probe.")
    session.reset()
    try:
        if request.case.execution == "managed-local":
            session.manage(cpus=1, gpus=0)
        return {
            "pid": os.getpid(), "session_allocation": session.current().allocation is not None,
            "semantic": run_worker_request(request.to_data()),
        }
    finally:
        session.reset()


def _validate_local_observation(observation, request: QualificationWorkerRequest, child_pid: int):
    """Validate child-observed local runtime facts before coordinator evidence use."""

    required = {
        "pid", "semantic_mode", "session_mode", "runtime_mode", "has_allocation",
        "allocation",
    }
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise FixtureManifestError("Local qualification child observation is malformed.")
    managed = request.case.execution == "managed-local"
    expected = {
        "pid": child_pid,
        "semantic_mode": request.case.execution,
        "session_mode": "managed" if managed else "python",
        "runtime_mode": "inline" if managed else "none",
        "has_allocation": managed,
    }
    if any(observation[name] != value for name, value in expected.items()):
        raise FixtureManifestError("Local qualification child observation disagrees with its actual session mode.")
    if type(observation["allocation"]) is not str or not observation["allocation"]:
        raise FixtureManifestError("Local qualification child allocation observation is malformed.")
    if managed == (observation["allocation"] == "no-session-allocation"):
        raise FixtureManifestError("Local qualification child allocation observation is inconsistent.")
    return dict(observation)


def _observed_local_evidence(evidence, request: QualificationWorkerRequest, observation, child_pid: int):
    """Bind semantic-local evidence to the coordinator-observed isolation child."""

    from dataclasses import replace as data_replace

    worker = dict(evidence.worker)
    worker.update({
        "execution_backend": "isolation-process",
        "semantic_backend": "local",
        "isolation": {"kind": "fresh-os-process", "parent_pid": os.getpid()},
        "worker_pid": observation["pid"],
        "worker_identity": f"pid:{observation['pid']}",
        "runtime_allocation": {
            "resource_mode": "managed-session" if observation["has_allocation"] else "unmanaged-session",
            "allocation": observation["allocation"],
            "admission": f"session:{observation['session_mode']}/runtime:{observation['runtime_mode']}",
        },
        "submitted": {
            "request_digest": request.request_id, "submission_id": f"isolation:{observation['pid']}",
            "result_ref": evidence.final_experiment_ref.digest(),
        },
        "shared_authority": {
            "fixture_store": os.fspath(Path(request.case.fixture_store).resolve()),
            "control_store": os.fspath(Path(request.control_store).resolve()),
        },
        "core_outcome": {"kind": "not-core-local", "publication_refs": (), "update_refs": ()},
        "coordinator_validated_refs": None,
    })
    return data_replace(evidence, worker=worker)


def _observed_core_evidence(evidence, request: QualificationWorkerRequest, snapshot):
    """Bind returned evidence to the Core Future's observed worker/admission facts."""

    from dataclasses import replace as data_replace

    backend = snapshot.backend
    worker_id, pid = backend.worker_id, backend.pid
    if not isinstance(worker_id, str) or not worker_id or type(pid) is not int or pid <= 0:
        raise FixtureManifestError("Core Execute completed without worker identity evidence.")
    observed_backend = "ray" if worker_id.startswith("ray:") else "subprocess" if worker_id.startswith("subprocess:") else None
    if observed_backend is None:
        raise FixtureManifestError("Core Execute worker identity does not identify the selected backend.")
    if snapshot.evidence is None or backend.report is None:
        raise FixtureManifestError("Core Execute completed without authority/admission evidence.")
    if request.resource_mode != "worker-process-no-session-allocation" or backend.allocation is not None:
        raise FixtureManifestError("Core worker session allocation disagrees with its submitted route facts.")
    worker = dict(evidence.worker)
    worker.update({
        "execution_backend": observed_backend,
        "semantic_backend": observed_backend,
        "isolation": {"kind": "core-worker", "parent_pid": os.getpid()},
        "worker_pid": pid,
        "worker_identity": worker_id,
        "runtime_allocation": {
            "resource_mode": "worker-process-no-session-allocation",
            "allocation": "worker-process-no-session-allocation",
            "admission": "admitted-without-session-allocation",
        },
        "submitted": {
            "request_digest": request.request_id, "submission_id": backend.submission_id,
            "result_ref": evidence.final_experiment_ref.digest(),
        },
        "shared_authority": {
            "fixture_store": os.fspath(Path(request.case.fixture_store).resolve()),
            "control_store": os.fspath(Path(request.control_store).resolve()),
        },
        "core_outcome": {
            "kind": observed_backend,
            "publication_refs": tuple(item.state_ref.digest() for item in snapshot.evidence.publications),
            "update_refs": tuple(item.digest() for item in snapshot.evidence.updates),
        },
        "coordinator_validated_refs": None,
    })
    return data_replace(evidence, worker=worker)


def _validate_coordinator_reference_metadata(evidence, *, output_store) -> None:
    """Validate every final exact reference without restoring any payload.

    Exact-load plans establish StateRef closure, projected/reference bindings,
    definitions, snapshot metadata, and selected-Store availability.  Model and
    Dataset references intentionally stop here: a coordinator never restores
    their native payloads.
    """

    from dryml.core import Repo
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.dir import DirStore

    output = DirStore.open_existing(output_store)
    fixture = DirStore.open_existing(evidence.case.fixture_store)
    repo = Repo((output, fixture))
    try:
        expected_model = evidence.final_experiment_ref.at("model")
        expected_test = evidence.final_experiment_ref.reference_value_at("test_data")
        if expected_model != evidence.model_ref or expected_test != evidence.test_ref:
            raise FixtureManifestError("Evidence model/test bindings disagree with final Experiment StateRef.")
        for reference in (
                evidence.final_experiment_ref, evidence.model_ref, evidence.test_ref,
                evidence.history_ref, evidence.artifact_ref):
            reference.object_projection()
            build_exact_state_load_plan(repo, reference, _defer_payload=True)
    except Exception as error:
        if isinstance(error, FixtureManifestError):
            raise
        raise FixtureManifestError("Qualification StateRef metadata authority is absent or corrupt.") from error
    finally:
        repo.close(flush=False)


def _load_coordinator_results(evidence, *, output_store):
    """Restore only history and the scalar Artifact result for a final receipt.

    The final Experiment receipt stays as its already metadata-validated exact
    StateRef because restoring an Experiment can materialize its model edge. This
    is the sole coordinator materialization exception and receives no model or
    Dataset reference.
    """

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    output = DirStore.open_existing(output_store)
    fixture = DirStore.open_existing(evidence.case.fixture_store)
    repo = Repo((output, fixture))
    try:
        history, artifact = _load_selected_coordinator_results(
            repo, history_ref=evidence.history_ref, artifact_ref=evidence.artifact_ref,
        )
        return evidence.final_experiment_ref, history, artifact
    except Exception as error:
        raise FixtureManifestError("Selected coordinator result authority cannot be restored.") from error
    finally:
        repo.close(flush=False)


def _load_selected_coordinator_results(repo, *, history_ref, artifact_ref):
    """Load exactly the two coordinator-approved payload result references.

    The Experiment receipt remains a StateRef metadata value. This narrow seam
    makes it impossible for coordinator payload loading to receive the Experiment,
    model, or Dataset references at all.
    """

    import warnings

    from dryml.artifacts import Fold, Value
    from dryml.core.symbol import ImportRef
    from dryml.models import ExperimentData

    def root_import(reference):
        candidate = reference.definition.cls
        if isinstance(candidate, ImportRef):
            return candidate
        if isinstance(candidate, type):
            return ImportRef.from_object(candidate)
        raise FixtureManifestError("Qualification result reference has unsupported root metadata.")

    def admit(reference, allowed, label):
        if root_import(reference) not in allowed:
            raise FixtureManifestError(
                f"Qualification {label} reference is not an approved result-compatible type."
            )

    admit(history_ref, {ImportRef.from_object(ExperimentData)}, "history")
    admit(
        artifact_ref,
        {ImportRef.from_object(Value), ImportRef.from_object(Fold)},
        "Artifact",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with _coordinator_result_scope():
            history = repo.load_state_ref(history_ref, reuse_live="never", cache="none")
            artifact = repo.load_state_ref(artifact_ref, reuse_live="never", cache="none")
    return history, artifact


def _validate_loaded_coordinator_results(evidence, final_receipt, history, artifact) -> None:
    """Validate the three explicitly restored lightweight coordinator results."""

    from dryml.artifacts import Value
    from dryml.models import ExperimentData

    if final_receipt != evidence.final_experiment_ref:
        raise FixtureManifestError("Qualification final receipt is not the metadata-validated Experiment StateRef.")
    if not isinstance(history, ExperimentData):
        raise FixtureManifestError("Qualification history receipt is not ExperimentData.")
    if not isinstance(artifact, Value) or not artifact.ready:
        raise FixtureManifestError("Qualification Artifact receipt is not a ready Value.")
    if history.last_state_ref != evidence.history_ref or artifact.last_state_ref != evidence.artifact_ref:
        raise FixtureManifestError("Qualification result receipt does not equal selected Store authority.")


def _finalize_coordinator_evidence(*, manifest, evidence, request: QualificationWorkerRequest):
    """Validate and non-replacing-publish final coordinator-owned evidence."""

    from dataclasses import replace as data_replace
    from .stage5_7_workloads import QualificationEvidence, validate_evidence
    from .stage5_7_fixtures import load_manifest

    if manifest is None:
        manifest = load_manifest(
            request.manifest_path, fixture_store=request.case.fixture_store,
            tfds_data_dir=request.tfds_data_dir, environment=installed_environment(),
        )
    refs = {
        "experiment": evidence.final_experiment_ref.digest(), "model": evidence.model_ref.digest(),
        "test": evidence.test_ref.digest(), "history": evidence.history_ref.digest(),
        "artifact": evidence.artifact_ref.digest(),
    }
    _validate_coordinator_reference_metadata(evidence, output_store=request.output_store)
    final_receipt, history, artifact = _load_coordinator_results(
        evidence, output_store=request.output_store,
    )
    _validate_loaded_coordinator_results(evidence, final_receipt, history, artifact)
    worker = dict(evidence.worker)
    worker["coordinator_validated_refs"] = {
        "output_store": os.fspath(Path(request.output_store).resolve()), "refs": refs,
    }
    accepted = data_replace(evidence, worker=worker)
    validate_request_evidence(request, accepted)
    validate_evidence(manifest, accepted, output_store=request.output_store, coordinator_results=(final_receipt, history, artifact))
    _publish_final_evidence(Path(request.evidence_dir) / "qualification-evidence.json", accepted)
    try:
        reopened = QualificationEvidence.from_data(json.loads(
            (Path(request.evidence_dir) / "qualification-evidence.json").read_text(encoding="ascii")
        ))
    except (OSError, json.JSONDecodeError, FixtureManifestError) as error:
        raise FixtureManifestError("Coordinator final evidence cannot be reopened.") from error
    if reopened != accepted:
        raise FixtureManifestError("Reopened final qualification evidence differs from accepted evidence.")
    return accepted


def _publish_final_evidence(path: Path, evidence) -> None:
    """Atomically publish final evidence once without accepting worker artifacts."""

    import tempfile

    if path.exists():
        raise FixtureManifestError("Refusing to replace existing final qualification evidence.")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(json.dumps(evidence.to_data(), sort_keys=True, separators=(",", ":")))
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FixtureManifestError("Refusing to replace existing final qualification evidence.") from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    """Synchronize a final evidence directory entry after hard-link publication."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise FixtureManifestError("Final qualification evidence directory cannot be synchronized.") from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_request_evidence(request: QualificationWorkerRequest, evidence) -> None:
    """Reject a closed result whose request, authority, or final receipt was substituted."""

    if evidence.case != request.case:
        raise FixtureManifestError("Qualification result case does not equal its submitted request.")
    submitted = evidence.worker["submitted"]
    authority = evidence.worker["shared_authority"]
    expected_authority = {
        "fixture_store": os.fspath(Path(request.case.fixture_store).resolve()),
        "control_store": os.fspath(Path(request.control_store).resolve()),
    }
    if submitted["request_digest"] != request.request_id or authority != expected_authority:
        raise FixtureManifestError("Qualification result request or shared Store authority was substituted.")
    if submitted["result_ref"] != evidence.final_experiment_ref.digest():
        raise FixtureManifestError("Qualification result receipt is not bound to its final Experiment reference.")


def run_worker_request(data: Mapping[str, object]) -> Mapping[str, object]:
    """Reconstruct a portable request inside a selected worker.

    This entry point intentionally performs no training until the opt-in real
    runner calls it.  Its return facts let routine process-boundary tests prove
    fresh-process construction without loading datasets or optional frameworks.
    """

    request = QualificationWorkerRequest.from_data(data)
    return {
        "request_id": request.request_id, "case_id": request.case.case_id,
        "worker_pid": os.getpid(), "execution_backend": request.execution_backend,
        "resource_mode": request.resource_mode, "recovery": request.recovery is not None,
        "optional_modules": tuple(sorted(name for name in sys.modules if name in {
            "tensorflow", "tensorflow_datasets", "torch", "pandas",
        })),
    }


def validate_recovery_report(report: Mapping[str, object]) -> None:
    """Validate the fixed recovery ordering before accepting real evidence.

    A real runner records the retained step, restored model/optimizer/progress and
    exposure/loss-window facts, then the failed Artifact repair before the next
    optimizer update.  Contradictions are failures, never an unrun success.
    """

    required = {"checkpoint_step", "before", "restored", "events", "final_refs"}
    if not isinstance(report, Mapping) or set(report) != required or report["checkpoint_step"] != 64:
        raise FixtureManifestError("Recovery evidence lacks the retained step-64 checkpoint.")
    fields = {"model_digest", "model_placement", "optimizer_digest", "optimizer_placement", "optimizer_iterations", "epoch", "next_batch", "step", "examples_seen", "loss_numerator", "loss_denominator", "checkpoint_ref", "history_occurrence", "artifact_status"}
    if not isinstance(report["before"], Mapping) or not isinstance(report["restored"], Mapping) or set(report["before"]) != fields or report["before"] != report["restored"]:
        raise FixtureManifestError("Recovery evidence does not prove exact restored training state.")
    events = report["events"]
    if not isinstance(events, (list, tuple)) or tuple(events)[:2] != ("artifact_repaired", "optimizer_update:65"):
        raise FixtureManifestError("Recovery performed an optimizer update before repairing failed Artifact evaluation.")
    if not isinstance(report["final_refs"], Mapping) or set(report["final_refs"]) != {"experiment", "model", "test", "history", "artifact"} or not all(isinstance(value, str) and value for value in report["final_refs"].values()):
        raise FixtureManifestError("Recovery evidence lacks exact final reference receipts.")


__all__ = [
    "QualificationWorkerRequest", "RecoveryControl", "inspect_worker_transport",
    "preflight_coordinator_request", "preflight_worker_request", "recovery_request", "run_worker_request",
    "run_real_worker_request", "execute_real_worker_request", "submit_worker_request",
    "validate_recovery_report", "validate_request_evidence", "worker_request", "worker_requests",
]
