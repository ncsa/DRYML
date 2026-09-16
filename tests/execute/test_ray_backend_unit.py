"""Dependency-free contract checks for the optional Execute Ray backend."""

from __future__ import annotations

import sys
from time import monotonic
from types import SimpleNamespace

import pytest

from dryml.execute._protocol import BootstrapDescriptor, Correlation, FrameState, FrameType, ProtocolConversation, decode_exact_frame, encode_control, encode_frame
from dryml.execute.output import ExecutionOutput
from dryml.execute.admission import _admit_observed_logical
from dryml.execute.models import EnvironmentCandidate, WorkerSetup
from dryml.execute.ray import RayBackendConfig, RayFuture, _logical_world, _native_error, _native_options, _requested_amounts, _resource_amounts
from dryml.environments.specs import PythonExecutableSpec
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement


def test_ray_config_and_future_are_inert_without_importing_ray(monkeypatch):
    """Construction creates no SDK object and native access starts empty."""
    monkeypatch.delitem(sys.modules, "ray", raising=False)

    config = RayBackendConfig(address="auto")
    backend = config.create_backend()
    future = backend.create_future("submission", ExecutionOutput())

    assert isinstance(future, RayFuture)
    assert future.object_ref is None
    assert future.server_address is None
    assert future.node_id is None
    assert "ray" not in sys.modules


@pytest.mark.parametrize("address", ["ray://127.0.0.1:10001", "http://127.0.0.1:6379", "127.0.0.1", "127.0.0.1:0"])
def test_ray_config_rejects_non_endpoint_or_provisioning_forms(address):
    """Only the closed existing-server endpoint grammar is accepted."""
    with pytest.raises(ValueError):
        RayBackendConfig(address=address)


def test_ray_native_options_omit_unconstrained_defaults():
    """An unconstrained task leaves scheduler defaults untouched."""
    assert _native_options(None) == {}
    assert _requested_amounts(None).cpus == 1.0


def test_ray_native_options_request_a_valid_two_cpu_world():
    """A two-CPU logical world maps to Ray's explicit scheduler request."""
    world = WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(2, 2))),
    })

    assert _requested_amounts(world).cpus == 2.0
    assert _native_options(world)["num_cpus"] == 2.0


def test_ray_native_errors_never_include_native_exception_text():
    """Native exception values cannot expose credentials or filesystem paths."""
    class CredentialError(Exception):
        pass

    message = _native_error(
        "Ray initialization failed",
        CredentialError("token=dummy-secret path=/private/dryml/credentials uri=ray://user:pass@host"),
        256,
    )

    assert message == "Ray initialization failed (CredentialError)"
    assert len(message.encode("utf-8")) <= 256


def test_ray_config_rejects_native_device_visibility_override():
    """Ray scheduler-selected device visibility cannot be contradicted per task."""
    with pytest.raises(ValueError, match="device visibility"):
        RayBackendConfig(env_vars={"CUDA_VISIBLE_DEVICES": "0"})


def test_ray_runtime_environment_merges_candidate_then_explicit_overrides():
    """Selected runtime variables survive while explicit config wins collisions."""
    candidate = EnvironmentCandidate(
        "candidate", PythonExecutableSpec("/existing/python", env={"FROM_CANDIDATE": "candidate", "OVERRIDE": "candidate"}),
        None, None, True, (),
    )
    backend = RayBackendConfig(env_vars={"OVERRIDE": "config", "FROM_CONFIG": "config"}).create_backend()

    runtime = backend._runtime_environment({"py_executable": "/existing/python"}, candidate)

    assert runtime == {
        "py_executable": "/existing/python",
        "env_vars": {"FROM_CANDIDATE": "candidate", "OVERRIDE": "config", "FROM_CONFIG": "config"},
    }


def test_ray_resource_observation_preserves_unknown_dimensions():
    """Native observations never manufacture memory, CPU, or device identifiers."""
    amounts = _resource_amounts({"CPU": 2.0, "GPU": 1.0})

    assert amounts.cpus == 2.0
    assert amounts.memory_bytes is None
    assert amounts.accelerators == {"gpu": 1.0}
    assert _resource_amounts({}).cpus is None


def test_ray_resource_observation_keeps_named_scheduler_keys_out_of_accelerators():
    """Only Ray GPU is an accelerator; object-store and node keys stay named."""
    amounts = _resource_amounts({"CPU": 2.0, "node:192.168.2.31": 1.0, "object_store_memory": 4.0, "TPU": 3.0})

    assert amounts.accelerators == {}
    assert amounts.named == {"node:192.168.2.31": 1.0, "object_store_memory": 4.0, "TPU": 3.0}


def test_ray_logical_cpu_grant_passes_owner_checks_without_cpu_ids():
    """Ray logical grants admit count constraints without fabricating affinity IDs."""
    requirement = WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
    })
    observed, controls = _logical_world(requirement, {
        "assigned_resources": {"CPU": 1.0},
        "accelerator_ids": {},
    })

    decision = _admit_observed_logical(
        world=requirement, observed_world=observed, controls=controls,
        deadline=monotonic() + 1,
    )

    assert decision.go
    assert decision.allocation is None


class _SetupSocket:
    """Record backend frames while exposing the narrow timeout/write socket surface."""

    def __init__(self):
        self.frames: list[bytes] = []
        self.timeout = object()

    def sendall(self, data):
        """Retain one complete fake wire write."""
        self.frames.append(bytes(data))

    def settimeout(self, timeout):
        """Record the coordinator's post-GO socket policy."""
        self.timeout = timeout


class _SetupReader:
    """Return a deterministic fake-Ray worker frame sequence."""

    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        """Return the next prevalidated worker frame."""
        try:
            return next(self.frames)
        except StopIteration as exc:
            raise EOFError from exc


def _setup_conversation(correlation):
    """Advance a private conversation through GO for direct setup-path tests."""
    conversation = ProtocolConversation(correlation, header_limit=512, invocation_limit=512, result_limit=512, output_limit=128, owner_limit=512, admission_limit=512)
    for state, control in ((FrameState.HELLO, {"worker": "ray"}), (FrameState.PREPARE, {"owners": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, correlation, control, header_limit=512))
    return conversation


def test_fake_ray_setup_sends_native_evidence_and_drains_output_before_readiness():
    """Ray setup receives Ray-only grant evidence and preserves pre-payload output."""
    backend = RayBackendConfig(control_header_limit_bytes=512, owner_envelope_limit_bytes=512, admission_message_limit_bytes=512).create_backend()
    correlation = Correlation("ray-setup", 0, 1)
    descriptor = BootstrapDescriptor(correlation, "token", "127.0.0.1", 43123, 512, 512, 512, 512, 512, 128)
    socket = _SetupSocket()
    output = ExecutionOutput()
    output._bind("ray-setup", output_limit_bytes=512, live_output_queue_limit_bytes=512, stream_output=False, start_live=False)
    future = RayFuture("ray-setup", output=output, termination_timeout=5)
    run = SimpleNamespace(connection=socket, native_node_id="node", native_task_id="task", worker_id="ray:worker", outcome_validated=False)
    call = SimpleNamespace(worker_setup=WorkerSetup(factory="tests.execute.test_ray_backend_unit:setup_factory", data={}), submission_id="ray-setup", output=output)
    reader = _SetupReader((
        decode_exact_frame(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, correlation, b"setup output", header_limit=512, stream="stdout", sequence=0), header_limit=512, payload_limit=512),
        decode_exact_frame(encode_control(FrameState.SETUP_READY, correlation, {"ready": True}, header_limit=512), header_limit=512, payload_limit=512),
    ))

    assert backend._send_setup(call, future, run, descriptor, _setup_conversation(correlation), reader, None, {"assigned_resources": {"CPU": 1.0}})
    sent = decode_exact_frame(socket.frames[0], header_limit=512, payload_limit=512)
    assert sent.state is FrameState.SETUP
    assert b'"kind":"ray"' in sent.payload
    assert output.snapshot().stdout == "setup output"


def test_fake_ray_setup_failure_keeps_payload_withheld_and_cleanup_incomplete():
    """A fake Ray setup terminal publishes failure without sending a workload payload."""
    backend = RayBackendConfig(control_header_limit_bytes=512, owner_envelope_limit_bytes=512, admission_message_limit_bytes=512).create_backend()
    correlation = Correlation("ray-failure", 0, 1)
    descriptor = BootstrapDescriptor(correlation, "token", "127.0.0.1", 43123, 512, 512, 512, 512, 512, 128)
    socket = _SetupSocket()
    output = ExecutionOutput()
    output._bind("ray-failure", output_limit_bytes=512, live_output_queue_limit_bytes=512, stream_output=False, start_live=False)
    future = RayFuture("ray-failure", output=output, termination_timeout=5)
    run = SimpleNamespace(connection=socket, native_node_id="node", native_task_id="task", worker_id="ray:worker", outcome_validated=False)
    call = SimpleNamespace(worker_setup=WorkerSetup(factory="tests.execute.test_ray_backend_unit:setup_factory", data={}), submission_id="ray-failure", output=output)
    terminal = b'{"cleanup":[{"type":"RuntimeError"}],"type":"ValueError"}'
    reader = _SetupReader((
        decode_exact_frame(encode_frame(FrameState.ERROR, FrameType.ERROR, correlation, terminal, header_limit=512), header_limit=512, payload_limit=512),
    ))

    assert not backend._send_setup(call, future, run, descriptor, _setup_conversation(correlation), reader, None, {})
    assert future.snapshot().cleanup_state == "incomplete"
    assert len(socket.frames) == 1


def setup_factory(_context, _data):
    """Provide an importable inert factory identifier for fake-Ray transport tests."""
    from contextlib import nullcontext

    return nullcontext()
