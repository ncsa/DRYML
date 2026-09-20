"""Dependency-free contract checks for the optional Execute Ray backend."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from threading import Event, Lock
from time import monotonic
from types import SimpleNamespace

import pytest

import dryml.execute.ray as ray_module
from dryml.execute._protocol import (
    BootstrapDescriptor, Correlation, FrameState, FrameType, OwnerEnvelopeType,
    ProtocolConversation, decode_exact_frame, encode_control, encode_frame,
    encode_owner_envelope,
)
from dryml.execute.errors import AdmissionError
from dryml.execute.output import ExecutionOutput
from dryml.execute.admission import _admit_observed_logical
from dryml.execute.models import (EnvironmentCandidate, ResourceAmounts,
                                  WorkerSetup)
from dryml.execute.ray import (RayBackendConfig, RayFuture, _logical_world,
                               _native_error, _native_options,
                               _requested_amounts, _resource_amounts)
from dryml.environments.specs import PythonExecutableSpec
from dryml.environments import CondaEnvironmentSpec, CurrentEnvironmentSpec
from dryml.environments.records import EnvironmentRecord, PythonRecord
from dryml.environments.selection import resolve_environment_spec
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
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


def test_ray_runtime_environment_honors_selector_pythonpath_policy():
    """
    Ray receives selector PYTHONPATH controls instead of silently dropping
    them.
    """
    candidate = EnvironmentCandidate(
        "candidate",
        PythonExecutableSpec("/existing/python",
                             pythonpath_policy="explicit",
                             extra_pythonpath=("/one", "/two")),
        None,
        None,
        True,
        (),
    )
    backend = RayBackendConfig().create_backend()

    runtime = backend._runtime_environment(
        {"py_executable": "/existing/python"}, candidate)

    assert runtime["env_vars"] == {
        "PYTHONPATH": os.pathsep.join(("/one", "/two")),
    }


def test_ray_exact_current_pin_stays_lazy_and_conda_run_fails_before_submission(  # noqa: E501
):
    """
    Ray admits supported pinned forms or rejects launcher semantics without SDK
    work.
    """
    backend = RayBackendConfig().create_backend()
    current = resolve_environment_spec(CurrentEnvironmentSpec())

    candidate, runtime = backend._select_environment(
        SimpleNamespace(environment_spec=current, environment=None))
    assert candidate.record == current.record
    assert runtime == {"py_executable": current.command[0]}

    conda_run = replace(
        current,
        spec=CondaEnvironmentSpec(prefix="/existing", launch_mode="conda-run"),
        command=("conda", "run", "-p", "/existing", "--no-capture-output",
                 "--", "python"),
        resolved_prefix="/existing",
    )
    with pytest.raises(Exception, match="launcher form"):
        backend._select_environment(
            SimpleNamespace(environment_spec=conda_run, environment=None))


def test_ray_exact_pin_world_plan_is_keyed_to_its_selected_candidate(
        monkeypatch):
    """
    Keep exact selector and logical-world feasibility evidence associated.
    """
    selection = resolve_environment_spec(CurrentEnvironmentSpec())
    backend = RayBackendConfig(
        automatic_environment_discovery=False,
    ).create_backend()
    resources = SimpleNamespace(
        total=ResourceAmounts(1, None, {}, {}), complete=True,
    )
    world = WorldRequirement({
        "main": RoleRequirement(
            resources=ResourceRequirement(cpus=CountConstraint(1, 1)),
        ),
    })
    monkeypatch.setattr(backend, "_resources_until",
                        lambda _deadline: resources)

    selected = backend.discover(
        environment_spec=selection, world=world, timeout=1,
    )
    unpinned = backend.discover(world=world, timeout=1)

    assert [plan.environment_key for plan in selected.plans] == [
        selected.environments[0].key,
    ]
    assert [plan.environment_key for plan in unpinned.plans] == [None]


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


class _HandshakeAuthority:
    """
    Reject unexpected grant confirmation before a failed exact-pin handshake.
    """

    def __init__(self):
        self.confirmations = 0

    def confirm_grant(self, *args, **kwargs):
        """Record an invalid post-rejection grant attempt."""
        del args, kwargs
        self.confirmations += 1
        raise AssertionError("selection mismatch must not confirm a Ray grant")


class _UnreadPayload:
    """
    Fail if a rejected Ray handshake tries to open invocation payload bytes.
    """

    def __init__(self):
        self.reads = 0

    @property
    def path(self):
        """Reject any pre-authorization payload-path access."""
        self.reads += 1
        raise AssertionError("selection mismatch must not read the payload")


class _HandshakeSocket(_SetupSocket):
    """Reuse the frame recorder for a handshake that must stop before GO."""


def _handshake_hello(descriptor, *, protocol=None):
    """Create one valid fake-Ray HELLO for pre-payload mismatch tests."""
    return decode_exact_frame(encode_control(
        FrameState.HELLO, descriptor.correlation,
        {
            "token": descriptor.rendezvous_token,
            "protocol": ray_module.WORKER_PROTOCOL_ID if protocol is None else protocol,
            "dill": ray_module.dill.__version__,
            "implementation": sys.implementation.name,
            "python": list(sys.version_info[:2]),
            "pid": 1234,
            "worker_id": "ray:worker",
            "native": {
                "node_id": "node",
                "worker_id": "worker",
                "task_id": "task",
                "ray": ray_module._RAY_VERSION,
                "python": list(sys.version_info[:3]),
                "process_create_time": 1.0,
                "assigned_resources": {},
                "accelerator_ids": {},
            },
        }, header_limit=65_536,
    ), header_limit=65_536, payload_limit=65_536)


def test_fake_ray_rejects_old_v3_worker_before_go_or_payload():
    """An old Ray worker identity cannot receive owner controls or workload bytes."""
    backend = RayBackendConfig().create_backend()
    descriptor = BootstrapDescriptor(
        Correlation("old-v3-ray", 0, 1), "token", "127.0.0.1", 43123,
        65_536, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 65_536,
    )
    hello = _handshake_hello(descriptor, protocol="dryml.execute.worker.v3")
    socket = _HandshakeSocket()
    future = RayFuture("old-v3-ray", output=ExecutionOutput(), termination_timeout=1)
    run = ray_module._Run(future, socket, SimpleNamespace())
    conversation = backend._conversation(descriptor)
    conversation.accept_frame(hello)

    with pytest.raises(AdmissionError, match="runtime is incompatible"):
        backend._handshake(
            SimpleNamespace(), future, run, descriptor, None,
            SimpleNamespace(node_id="node"), hello, conversation,
        )

    assert socket.frames == []


@pytest.mark.parametrize(
    "mismatch", ("missing", "executable", "prefix", "base_prefix", "software"))
def test_fake_ray_selection_handshake_rejects_before_go_payload_or_grant(
        monkeypatch, mismatch):
    """
    Reject absent or incompatible exact-pin evidence without authorizing work.
    """
    selection = resolve_environment_spec(CurrentEnvironmentSpec())
    backend = RayBackendConfig().create_backend()
    descriptor = BootstrapDescriptor(
        Correlation("ray-pin-reject", 0, 1), "token", "127.0.0.1", 43123,
        65_536, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 65_536,
    )
    hello = _handshake_hello(descriptor)
    ready = decode_exact_frame(encode_control(
        FrameState.READY, descriptor.correlation,
        {"ready": True, "pid": 1234, "worker_id": "ray:worker"},
        header_limit=65_536,
    ), header_limit=65_536, payload_limit=65_536)
    observed: EnvironmentRecord | None = selection.record
    if mismatch == "executable":
        observed = replace(observed,
                           python=replace(observed.python,
                                          executable="/wrong/python"))
    elif mismatch == "prefix":
        observed = replace(observed,
                           python=replace(observed.python,
                                          prefix="/wrong/prefix"))
    elif mismatch == "base_prefix":
        observed = replace(observed,
                           python=replace(observed.python,
                                          base_prefix="/wrong/base"))
    elif mismatch == "software":
        observed = replace(observed,
                           python=PythonRecord(
                               version="0.0.0",
                               implementation=observed.python.implementation,
                               executable=observed.python.executable,
                               prefix=observed.python.prefix,
                               base_prefix=observed.python.base_prefix,
                           ))
    if mismatch == "missing":
        evidence = decode_exact_frame(encode_owner_envelope(
            FrameState.READY, descriptor.correlation, OwnerEnvelopeType.WORLD,
            b"{}", header_limit=65_536, owner_limit=1_000_000,
        ), header_limit=65_536, payload_limit=1_000_000)
    else:
        assert observed is not None
        evidence = decode_exact_frame(encode_owner_envelope(
            FrameState.READY, descriptor.correlation,
            OwnerEnvelopeType.ENVIRONMENT,
            canonical_json_bytes(
                observed.to_data(), max_entries=65_536, max_nodes=65_536,
            ), header_limit=65_536,
            owner_limit=1_000_000,
        ), header_limit=65_536, payload_limit=1_000_000)

    class Reader:
        """
        Supply only the worker's readiness and environment-evidence frames.
        """

        def __init__(self, *_args, **_kwargs):
            self.frames = iter((ready, evidence))

        def read(self):
            """Return exactly one ordered fake worker frame."""
            return next(self.frames)

    socket = _HandshakeSocket()
    output = ExecutionOutput()
    future = RayFuture("ray-pin-reject", output=output, termination_timeout=5)
    run = ray_module._Run(future, socket, SimpleNamespace())
    authority = _HandshakeAuthority()
    backend._authority = authority
    payload = _UnreadPayload()
    call = SimpleNamespace(
        environment=None, environment_spec=selection, world=None,
        admission_deadline=monotonic() + 5, payload=payload,
    )
    conversation = backend._conversation(descriptor)
    conversation.accept_frame(hello)
    monkeypatch.setattr(ray_module, "SocketFrameReader", Reader)

    with pytest.raises(AdmissionError):
        backend._handshake(
            call, future, run, descriptor, None,
            SimpleNamespace(node_id="node"), hello, conversation,
        )

    sent = [
        decode_exact_frame(frame, header_limit=65_536, payload_limit=1_000_000)
        for frame in socket.frames
    ]
    assert all(frame.state not in {FrameState.GO, FrameState.PAYLOAD}
               for frame in sent)
    assert payload.reads == 0
    assert authority.confirmations == 0
    assert not run.issued


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

    assert backend._send_setup(call, future, run, descriptor, _setup_conversation(correlation), reader, None, {
        "assigned_resources": {"CPU": 1.0, "memory": 4096},
        "accelerator_ids": {"GPU": ["0"]},
    })
    sent = decode_exact_frame(socket.frames[0], header_limit=512, payload_limit=512)
    assert sent.state is FrameState.SETUP
    assert b'"kind":"ray"' in sent.payload
    grant = canonical_json_load_bytes(sent.payload, max_depth=64, max_nodes=65_536, max_entries=65_536, max_string=512, max_int_bits=4096)["context"]["native_grant"]
    assert grant["memory_bytes"] == 4096
    assert grant["accelerator_ids"] == {"GPU": ("0",)}
    assert output.snapshot().stdout == "setup output"


def test_fake_ray_setup_failure_publishes_validated_terminal_before_withholding_payload(monkeypatch):
    """A fake Ray setup terminal publishes failure without sending a workload payload."""
    backend = RayBackendConfig(control_header_limit_bytes=512, owner_envelope_limit_bytes=512, admission_message_limit_bytes=512).create_backend()
    correlation = Correlation("ray-failure", 0, 1)
    descriptor = BootstrapDescriptor(correlation, "token", "127.0.0.1", 43123, 512, 512, 512, 512, 512, 128)
    socket = _SetupSocket()
    output = ExecutionOutput()
    output._bind("ray-failure", output_limit_bytes=512, live_output_queue_limit_bytes=512, stream_output=False, start_live=False)
    future = RayFuture("ray-failure", output=output, termination_timeout=5)
    run = SimpleNamespace(
        connection=socket, native_node_id="node", native_task_id="task",
        worker_id="ray:worker", outcome_validated=False, lock=Lock(),
        deadline_expired=False, outcome_claimed=False, cancelling=False,
        cancellation_starting=False, terminal_event=Event(),
    )
    publish_exception = future._publish_exception

    def assert_validated_before_terminal(error):
        """Prevent Future publication from reopening the validated-terminal race."""
        assert run.outcome_validated
        publish_exception(error)

    monkeypatch.setattr(future, "_publish_exception", assert_validated_before_terminal)
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
