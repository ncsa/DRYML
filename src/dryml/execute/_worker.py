"""Dependency-light one-shot worker entry for the private Execute protocol."""

from __future__ import annotations

import argparse
import base64
import importlib
import os
import secrets
import socket
import sys
import threading
import time
from collections.abc import Mapping
from functools import lru_cache

import dill

from dryml.environments import EnvironmentRecord, EnvironmentRequirement, inspect_current
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from dryml.worlds import ProcessSpec, ResourceSpec, RoleSpec, WorldAllocation, WorldRequirement, WorldSpec

from ._protocol import WORKER_PROTOCOL_ID, BootstrapDescriptor, FrameError, FrameState, FrameType, OwnerEnvelopeType, ProtocolConversation, SocketFrameReader, decode_bootstrap_descriptor, decode_control, decode_owner_envelopes, encode_control, encode_frame, encode_owner_envelope, encode_worker_error
from ._spooling import deserialize_call, serialize_result
from .admission import _admit_observed_logical, admit
from .models import WorkerSetupContext


def main(argv: list[str] | None = None) -> int:
    """Run one descriptor-only worker and return a process exit code.

    The worker validates HELLO/PREPARE/GO before it reads or deserializes payload
    bytes. It sends captured descriptor output over the private loopback socket and
    never writes invocation or result spools to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True)
    namespace = parser.parse_args(argv)
    try:
        encoded = base64.urlsafe_b64decode(namespace.bootstrap.encode("ascii"))
        # The descriptor is a fixed scalar envelope.  Its own configured limit
        # governs the first socket frame rather than an unrelated worker default.
        descriptor = decode_bootstrap_descriptor(encoded, header_limit=len(encoded))
        _run(descriptor)
    except BaseException:
        return 1
    return 0


def _run(descriptor: BootstrapDescriptor, *, worker_id: str | None = None, native: Mapping[str, object] | None = None) -> None:
    """Exchange one checked handshake, then invoke exactly once after GO."""
    connection = socket.create_connection((descriptor.rendezvous_host, descriptor.rendezvous_port), timeout=5)
    send_lock = threading.Lock()
    conversation: ProtocolConversation | None = None
    try:
        conversation = _conversation(descriptor)
        worker_id = worker_id or f"subprocess:{secrets.token_hex(16)}"
        hello_control: dict[str, object] = {
            "dill": dill.__version__, "implementation": sys.implementation.name,
            "pid": os.getpid(), "protocol": WORKER_PROTOCOL_ID,
            "python": list(sys.version_info[:2]), "token": descriptor.rendezvous_token,
            "worker_id": worker_id,
        }
        # HELLO controls have a closed shape.  Subprocess retains its established
        # shape while the Ray bootstrap explicitly opts into native evidence.
        if native is not None:
            hello_control["native"] = dict(native)
        hello = encode_control(FrameState.HELLO, descriptor.correlation, hello_control, header_limit=descriptor.control_header_limit_bytes)
        _send(connection, send_lock, hello)
        conversation.accept(hello)
        reader = SocketFrameReader(connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=_limits(descriptor))
        frame = reader.read()
        conversation.accept_frame(frame)
        control = decode_control(frame, limit_bytes=descriptor.control_header_limit_bytes, required_keys={"cwd", "deadline", "owners"})
        deadline = control["deadline"]
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or time.monotonic() >= deadline:
            return
        cwd = control["cwd"]
        if not isinstance(cwd, str):
            return
        requested_owners = control["owners"]
        if not isinstance(requested_owners, list) or len(requested_owners) != len(set(requested_owners)):
            return
        try:
            owners = tuple(OwnerEnvelopeType(owner) for owner in requested_owners)
        except (TypeError, ValueError):
            return
        received = []
        for _ in owners:
            owner_frame = reader.read()
            conversation.accept_frame(owner_frame)
            received.append(owner_frame)
        envelopes = decode_owner_envelopes(received, state=FrameState.PREPARE, correlation=descriptor.correlation, owner_limit=descriptor.owner_envelope_limit_bytes, aggregate_limit=descriptor.admission_message_limit_bytes)
        if set(envelopes) != set(owners):
            return
        environment: EnvironmentRequirement | None = None
        world: WorldRequirement | None = None
        allocation: WorldAllocation | None = None
        if OwnerEnvelopeType.ENVIRONMENT in envelopes:
            environment = EnvironmentRequirement.from_data(_json(envelopes[OwnerEnvelopeType.ENVIRONMENT], descriptor.owner_envelope_limit_bytes))
        if OwnerEnvelopeType.WORLD in envelopes:
            world = WorldRequirement.from_data(_json(envelopes[OwnerEnvelopeType.WORLD], descriptor.owner_envelope_limit_bytes))
            if native is None and OwnerEnvelopeType.ALLOCATION not in envelopes:
                _send_error(connection, send_lock, descriptor, conversation, "AdmissionError")
                return
            if OwnerEnvelopeType.ALLOCATION in envelopes:
                allocation = WorldAllocation.from_data(_json(envelopes[OwnerEnvelopeType.ALLOCATION], descriptor.owner_envelope_limit_bytes))
        # Exact selections require fresh identity evidence even with no
        # software requirement; the coordinator compares it before GO and
        # payload transfer.
        record = (inspect_current() if environment is not None
                  or OwnerEnvelopeType.SELECTION in envelopes else None)
        if allocation is not None:
            allocation = _apply_allocation(allocation, environment_id=None if record is None else record.semantic_id)
        os.chdir(cwd)
        if native is not None and world is not None:
            logical_world, controls = _ray_logical_world(world, native)
            decision = _admit_observed_logical(
                environment=environment, world=world, record=record,
                observed_world=logical_world, controls=controls, deadline=deadline,
            )
        else:
            decision = admit(environment=environment, world=world, record=record, allocation=allocation, deadline=deadline)
        if not decision.go:
            _send_error(connection, send_lock, descriptor, conversation, "AdmissionError")
            return
        ready = encode_control(FrameState.READY, descriptor.correlation, {"pid": os.getpid(), "ready": True, "worker_id": worker_id}, header_limit=descriptor.control_header_limit_bytes)
        _send(connection, send_lock, ready)
        conversation.accept(ready)
        if environment is not None or OwnerEnvelopeType.SELECTION in envelopes:
            assert record is not None
            record_data = canonical_json_bytes(record.to_data(), max_depth=32, max_nodes=65536, max_entries=4096, max_string=descriptor.owner_envelope_limit_bytes, max_int_bits=64)
            evidence = encode_owner_envelope(FrameState.READY, descriptor.correlation, OwnerEnvelopeType.ENVIRONMENT, record_data, header_limit=descriptor.control_header_limit_bytes, owner_limit=descriptor.owner_envelope_limit_bytes)
            _send(connection, send_lock, evidence)
            conversation.accept(evidence)
        if allocation is not None:
            allocation_data = canonical_json_bytes(allocation.to_data(), max_depth=32, max_nodes=65536, max_entries=4096, max_string=descriptor.owner_envelope_limit_bytes, max_int_bits=64)
            evidence = encode_owner_envelope(FrameState.READY, descriptor.correlation, OwnerEnvelopeType.ALLOCATION, allocation_data, header_limit=descriptor.control_header_limit_bytes, owner_limit=descriptor.owner_envelope_limit_bytes)
            _send(connection, send_lock, evidence)
            conversation.accept(evidence)
        go_frame = reader.read()
        conversation.accept_frame(go_frame)
        if go_frame.state is FrameState.STOP:
            return
        permit = decode_control(go_frame, limit_bytes=descriptor.control_header_limit_bytes, required_keys={"deadline", "permit"})
        if permit["permit"] != descriptor.rendezvous_token:
            return
        workload_deadline = permit["deadline"]
        if workload_deadline is not None and (not isinstance(workload_deadline, (int, float)) or isinstance(workload_deadline, bool) or time.monotonic() >= workload_deadline):
            return
        # GO begins the configured execution period. A disabled execution deadline
        # deliberately leaves setup and invocation without an admission timeout.
        connection.settimeout(None)
        first_frame = reader.read()
        conversation.accept_frame(first_frame)
        if workload_deadline is not None and time.monotonic() >= workload_deadline:
            return
        if first_frame.state is FrameState.SETUP:
            if first_frame.owner is not OwnerEnvelopeType.SETUP:
                return
            setup = _setup_data(
                first_frame.payload, descriptor.owner_envelope_limit_bytes,
                descriptor.admission_message_limit_bytes,
            )
            _invoke(
                connection, send_lock, descriptor, None, workload_deadline,
                setup=setup, reader=reader, conversation=conversation,
            )
        elif first_frame.state is FrameState.PAYLOAD and first_frame.frame_type is FrameType.PAYLOAD:
            _invoke(connection, send_lock, descriptor, first_frame.payload, workload_deadline)
    except BaseException as exc:
        # The coordinator receives only a bounded exception type, never raw worker
        # arguments, environment values, tokens, or traceback paths.
        if conversation is not None:
            try:
                _send_error(connection, send_lock, descriptor, conversation, type(exc).__name__[:128])
            except BaseException:
                pass
    finally:
        connection.close()


def run_ray_bootstrap(encoded_descriptor: bytes) -> dict[str, str]:
    """Run one Ray-isolated Execute worker and return only a completion marker.

    Args:
        encoded_descriptor: The closed, bounded bootstrap descriptor supplied as
            the sole native Ray task argument.

    Returns:
        A closed bootstrap receipt naming the exact Ray task and worker after the
        common Execute channel has reached its terminal worker boundary.

    Raises:
        BaseException: Propagates bootstrap validation or worker failures to the
            Ray task.  User payload values never enter Ray's task arguments.

    Side Effects:
        Opens the descriptor's loopback channel and invokes at most one accepted
        payload.  Ray is imported only inside this native task entry point.
    """
    from dryml.execute._protocol import decode_bootstrap_descriptor
    import psutil
    import ray

    if not isinstance(encoded_descriptor, bytes):
        raise TypeError("Ray bootstrap descriptor must be bytes")
    descriptor = decode_bootstrap_descriptor(encoded_descriptor, header_limit=len(encoded_descriptor))
    context = ray.get_runtime_context()
    worker = str(context.get_worker_id())
    node = str(context.get_node_id())
    task = str(context.get_task_id())
    native = {
        "accelerator_ids": context.get_accelerator_ids(),
        "assigned_resources": context.get_assigned_resources(),
        "node_id": node,
        "process_create_time": psutil.Process().create_time(),
        "python": list(sys.version_info[:3]),
        "ray": ray.__version__,
        "task_id": task,
        "worker_id": worker,
    }
    _run(descriptor, worker_id=f"ray:{worker}", native=native)
    return {"marker": "dryml.execute.ray.bootstrap.v1", "node_id": node, "task_id": task, "worker_id": worker}


def _ray_logical_world(requirement: WorldRequirement, native: Mapping[str, object]) -> tuple[WorldSpec | None, dict[str, str]]:
    """Reconstruct one Ray logical grant for owner checks without exact CPU IDs.

    The Ray scheduler reports process-level logical quantities and assigned GPU
    identifiers.  It never exposes an exact CPU affinity allocation, so this
    helper deliberately returns a :class:`WorldSpec`, not ``WorldAllocation``.
    """
    if len(requirement.roles) != 1:
        return None, {}
    assigned = native.get("assigned_resources")
    accelerator_ids = native.get("accelerator_ids")
    if not isinstance(assigned, Mapping) or not isinstance(accelerator_ids, Mapping):
        return None, {}
    role_name = next(iter(requirement.roles))
    cpu = _logical_count(assigned.get("CPU"))
    memory = _logical_count(assigned.get("memory"))
    gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("gpu"))
    gpu = tuple(gpu_ids) if isinstance(gpu_ids, (list, tuple)) and not isinstance(gpu_ids, (str, bytes)) else ()
    granted_gpu = _logical_count(assigned.get("GPU"))
    accelerators: dict[str, int] = {}
    controls: dict[str, str] = {}
    if cpu is not None:
        controls["cpus"] = "logical"
    if memory is not None:
        controls["memory"] = "logical"
    if granted_gpu is not None and len(gpu) == granted_gpu:
        accelerators["gpu"] = granted_gpu
        controls["accelerators"] = "logical"
    resources = ResourceSpec(cpus=cpu or 0, memory=memory, accelerators=accelerators)
    return WorldSpec({role_name: RoleSpec(1, ProcessSpec(resources))}, backend={"kind": "ray", "logical_grant": True}), controls


def _logical_count(value: object) -> int | None:
    """Return a nonnegative integral Ray logical quantity or unavailable evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or int(value) != value:
        return None
    return int(value)


def _invoke(
    connection: socket.socket,
    send_lock: threading.Lock,
    descriptor: BootstrapDescriptor,
    payload: bytes | None,
    deadline: float | None,
    *,
    setup: dict[str, object] | None = None,
    reader: SocketFrameReader | None = None,
    conversation: ProtocolConversation | None = None,
) -> None:
    """Capture setup/workload output and preserve outcomes across setup teardown.

    Setup resolution and entry occur only after GO, under redirected descriptors.
    A setup-bearing worker acknowledges readiness before it reads payload bytes, so
    the coordinator can prove a pre-ack failure withheld deserialization.
    """
    original = {1: os.dup(1), 2: os.dup(2)}
    readers: list[threading.Thread] = []
    sequences = {"stdout": 0, "stderr": 0}
    outcome: bytes | None = None
    outcome_type: FrameType | None = None
    cleanup_issues: list[str] = []
    manager: object | None = None
    entered = False
    workload_error: BaseException | None = None
    deadline_elapsed = False
    try:
        for fd, stream in ((1, "stdout"), (2, "stderr")):
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
            drainer = threading.Thread(target=_drain, args=(read_fd, stream, sequences, connection, send_lock, descriptor), daemon=True)
            drainer.start()
            readers.append(drainer)
        if setup is not None:
            try:
                manager = _resolve_setup(setup)
                enter = getattr(manager, "__enter__")
                exit_ = getattr(manager, "__exit__")
                if not callable(enter) or not callable(exit_):
                    raise TypeError("worker setup factory did not return a context manager")
                enter()
                entered = True
                assert conversation is not None and reader is not None
                ready = encode_control(FrameState.SETUP_READY, descriptor.correlation, {"ready": True}, header_limit=descriptor.control_header_limit_bytes)
                _send(connection, send_lock, ready)
                conversation.accept(ready)
                payload_frame = reader.read()
                conversation.accept_frame(payload_frame)
                if payload_frame.state is not FrameState.PAYLOAD or payload_frame.frame_type is not FrameType.PAYLOAD:
                    raise FrameError("worker payload frame is invalid")
                payload = payload_frame.payload
            except BaseException as exc:
                workload_error = exc
        if workload_error is None:
            try:
                if payload is None:
                    raise TimeoutError("execution payload is unavailable before deserialization")
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_elapsed = True
                    raise _PreInvocationDeadline("execution deadline elapsed before payload deserialization")
                fn, args, kwargs = deserialize_call(payload, limit_bytes=descriptor.invocation_limit_bytes)
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_elapsed = True
                    raise _PreInvocationDeadline("execution deadline elapsed before invocation")
                value = fn(*args, **kwargs)
                outcome = serialize_result(
                    value,
                    limit_bytes=_setup_result_payload_limit(descriptor.result_limit_bytes)
                    if setup is not None else descriptor.result_limit_bytes,
                )
                outcome_type = FrameType.RESULT
            except BaseException as exc:
                workload_error = exc
        if entered:
            try:
                assert manager is not None
                suppressed = manager.__exit__(
                    None if workload_error is None else type(workload_error),
                    workload_error,
                    None if workload_error is None else workload_error.__traceback__,
                )
                if workload_error is not None and suppressed:
                    workload_error = RuntimeError("worker setup suppressed an execution failure")
            except BaseException as exc:
                cleanup_issues.append(type(exc).__name__[:128])
        if workload_error is not None:
            outcome = encode_worker_error(
                None if deadline_elapsed else type(workload_error).__name__[:128],
                deadline_elapsed=deadline_elapsed,
                cleanup_types=cleanup_issues,
                setup=setup is not None,
                limit_bytes=descriptor.result_limit_bytes,
            )
            outcome_type = FrameType.ERROR
    finally:
        _flush_standard_streams()
        for fd in (1, 2):
            os.dup2(original[fd], fd)
            os.close(original[fd])
        _flush_standard_streams()
        # Final fences are emitted only by a drainer that observed EOF.  A child
        # retaining a descriptor therefore leaves capture incomplete, not forged.
        finish_at = time.monotonic() + descriptor.output_final_timeout
        for reader in readers:
            reader.join(max(0.0, finish_at - time.monotonic()))
        if outcome is not None and outcome_type is not None:
            state = FrameState.RESULT if outcome_type is FrameType.RESULT else FrameState.ERROR
            terminal = outcome if setup is None or outcome_type is FrameType.ERROR else _terminal_envelope(outcome, cleanup_issues, descriptor.result_limit_bytes)
            _send(connection, send_lock, encode_frame(state, outcome_type, descriptor.correlation, terminal, header_limit=descriptor.control_header_limit_bytes))


class _PreInvocationDeadline(TimeoutError):
    """Distinguish worker-owned pre-invocation expiry from callable failures."""


def _setup_data(payload: bytes, owner_limit: int, admission_limit: int) -> dict[str, object]:
    """Decode one closed setup envelope before any factory import occurs."""
    limit = min(owner_limit, admission_limit)
    if len(payload) > owner_limit or len(payload) > admission_limit:
        raise FrameError("worker setup exceeds configured admission limits")
    value = canonical_json_load_bytes(payload, max_depth=64, max_nodes=65_536, max_entries=65_536, max_string=limit, max_int_bits=4096)
    if not isinstance(value, Mapping) or set(value) != {"context", "data", "factory"}:
        raise FrameError("worker setup envelope has an invalid shape")
    if not isinstance(value["factory"], str) or not isinstance(value["data"], Mapping) or not isinstance(value["context"], Mapping):
        raise FrameError("worker setup envelope has invalid fields")
    return {"factory": value["factory"], "data": dict(value["data"]), "context": dict(value["context"])}


def _resolve_setup(setup: Mapping[str, object]) -> object:
    """Import and construct one setup context manager after output capture starts."""
    factory_id = setup["factory"]
    context_data = setup["context"]
    data = setup["data"]
    if not isinstance(factory_id, str) or not isinstance(context_data, Mapping) or not isinstance(data, Mapping):
        raise FrameError("worker setup fields are invalid")
    module_name, separator, qualname = factory_id.partition(":")
    if not separator:
        raise FrameError("worker setup factory is invalid")
    factory: object = importlib.import_module(module_name)
    for attribute in qualname.split("."):
        factory = getattr(factory, attribute)
    if not callable(factory):
        raise TypeError("worker setup factory is not callable")
    context = _setup_context(context_data)
    return factory(context, dict(data))


def _setup_context(data: Mapping[str, object]) -> WorkerSetupContext:
    """Rebuild verified setup evidence without accepting caller allocation authority."""
    expected = {"allocation", "backend", "environment", "native_grant", "submission_id"}
    if set(data) != expected:
        raise FrameError("worker setup context has an invalid shape")
    environment = None if data["environment"] is None else EnvironmentRecord.from_data(data["environment"])
    allocation = None if data["allocation"] is None else WorldAllocation.from_data(data["allocation"])
    return WorkerSetupContext(
        submission_id=data["submission_id"], backend=data["backend"],
        environment=environment, allocation=allocation, native_grant=data["native_grant"],
    )


def _terminal_envelope(payload: bytes, cleanup_issues: list[str], limit: int) -> bytes:
    """Encode a setup result without letting teardown replace its value."""
    data: dict[str, object] = {
        "cleanup": [{"type": value} for value in cleanup_issues],
        "payload": base64.b64encode(payload).decode("ascii"),
    }
    try:
        encoded = canonical_json_bytes(data, max_depth=3, max_nodes=128, max_entries=64, max_string=limit, max_int_bits=64)
    except BaseException as exc:
        raise FrameError("worker setup terminal exceeds configured result limit") from exc
    if len(encoded) > limit:
        raise FrameError("worker setup terminal exceeds configured result limit")
    return encoded


def _setup_result_payload_limit(limit: int) -> int:
    """Reserve terminal-envelope space before serializing a setup-bearing result."""
    # A single setup exit can add one bounded type-only cleanup issue. Reserving
    # its JSON and base64 expansion keeps an already encoded result transportable.
    return max(1, ((limit - 256) * 3) // 4)


def setup_result_bytes_limit(limit: int) -> int:
    """Return the safely usable byte value for a setup-bearing generic result.

    Core execution transports an already encoded outcome as a ``bytes`` value.
    The generic worker serializes that value with dill before wrapping it in the
    setup terminal, so its usable budget reserves the serializer's maximum
    protocol-5 bytes framing overhead. Tiny configured limits return zero for
    caller-side rejection before a worker is launched.
    """
    payload_limit = _setup_result_payload_limit(limit)
    if payload_limit <= _setup_result_serialization_overhead():
        return 0
    return payload_limit - _setup_result_serialization_overhead()


@lru_cache(maxsize=1)
def _setup_result_serialization_overhead() -> int:
    """Verify dill's bounded bytes framing before any configured-budget preflight.

    Protocol-5 dill bytes framing has at most 18 bytes of overhead across short,
    16-bit, 32-bit, and larger-length opcode forms. Fixed representatives verify
    the owning serializer while limiting every dummy allocation to 65,536 bytes;
    the static ceiling safely covers lengths that cannot be probed cheaply.
    """
    ceiling = 18
    for size in (0, 1, 255, 256, 65_535, 65_536):
        encoded = serialize_result(b"\0" * size, limit_bytes=size + ceiling)
        if len(encoded) - size > ceiling:
            raise ValueError("result serializer bytes framing exceeds the setup budget")
    return ceiling


def _drain(read_fd: int, stream: str, sequences: dict[str, int], connection: socket.socket, send_lock: threading.Lock, descriptor: BootstrapDescriptor) -> None:
    """Continuously forward actual descriptor writes without retaining worker spools."""
    eof = False
    try:
        while True:
            data = os.read(read_fd, descriptor.output_frame_limit_bytes)
            if not data:
                eof = True
                return
            sequence = sequences[stream]
            sequences[stream] += 1
            _send(connection, send_lock, encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, descriptor.correlation, data, header_limit=descriptor.control_header_limit_bytes, stream=stream, sequence=sequence))
    finally:
        os.close(read_fd)
        if eof:
            _send(connection, send_lock, encode_control(FrameState.OUTPUT_FINAL, descriptor.correlation, {"next_sequence": sequences[stream], "stream": stream}, header_limit=descriptor.control_header_limit_bytes))


def _limits(descriptor: BootstrapDescriptor) -> dict[FrameType, int]:
    """Return the descriptor's effective typed receive bounds."""
    return {FrameType.CONTROL: descriptor.control_header_limit_bytes, FrameType.OWNER: descriptor.owner_envelope_limit_bytes, FrameType.PAYLOAD: descriptor.invocation_limit_bytes, FrameType.OUTPUT: descriptor.output_frame_limit_bytes, FrameType.RESULT: descriptor.result_limit_bytes, FrameType.ERROR: descriptor.result_limit_bytes}


def _conversation(descriptor: BootstrapDescriptor) -> ProtocolConversation:
    """Create the correlation-locked local conversation for this descriptor."""
    return ProtocolConversation(descriptor.correlation, header_limit=descriptor.control_header_limit_bytes, invocation_limit=descriptor.invocation_limit_bytes, result_limit=descriptor.result_limit_bytes, output_limit=descriptor.output_frame_limit_bytes, owner_limit=descriptor.owner_envelope_limit_bytes, admission_limit=descriptor.admission_message_limit_bytes)


def _send(connection: socket.socket, lock: threading.Lock, data: bytes) -> None:
    """Serialize socket writes from output drainers and the invocation thread."""
    with lock:
        connection.sendall(data)


def _send_error(connection: socket.socket, lock: threading.Lock, descriptor: BootstrapDescriptor, conversation: ProtocolConversation, error_type: str) -> None:
    """End failed admission with a bounded typed error before payload loading."""
    frame = encode_frame(FrameState.ERROR, FrameType.ERROR, descriptor.correlation, canonical_json_bytes({"type": error_type}, max_depth=1, max_nodes=2, max_entries=1, max_string=128, max_int_bits=64), header_limit=descriptor.control_header_limit_bytes)
    _send(connection, lock, frame)
    conversation.accept(frame)


def _flush_standard_streams() -> None:
    """Flush Python buffering on the descriptor currently active at the call boundary."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:
            pass


def _apply_allocation(allocation: WorldAllocation, *, environment_id: str | None) -> WorldAllocation:
    """Apply only verifiable local affinity/visibility controls before READY.

    The worker never pretends that inherited capacity is a grant.  Unsupported
    exact CPU or CUDA controls raise before payload loading and therefore become a
    typed admission failure at the coordinator.
    """
    processes = tuple(process for values in allocation.roles.values() for process in values)
    if len(processes) != 1:
        raise ValueError("one worker requires one process allocation")
    process = processes[0]
    controls: dict[str, str] = {}
    if process.cpus:
        if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
            raise ValueError("exact CPU affinity is unsupported on this platform")
        requested = set(process.cpus)
        os.sched_setaffinity(0, requested)
        if set(os.sched_getaffinity(0)) != requested:
            raise ValueError("exact CPU affinity could not be verified")
        controls["cpus"] = "applied"
    if process.accelerators:
        if set(process.accelerators) != {"gpu"} or any(not isinstance(device, int) for device in process.accelerators["gpu"]):
            raise ValueError("exact accelerator visibility is unsupported")
        visible = ",".join(str(device) for device in process.accelerators["gpu"])
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited not in {None, visible}:
            raise ValueError("CUDA visibility conflicts with the reserved allocation")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
            raise ValueError("exact accelerator visibility could not be verified")
        controls["accelerators"] = "applied"
    payload = allocation.to_payload()
    backend: dict[str, object] = {"kind": "subprocess", "execute_controls": controls}
    if environment_id is not None:
        backend["environment_id"] = environment_id
    payload["backend"] = backend
    return WorldAllocation.from_payload(payload)


def _json(data: bytes, limit: int) -> dict[str, object]:
    """Decode one owner envelope only after common framing aggregate validation."""
    value = canonical_json_load_bytes(data, max_depth=32, max_nodes=65536, max_entries=4096, max_string=limit, max_int_bits=64)
    if not isinstance(value, Mapping):
        raise FrameError("owner envelope is not a mapping")
    return dict(value)


if __name__ == "__main__":
    raise SystemExit(main())
