"""Private bounded framing for the future Execute coordinator-worker channel.

The correlation fields guard accidental cross-submission delivery.  They do not
authenticate a peer or turn the trusted worker channel into a security sandbox.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import ipaddress
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import BinaryIO

from dryml.formats import CanonicalJSONError, canonical_json_bytes, canonical_json_load_bytes, json_ready


PROTOCOL_VERSION = 1
# This is deliberately a protocol identity, rather than a package version: a
# selected interpreter must execute the same worker implementation contract.
WORKER_PROTOCOL_ID = "dryml.execute.worker.v1"
_HEADER_LENGTH_BYTES = 4
_PAYLOAD_LENGTH_BYTES = 8
_MAX_HEADER_LENGTH = (1 << (_HEADER_LENGTH_BYTES * 8)) - 1
_MAX_PAYLOAD_LENGTH = (1 << (_PAYLOAD_LENGTH_BYTES * 8)) - 1
_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class FrameError(ValueError):
    """Report a malformed, oversized, mismatched, or out-of-order private frame."""


class FrameType(str, Enum):
    """Name the closed categories of private transport frames."""

    CONTROL = "control"
    OWNER = "owner"
    PAYLOAD = "payload"
    OUTPUT = "output"
    RESULT = "result"
    ERROR = "error"


class FrameState(str, Enum):
    """Name the closed one-shot coordinator-worker protocol phases."""

    HELLO = "hello"
    PREPARE = "prepare"
    READY = "ready"
    GO = "go"
    STOP = "stop"
    PAYLOAD = "payload"
    OUTPUT = "output"
    OUTPUT_FINAL = "output-final"
    RESULT = "result"
    ERROR = "error"


class OwnerEnvelopeType(str, Enum):
    """Name owner-encoded admission facts that retain their owner-specific codec."""

    ENVIRONMENT = "environment"
    WORLD = "world"
    ALLOCATION = "allocation"
    CONTROLS = "controls"


_STATE_TYPES = {
    FrameState.HELLO: {FrameType.CONTROL},
    FrameState.PREPARE: {FrameType.CONTROL, FrameType.OWNER},
    FrameState.READY: {FrameType.CONTROL, FrameType.OWNER},
    FrameState.GO: {FrameType.CONTROL},
    FrameState.STOP: {FrameType.CONTROL},
    FrameState.PAYLOAD: {FrameType.PAYLOAD},
    FrameState.OUTPUT: {FrameType.OUTPUT},
    FrameState.OUTPUT_FINAL: {FrameType.CONTROL},
    FrameState.RESULT: {FrameType.RESULT},
    FrameState.ERROR: {FrameType.ERROR},
}


@dataclass(frozen=True, slots=True)
class Correlation:
    """Identify one submission attempt within one backend generation.

    Args:
        submission_id: Bounded coordinator-assigned submission identifier.
        attempt: Nonnegative attempt number for the submission.
        generation: Nonnegative backend generation number.
    """

    submission_id: str
    attempt: int
    generation: int

    def __post_init__(self) -> None:
        """Validate the closed correlation identity without external work."""
        _validate_identity(self.submission_id, "submission_id")
        for name, value in (("attempt", self.attempt), ("generation", self.generation)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_PAYLOAD_LENGTH:
                raise FrameError(f"{name} is invalid")


@dataclass(frozen=True, slots=True)
class BootstrapDescriptor:
    """Carry only first-frame correlation and effective transport limits.

    It intentionally excludes workload values, callbacks, environment maps, and
    backend handles.  Construction performs no connection or launcher work.
    """

    correlation: Correlation
    rendezvous_token: str = field(repr=False)
    rendezvous_host: str
    rendezvous_port: int
    control_header_limit_bytes: int
    owner_envelope_limit_bytes: int
    admission_message_limit_bytes: int
    invocation_limit_bytes: int
    result_limit_bytes: int
    output_frame_limit_bytes: int
    output_final_timeout: float = 5.0

    def __post_init__(self) -> None:
        """Validate values that must be known before the first frame is read."""
        if not isinstance(self.correlation, Correlation):
            raise FrameError("bootstrap correlation is invalid")
        _validate_identity(self.rendezvous_token, "rendezvous_token")
        try:
            endpoint = ipaddress.ip_address(self.rendezvous_host)
        except (TypeError, ValueError) as exc:
            raise FrameError("rendezvous host is not a loopback address") from exc
        if not endpoint.is_loopback:
            raise FrameError("rendezvous host is not a loopback address")
        if isinstance(self.rendezvous_port, bool) or not isinstance(self.rendezvous_port, int) or not 0 < self.rendezvous_port <= 65535:
            raise FrameError("rendezvous port is invalid")
        values = (
            ("control_header_limit_bytes", self.control_header_limit_bytes, _MAX_HEADER_LENGTH),
            ("owner_envelope_limit_bytes", self.owner_envelope_limit_bytes, _MAX_PAYLOAD_LENGTH),
            ("admission_message_limit_bytes", self.admission_message_limit_bytes, _MAX_PAYLOAD_LENGTH),
            ("invocation_limit_bytes", self.invocation_limit_bytes, _MAX_PAYLOAD_LENGTH),
            ("result_limit_bytes", self.result_limit_bytes, _MAX_PAYLOAD_LENGTH),
            ("output_frame_limit_bytes", self.output_frame_limit_bytes, _MAX_PAYLOAD_LENGTH),
        )
        for name, value, maximum in values:
            _validate_limit(name, value, maximum)
        if (
            isinstance(self.output_final_timeout, bool)
            or not isinstance(self.output_final_timeout, (int, float))
            or not math.isfinite(self.output_final_timeout)
            or self.output_final_timeout <= 0
        ):
            raise FrameError("output_final_timeout is invalid")
        if self.control_header_limit_bytes > self.admission_message_limit_bytes:
            raise FrameError("control header limit exceeds admission message limit")


@dataclass(frozen=True, slots=True)
class Frame:
    """Hold one validated private frame and its correlation metadata."""

    state: FrameState
    frame_type: FrameType
    correlation: Correlation
    payload: bytes
    owner: OwnerEnvelopeType | None = None
    stream: str | None = None
    sequence: int | None = None


def _validate_identity(value: str, name: str = "identity") -> None:
    """Reject unbounded or structurally unsafe correlation strings."""
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise FrameError(f"{name} is invalid")


def _validate_limit(name: str, value: object, maximum: int) -> None:
    """Validate a non-boolean positive representable protocol limit."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise FrameError(f"{name} is invalid")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes, handling ordinary partial stream reads."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise FrameError("frame ended before its declared length")
        chunks.extend(chunk)
    return bytes(chunks)


def _header_bounds(header_limit: int) -> dict[str, int]:
    """Return fixed grammar bounds without deriving hidden operational caps."""
    _validate_limit("header_limit", header_limit, _MAX_HEADER_LENGTH)
    return {"max_depth": 2, "max_nodes": 20, "max_entries": 11, "max_string": 128, "max_int_bits": 64}


def encode_frame(
    state: FrameState,
    frame_type: FrameType,
    correlation: Correlation,
    payload: bytes,
    *,
    header_limit: int,
    owner: OwnerEnvelopeType | None = None,
    stream: str | None = None,
    sequence: int | None = None,
) -> bytes:
    """Encode one typed frame with canonical metadata and a payload digest.

    Raises ``FrameError`` before writing a nonconforming frame.  The opaque
    payload bound belongs to the caller's typed channel, not this encoder.
    """
    if not isinstance(state, FrameState) or not isinstance(frame_type, FrameType):
        raise FrameError("frame state or type is invalid")
    if not isinstance(correlation, Correlation) or not isinstance(payload, bytes):
        raise FrameError("frame correlation and payload are required")
    if frame_type not in _STATE_TYPES[state]:
        raise FrameError("frame type is not permitted in this state")
    if frame_type is FrameType.OWNER:
        if not isinstance(owner, OwnerEnvelopeType):
            raise FrameError("owner frames require an owner type")
    elif owner is not None:
        raise FrameError("only owner frames may identify an owner type")
    if frame_type is FrameType.OUTPUT:
        if state is not FrameState.OUTPUT or stream not in {"stdout", "stderr"}:
            raise FrameError("output frames require a stdout or stderr stream")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise FrameError("output frame sequence is invalid")
    elif stream is not None or sequence is not None:
        raise FrameError("only output frames may identify a stream sequence")
    _validate_limit("header_limit", header_limit, _MAX_HEADER_LENGTH)
    if len(payload) > _MAX_PAYLOAD_LENGTH:
        raise FrameError("frame payload exceeds framing range")
    metadata: dict[str, object] = {
        "attempt": correlation.attempt,
        "digest": hashlib.sha256(payload).hexdigest(),
        "generation": correlation.generation,
        "length": len(payload),
        "state": state.value,
        "submission_id": correlation.submission_id,
        "type": frame_type.value,
        "version": PROTOCOL_VERSION,
    }
    if owner is not None:
        metadata["owner"] = owner.value
    if frame_type is FrameType.OUTPUT:
        metadata["stream"] = stream
        metadata["sequence"] = sequence
    try:
        header = canonical_json_bytes(metadata, **_header_bounds(header_limit))
    except CanonicalJSONError as exc:
        raise FrameError("frame header is not canonical") from exc
    if len(header) > header_limit:
        raise FrameError("frame header exceeds configured limit")
    return len(header).to_bytes(_HEADER_LENGTH_BYTES, "big") + header + len(payload).to_bytes(_PAYLOAD_LENGTH_BYTES, "big") + payload


def _payload_limit_for(frame_type: FrameType, payload_limit: int | Mapping[FrameType, int]) -> int:
    """Select and validate a caller-provided generic or per-type byte limit."""
    if isinstance(payload_limit, Mapping):
        try:
            value = payload_limit[frame_type]
        except KeyError as exc:
            raise FrameError("frame type has no configured payload limit") from exc
    else:
        value = payload_limit
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_PAYLOAD_LENGTH:
        raise FrameError("payload limit is invalid")
    return value


def decode_frame(stream: BinaryIO, *, header_limit: int, payload_limit: int | Mapping[FrameType, int]) -> Frame:
    """Decode one complete frame without parsing its opaque payload."""
    _validate_limit("header_limit", header_limit, _MAX_HEADER_LENGTH)
    header_size = int.from_bytes(_read_exact(stream, _HEADER_LENGTH_BYTES), "big")
    if header_size > header_limit:
        raise FrameError("frame header exceeds configured limit")
    try:
        header = canonical_json_load_bytes(_read_exact(stream, header_size), **_header_bounds(header_limit))
    except CanonicalJSONError as exc:
        raise FrameError("frame header is not canonical") from exc
    if not isinstance(header, Mapping):
        raise FrameError("frame header has an invalid shape")
    is_owner = header.get("type") == FrameType.OWNER.value
    is_output = header.get("type") == FrameType.OUTPUT.value
    expected = {"attempt", "digest", "generation", "length", "state", "submission_id", "type", "version"}
    if is_owner:
        expected.add("owner")
    if is_output:
        expected.update({"stream", "sequence"})
    version = header.get("version")
    if set(header) != expected or isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise FrameError("frame header has an invalid shape or version")
    try:
        state = FrameState(header["state"])
        frame_type = FrameType(header["type"])
        correlation = Correlation(header["submission_id"], header["attempt"], header["generation"])
        owner = OwnerEnvelopeType(header["owner"]) if is_owner else None
    except (TypeError, ValueError, FrameError) as exc:
        raise FrameError("frame header identity is invalid") from exc
    if frame_type not in _STATE_TYPES[state]:
        raise FrameError("frame type is not permitted in this state")
    stream_name = header.get("stream") if is_output else None
    sequence = header.get("sequence") if is_output else None
    if is_output and (state is not FrameState.OUTPUT or stream_name not in {"stdout", "stderr"} or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0):
        raise FrameError("output frame metadata is invalid")
    declared_length = header["length"]
    if isinstance(declared_length, bool) or not isinstance(declared_length, int) or not 0 <= declared_length <= _MAX_PAYLOAD_LENGTH:
        raise FrameError("frame payload length is invalid")
    typed_limit = _payload_limit_for(frame_type, payload_limit)
    if declared_length > typed_limit:
        raise FrameError("frame payload exceeds configured limit")
    wire_length = int.from_bytes(_read_exact(stream, _PAYLOAD_LENGTH_BYTES), "big")
    if wire_length != declared_length:
        raise FrameError("frame payload length does not match header")
    payload = _read_exact(stream, wire_length)
    digest = header["digest"]
    if not isinstance(digest, str) or not digest.isascii() or not hmac.compare_digest(digest, hashlib.sha256(payload).hexdigest()):
        raise FrameError("frame payload digest does not match header")
    return Frame(state, frame_type, correlation, payload, owner, stream_name, sequence)


class SocketFrameReader:
    """Read typed bounded frames directly from one connected socket.

    Args:
        connection: A connected socket-like object exposing ``recv``.
        header_limit: Effective control-header bound for this conversation.
        payload_limit: Effective per-frame-type payload bounds.

    Returns:
        :meth:`read` returns one parsed frame after applying its typed payload
        limit before reading the payload body.

    Failure behavior:
        Closed or malformed transports raise :class:`FrameError` or ``EOFError``;
        no caller receives an oversized opaque payload buffer.
    """

    def __init__(self, connection: object, *, header_limit: int, payload_limit: int | Mapping[FrameType, int]) -> None:
        """Retain one narrow socket adapter and its already-effective bounds."""
        self._connection = connection
        self._header_limit = header_limit
        self._payload_limit = payload_limit

    def read(self) -> Frame:
        """Decode one frame using the common framing parser and typed limits."""
        return decode_frame(_SocketStream(self._connection), header_limit=self._header_limit, payload_limit=self._payload_limit)


class _SocketStream:
    """Adapt ``recv`` to the minimal bounded ``read`` protocol used by framing."""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    def read(self, size: int = -1) -> bytes:
        """Read at most one requested socket segment without an unbounded default."""
        if size < 0:
            raise FrameError("socket frame reads must be bounded")
        try:
            data = self._connection.recv(size)  # type: ignore[attr-defined]
        except OSError:
            raise
        if not data:
            raise EOFError("protocol peer closed the transport")
        return data


def decode_exact_frame(encoded: bytes, *, header_limit: int, payload_limit: int | Mapping[FrameType, int]) -> Frame:
    """Decode one byte string and reject a trailing frame or trailing garbage."""
    stream = io.BytesIO(encoded)
    frame = decode_frame(stream, header_limit=header_limit, payload_limit=payload_limit)
    if stream.read(1):
        raise FrameError("frame has trailing bytes")
    return frame


def encode_control(state: FrameState, correlation: Correlation, control: Mapping[str, object], *, header_limit: int) -> bytes:
    """Encode a bounded canonical control whose payload uses the configured cap."""
    try:
        payload = canonical_json_bytes(control, max_depth=8, max_nodes=1024, max_entries=64, max_string=header_limit, max_int_bits=64)
    except CanonicalJSONError as exc:
        raise FrameError("control is not canonical") from exc
    if len(payload) > header_limit:
        raise FrameError("control exceeds configured limit")
    return encode_frame(state, FrameType.CONTROL, correlation, payload, header_limit=header_limit)


def decode_control(frame: Frame, *, limit_bytes: int, required_keys: set[str]) -> dict[str, object]:
    """Decode a closed-shape control using its caller-supplied configured bound."""
    if frame.frame_type is not FrameType.CONTROL:
        raise FrameError("frame is not a control")
    _validate_limit("control limit", limit_bytes, _MAX_PAYLOAD_LENGTH)
    if len(frame.payload) > limit_bytes:
        raise FrameError("control exceeds configured limit")
    try:
        control = canonical_json_load_bytes(frame.payload, max_depth=8, max_nodes=1024, max_entries=64, max_string=limit_bytes, max_int_bits=64)
    except CanonicalJSONError as exc:
        raise FrameError("control is not canonical") from exc
    if not isinstance(control, Mapping) or set(control) != required_keys:
        raise FrameError("control has an invalid shape")
    return json_ready(control, max_depth=8, max_nodes=1024, max_entries=64, max_string=limit_bytes, max_int_bits=64)


def encode_owner_envelope(state: FrameState, correlation: Correlation, owner: OwnerEnvelopeType, envelope: bytes, *, header_limit: int, owner_limit: int) -> bytes:
    """Encode one opaque owner envelope in its own typed frame and own limit."""
    _validate_limit("owner envelope limit", owner_limit, _MAX_PAYLOAD_LENGTH)
    if not isinstance(envelope, bytes) or len(envelope) > owner_limit:
        raise FrameError("owner envelope exceeds configured limit")
    return encode_frame(state, FrameType.OWNER, correlation, envelope, header_limit=header_limit, owner=owner)


def decode_owner_envelopes(frames: Iterable[Frame], *, state: FrameState, correlation: Correlation, owner_limit: int, aggregate_limit: int) -> dict[OwnerEnvelopeType, bytes]:
    """Validate a phase's owner frames and aggregate bytes before owner parsing."""
    _validate_limit("owner envelope limit", owner_limit, _MAX_PAYLOAD_LENGTH)
    _validate_limit("admission message limit", aggregate_limit, _MAX_PAYLOAD_LENGTH)
    result: dict[OwnerEnvelopeType, bytes] = {}
    total = 0
    for frame in frames:
        if frame.state is not state or frame.frame_type is not FrameType.OWNER or frame.correlation != correlation or frame.owner is None:
            raise FrameError("owner frame does not match its phase or correlation")
        if len(frame.payload) > owner_limit:
            raise FrameError("owner envelope exceeds configured limit")
        total += len(frame.payload)
        if total > aggregate_limit:
            raise FrameError("owner envelopes exceed aggregate admission limit")
        if frame.owner in result:
            raise FrameError("duplicate owner envelope type")
        result[frame.owner] = frame.payload
    return result


def encode_bootstrap_descriptor(descriptor: BootstrapDescriptor) -> bytes:
    """Encode the fixed bootstrap descriptor under its first-frame header cap."""
    if not isinstance(descriptor, BootstrapDescriptor):
        raise FrameError("bootstrap descriptor is invalid")
    data = {
        "attempt": descriptor.correlation.attempt,
        "generation": descriptor.correlation.generation,
        "rendezvous_token": descriptor.rendezvous_token,
        "rendezvous_host": descriptor.rendezvous_host,
        "rendezvous_port": descriptor.rendezvous_port,
        "submission_id": descriptor.correlation.submission_id,
        "control_header_limit_bytes": descriptor.control_header_limit_bytes,
        "owner_envelope_limit_bytes": descriptor.owner_envelope_limit_bytes,
        "admission_message_limit_bytes": descriptor.admission_message_limit_bytes,
        "invocation_limit_bytes": descriptor.invocation_limit_bytes,
        "result_limit_bytes": descriptor.result_limit_bytes,
        "output_frame_limit_bytes": descriptor.output_frame_limit_bytes,
        "output_final_timeout": descriptor.output_final_timeout,
    }
    try:
        encoded = canonical_json_bytes(data, max_depth=1, max_nodes=15, max_entries=13, max_string=128, max_int_bits=64)
    except CanonicalJSONError as exc:
        raise FrameError("bootstrap descriptor is not canonical") from exc
    if len(encoded) > descriptor.control_header_limit_bytes:
        raise FrameError("bootstrap descriptor exceeds control header limit")
    return encoded


def decode_bootstrap_descriptor(data: bytes, *, header_limit: int) -> BootstrapDescriptor:
    """Decode a fixed descriptor before opening any transport endpoint."""
    _validate_limit("header_limit", header_limit, _MAX_HEADER_LENGTH)
    if not isinstance(data, bytes) or len(data) > header_limit:
        raise FrameError("bootstrap descriptor exceeds control header limit")
    try:
        value = canonical_json_load_bytes(data, max_depth=1, max_nodes=15, max_entries=13, max_string=128, max_int_bits=64)
    except CanonicalJSONError as exc:
        raise FrameError("bootstrap descriptor is not canonical") from exc
    fields = {"attempt", "generation", "rendezvous_token", "rendezvous_host", "rendezvous_port", "submission_id", "control_header_limit_bytes", "owner_envelope_limit_bytes", "admission_message_limit_bytes", "invocation_limit_bytes", "result_limit_bytes", "output_frame_limit_bytes", "output_final_timeout"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FrameError("bootstrap descriptor has an invalid shape")
    try:
        return BootstrapDescriptor(Correlation(value["submission_id"], value["attempt"], value["generation"]), value["rendezvous_token"], value["rendezvous_host"], value["rendezvous_port"], value["control_header_limit_bytes"], value["owner_envelope_limit_bytes"], value["admission_message_limit_bytes"], value["invocation_limit_bytes"], value["result_limit_bytes"], value["output_frame_limit_bytes"], value["output_final_timeout"])
    except (FrameError, TypeError) as exc:
        raise FrameError("bootstrap descriptor is invalid") from exc


class ProtocolConversation:
    """Validate the closed, terminal-aware order for one correlation identity."""

    def __init__(self, correlation: Correlation, *, header_limit: int, invocation_limit: int, result_limit: int, output_limit: int, owner_limit: int, admission_limit: int) -> None:
        """Create an unstarted conversation with explicit effective wire limits."""
        self._correlation = correlation
        self._header_limit = header_limit
        self._limits = {FrameType.CONTROL: header_limit, FrameType.OWNER: owner_limit, FrameType.PAYLOAD: invocation_limit, FrameType.OUTPUT: output_limit, FrameType.RESULT: result_limit, FrameType.ERROR: result_limit}
        self._admission_limit = admission_limit
        self._phase = "hello"
        self._terminal = False
        self._phase_total = 0
        self._phase_owners: set[OwnerEnvelopeType] = set()
        self._outcome: FrameType | None = None
        self._stream_sequences = {"stdout": 0, "stderr": 0}
        self._stream_finals: set[str] = set()

    def accept(self, encoded: bytes) -> Frame:
        """Validate one frame without mutating state on an invalid transition.

        PREPARE and READY each admit one control followed by distinct owner
        envelopes.  GO authorizes exactly one payload.  After that payload, the
        result/error and the two output stream fences may arrive independently.
        """
        if self._terminal:
            raise FrameError("frame follows a terminal protocol state")
        frame = decode_exact_frame(encoded, header_limit=self._header_limit, payload_limit=self._limits)
        return self.accept_frame(frame)

    def accept_frame(self, frame: Frame) -> Frame:
        """Validate an already bounded frame without re-reading or re-encoding it."""
        if not isinstance(frame, Frame):
            raise FrameError("protocol frame is invalid")
        if self._terminal:
            raise FrameError("frame follows a terminal protocol state")
        if frame.correlation != self._correlation:
            raise FrameError("frame correlation does not match conversation")
        if self._phase == "hello":
            self._require(frame, FrameState.HELLO, FrameType.CONTROL)
            self._phase = "prepare-control"
            return frame
        if self._phase == "prepare-control":
            if self._is_admission_error(frame):
                self._terminal = True
                return frame
            self._require(frame, FrameState.PREPARE, FrameType.CONTROL)
            self._begin_admission_phase(frame)
            self._phase = "prepare-owners"
            return frame
        if self._phase == "prepare-owners":
            if self._is_admission_error(frame):
                self._terminal = True
                return frame
            if frame.state is FrameState.READY and frame.frame_type is FrameType.CONTROL:
                self._begin_admission_phase(frame)
                self._phase = "ready-owners"
                return frame
            self._accept_owner(frame, FrameState.PREPARE)
            return frame
        if self._phase == "ready-owners":
            if self._is_admission_error(frame):
                self._terminal = True
                return frame
            if frame.state is FrameState.GO and frame.frame_type is FrameType.CONTROL:
                self._phase = "payload"
                return frame
            if frame.state is FrameState.STOP and frame.frame_type is FrameType.CONTROL:
                self._terminal = True
                return frame
            self._accept_owner(frame, FrameState.READY)
            return frame
        if self._phase == "payload":
            if self._is_admission_error(frame):
                self._terminal = True
                return frame
            if frame.state is FrameState.STOP and frame.frame_type is FrameType.CONTROL:
                self._terminal = True
                return frame
            self._require(frame, FrameState.PAYLOAD, FrameType.PAYLOAD)
            self._phase = "active"
            return frame
        self._accept_active(frame)
        return frame

    @staticmethod
    def _require(frame: Frame, state: FrameState, frame_type: FrameType) -> None:
        """Reject a state/type mismatch before a caller-visible transition."""
        if frame.state is not state or frame.frame_type is not frame_type:
            raise FrameError("frame is out of order")

    @staticmethod
    def _is_admission_error(frame: Frame) -> bool:
        """Identify the error frame that ends an unpermitted admission exchange."""
        return frame.state is FrameState.ERROR and frame.frame_type is FrameType.ERROR

    def _begin_admission_phase(self, control: Frame) -> None:
        """Start a bounded PREPARE or READY aggregate with its control bytes."""
        if len(control.payload) > self._admission_limit:
            raise FrameError("admission controls exceed aggregate admission limit")
        self._phase_total = len(control.payload)
        self._phase_owners = set()

    def _accept_owner(self, frame: Frame, state: FrameState) -> None:
        """Validate a unique bounded owner envelope before recording its phase sum."""
        self._require(frame, state, FrameType.OWNER)
        assert frame.owner is not None
        if frame.owner in self._phase_owners:
            raise FrameError("duplicate owner envelope type")
        if self._phase_total + len(frame.payload) > self._admission_limit:
            raise FrameError("owner envelopes exceed aggregate admission limit")
        self._phase_owners.add(frame.owner)
        self._phase_total += len(frame.payload)

    def _accept_active(self, frame: Frame) -> None:
        """Accept independent output fences and exactly one validated outcome."""
        if frame.state is FrameState.OUTPUT and frame.frame_type is FrameType.OUTPUT:
            assert frame.stream is not None and frame.sequence is not None
            if frame.stream in self._stream_finals or frame.sequence != self._stream_sequences[frame.stream]:
                raise FrameError("output frame is outside its stream sequence")
            self._stream_sequences[frame.stream] += 1
            return
        if frame.state is FrameState.OUTPUT_FINAL and frame.frame_type is FrameType.CONTROL:
            stream, sequence = self._output_final_fence(frame)
            if stream in self._stream_finals or sequence != self._stream_sequences[stream]:
                raise FrameError("output final is outside its stream sequence")
            self._stream_finals.add(stream)
            self._finish_if_complete()
            return
        if frame.state in {FrameState.RESULT, FrameState.ERROR} and frame.frame_type in {FrameType.RESULT, FrameType.ERROR}:
            if self._outcome is not None:
                raise FrameError("frame is out of order")
            self._outcome = frame.frame_type
            self._finish_if_complete()
            return
        raise FrameError("frame is out of order")

    @staticmethod
    def _output_final_fence(frame: Frame) -> tuple[str, int]:
        """Decode the closed control fence without treating outcome as a fence."""
        try:
            control = canonical_json_load_bytes(frame.payload, max_depth=1, max_nodes=3, max_entries=2, max_string=16, max_int_bits=64)
        except CanonicalJSONError as exc:
            raise FrameError("output final is not canonical") from exc
        if not isinstance(control, Mapping) or set(control) != {"stream", "next_sequence"}:
            raise FrameError("output final has an invalid shape")
        stream = control["stream"]
        sequence = control["next_sequence"]
        if stream not in {"stdout", "stderr"} or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise FrameError("output final has an invalid stream sequence")
        return stream, sequence

    def _finish_if_complete(self) -> None:
        """Mark the conversation terminal only after outcome and both stream fences."""
        if self._outcome is not None and self._stream_finals == {"stdout", "stderr"}:
            self._terminal = True


__all__ = ["BootstrapDescriptor", "Correlation", "Frame", "FrameError", "FrameState", "FrameType", "OwnerEnvelopeType", "PROTOCOL_VERSION", "ProtocolConversation", "decode_bootstrap_descriptor", "decode_control", "decode_exact_frame", "decode_frame", "decode_owner_envelopes", "encode_bootstrap_descriptor", "encode_control", "encode_frame", "encode_owner_envelope"]
