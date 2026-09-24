from __future__ import annotations

import io

import pytest

from dryml.formats import canonical_json_bytes, canonical_json_load_bytes

from dryml.execute._protocol import (
    BootstrapDescriptor,
    Correlation,
    FrameError,
    FrameState,
    FrameType,
    OwnerEnvelopeType,
    PROTOCOL_VERSION,
    ProtocolConversation,
    decode_bootstrap_descriptor,
    decode_control,
    decode_worker_error,
    decode_exact_frame,
    decode_frame,
    decode_owner_envelopes,
    encode_bootstrap_descriptor,
    encode_control,
    encode_frame,
    encode_frame_parts,
    encode_owner_envelope,
)


def _correlation() -> Correlation:
    return Correlation("submission-1", 2, 3)


def _conversation() -> ProtocolConversation:
    return ProtocolConversation(_correlation(), header_limit=512, invocation_limit=128, result_limit=128, output_limit=64, owner_limit=32, admission_limit=48)


def test_frames_round_trip_closed_control_and_identity_triple():
    """Private frames bind exact type/state and full attempt-generation identity."""
    encoded = encode_control(FrameState.HELLO, _correlation(), {"ready": True}, header_limit=512)
    frame = decode_exact_frame(encoded, header_limit=512, payload_limit=512)
    assert frame.frame_type is FrameType.CONTROL
    assert frame.correlation == _correlation()
    assert decode_control(frame, limit_bytes=512, required_keys={"ready"}) == {"ready": True}

    bad = bytearray(encoded)
    bad[-1] ^= 1
    with pytest.raises(FrameError, match="digest"):
        decode_exact_frame(bytes(bad), header_limit=512, payload_limit=512)
    with pytest.raises(FrameError, match="permitted"):
        encode_frame(FrameState.HELLO, FrameType.RESULT, _correlation(), b"result", header_limit=512)
    with pytest.raises(FrameError, match="trailing"):
        decode_exact_frame(encoded + b"x", header_limit=512, payload_limit=512)


@pytest.mark.parametrize("setup", (False, True))
def test_worker_deadline_error_is_closed_and_distinct_from_user_timeout(setup):
    """Only the private marker, not a remote type name, denotes worker expiry."""
    cleanup = b'"cleanup":[],' if setup else b""
    deadline = b"{" + cleanup + b'"deadline":"pre-invocation"}'
    timeout = b"{" + cleanup + b'"type":"TimeoutError"}'

    deadline_frame = decode_exact_frame(
        encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), deadline, header_limit=512),
        header_limit=512,
        payload_limit=512,
    )
    timeout_frame = decode_exact_frame(
        encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), timeout, header_limit=512),
        header_limit=512,
        payload_limit=512,
    )

    assert decode_worker_error(deadline_frame, limit_bytes=512, setup=setup) == (None, True, ())
    assert decode_worker_error(timeout_frame, limit_bytes=512, setup=setup) == ("TimeoutError", False, ())


@pytest.mark.parametrize(
    "payload",
    (
        b'{"deadline":"pre-invocation","type":"TimeoutError"}',
        b'{"deadline":true}',
        b'{"deadline":"expired"}',
        b'{"cleanup":[],"deadline":"pre-invocation"}',
    ),
)
def test_worker_deadline_error_rejects_malformed_or_wrong_variant_fields(payload):
    """Deadline evidence has one exact ordinary-terminal shape."""
    frame = decode_exact_frame(
        encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), payload, header_limit=512),
        header_limit=512,
        payload_limit=512,
    )
    with pytest.raises(FrameError):
        decode_worker_error(frame, limit_bytes=512, setup=False)


def test_short_reads_malformed_headers_and_controls_fail_closed_without_hidden_cap():
    """Frame/parser limits are caller supplied and every exact read is required."""
    encoded = encode_control(FrameState.HELLO, _correlation(), {"message": "x" * 300}, header_limit=512)

    class ShortReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(1 if size > 1 else size)

    assert decode_frame(ShortReader(encoded), header_limit=512, payload_limit=512).payload
    with pytest.raises(FrameError, match="frame ended"):
        decode_frame(io.BytesIO(encoded[:-1]), header_limit=512, payload_limit=512)
    duplicate = encode_frame(FrameState.HELLO, FrameType.CONTROL, _correlation(), b'{"x":1,"x":2}', header_limit=512)
    with pytest.raises(FrameError, match="canonical"):
        decode_control(decode_exact_frame(duplicate, header_limit=512, payload_limit=512), limit_bytes=512, required_keys={"x"})
    with pytest.raises(FrameError, match="header"):
        decode_frame(io.BytesIO((513).to_bytes(4, "big") + b"x" * 513), header_limit=512, payload_limit=512)


def test_owner_envelopes_are_separate_typed_frames_and_aggregate_before_parsing():
    """A maximal owner payload fits apart from metadata and phase sums are bounded."""
    first = decode_exact_frame(encode_owner_envelope(FrameState.PREPARE, _correlation(), OwnerEnvelopeType.ENVIRONMENT, b"a" * 32, header_limit=512, owner_limit=32), header_limit=512, payload_limit=32)
    second = decode_exact_frame(encode_owner_envelope(FrameState.PREPARE, _correlation(), OwnerEnvelopeType.WORLD, b"b" * 17, header_limit=512, owner_limit=32), header_limit=512, payload_limit=32)
    with pytest.raises(FrameError, match="aggregate"):
        decode_owner_envelopes((first, second), state=FrameState.PREPARE, correlation=_correlation(), owner_limit=32, aggregate_limit=48)
    parsed = decode_owner_envelopes((first,), state=FrameState.PREPARE, correlation=_correlation(), owner_limit=32, aggregate_limit=48)
    assert parsed[OwnerEnvelopeType.ENVIRONMENT] == b"a" * 32


def test_conversation_models_stop_output_final_result_and_error_paths():
    """Terminal and output paths cannot be reclassified as a successful result."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))
    conversation.accept(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, _correlation(), b"line", header_limit=512, stream="stdout", sequence=0))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stdout", "next_sequence": 1}, header_limit=512))
    conversation.accept(encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), b"failure", header_limit=512))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stderr", "next_sequence": 0}, header_limit=512))
    with pytest.raises(FrameError, match="terminal"):
        conversation.accept(encode_frame(FrameState.RESULT, FrameType.RESULT, _correlation(), b"result", header_limit=512))

    stopped = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.STOP, {"reason": "cancelled"})):
        stopped.accept(encode_control(state, _correlation(), control, header_limit=512))
    with pytest.raises(FrameError, match="terminal"):
        stopped.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))


def test_conversation_rejects_payload_or_repeated_go_before_permission_without_mutation():
    """Payload and repeated GO cannot bypass the one-shot authorization gate."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))

    payload = encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512)
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(payload)
    conversation.accept(encode_control(FrameState.GO, _correlation(), {"permit": True}, header_limit=512))
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(encode_control(FrameState.GO, _correlation(), {"permit": True}, header_limit=512))
    conversation.accept(payload)
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(payload)


def test_conversation_withholds_payload_until_setup_ready_but_allows_setup_output():
    """A post-GO setup exchange gates payload while its captured output drains."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    setup = encode_owner_envelope(FrameState.SETUP, _correlation(), OwnerEnvelopeType.SETUP, b"{}", header_limit=512, owner_limit=32)
    conversation.accept(setup)
    conversation.accept(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, _correlation(), b"setup output", header_limit=512, stream="stdout", sequence=0))
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))
    conversation.accept(encode_control(FrameState.SETUP_READY, _correlation(), {"ready": True}, header_limit=512))
    conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))


def test_setup_rejects_pre_invocation_result_and_cannot_resume_after_error():
    """A failed setup may drain final output but never authorize payload transfer."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    conversation.accept(encode_owner_envelope(FrameState.SETUP, _correlation(), OwnerEnvelopeType.SETUP, b"{}", header_limit=512, owner_limit=32))
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(encode_frame(FrameState.RESULT, FrameType.RESULT, _correlation(), b"result", header_limit=512))

    conversation.accept(encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), b"failure", header_limit=512))
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(encode_control(FrameState.SETUP_READY, _correlation(), {"ready": True}, header_limit=512))
    with pytest.raises(FrameError, match="out of order"):
        conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stdout", "next_sequence": 0}, header_limit=512))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stderr", "next_sequence": 0}, header_limit=512))


def test_conversation_bounds_each_admission_phase_before_owner_parsing():
    """PREPARE and READY sums include their control payloads and reset per phase."""
    conversation = _conversation()
    conversation.accept(encode_control(FrameState.HELLO, _correlation(), {"worker": "w"}, header_limit=512))
    conversation.accept(encode_control(FrameState.PREPARE, _correlation(), {"controls": "x" * 20}, header_limit=512))
    owner = encode_owner_envelope(FrameState.PREPARE, _correlation(), OwnerEnvelopeType.ENVIRONMENT, b"a" * 20, header_limit=512, owner_limit=32)
    with pytest.raises(FrameError, match="aggregate"):
        conversation.accept(owner)

    fresh = _conversation()
    fresh.accept(encode_control(FrameState.HELLO, _correlation(), {"worker": "w"}, header_limit=512))
    fresh.accept(encode_control(FrameState.PREPARE, _correlation(), {"controls": []}, header_limit=512))
    fresh.accept(encode_owner_envelope(FrameState.PREPARE, _correlation(), OwnerEnvelopeType.ENVIRONMENT, b"a" * 20, header_limit=512, owner_limit=32))
    fresh.accept(encode_control(FrameState.READY, _correlation(), {"ready": True}, header_limit=512))
    fresh.accept(encode_owner_envelope(FrameState.READY, _correlation(), OwnerEnvelopeType.ENVIRONMENT, b"b" * 20, header_limit=512, owner_limit=32))


def test_known_outcome_allows_later_other_stream_final_but_not_duplicate_outcome():
    """Outcome and stream fences are independent after payload authorization."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))
    conversation.accept(encode_frame(FrameState.RESULT, FrameType.RESULT, _correlation(), b"result", header_limit=512))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stdout", "next_sequence": 0}, header_limit=512))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stderr", "next_sequence": 0}, header_limit=512))
    with pytest.raises(FrameError, match="terminal"):
        conversation.accept(encode_frame(FrameState.ERROR, FrameType.ERROR, _correlation(), b"duplicate", header_limit=512))


def test_sender_frame_parts_preserve_wire_bytes_without_self_decoding():
    """A near-limit payload can advance a conversation from its validated Frame."""
    payload = b"x" * 128
    frame, prefix, payload_view = encode_frame_parts(
        FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), payload,
        header_limit=512,
    )
    assert prefix + payload_view == encode_frame(
        FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), payload,
        header_limit=512,
    )
    assert payload_view.obj is payload

    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    assert conversation.accept_frame(frame) is frame


def test_output_streams_require_contiguous_sequences_and_closed_fences():
    """Per-stream output frames cannot arrive after a fence or skip a sequence."""
    conversation = _conversation()
    for state, control in ((FrameState.HELLO, {"worker": "w"}), (FrameState.PREPARE, {"controls": []}), (FrameState.READY, {"ready": True}), (FrameState.GO, {"permit": True})):
        conversation.accept(encode_control(state, _correlation(), control, header_limit=512))
    conversation.accept(encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"call", header_limit=512))
    with pytest.raises(FrameError, match="sequence"):
        conversation.accept(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, _correlation(), b"skip", header_limit=512, stream="stdout", sequence=1))
    conversation.accept(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, _correlation(), b"first", header_limit=512, stream="stdout", sequence=0))
    conversation.accept(encode_control(FrameState.OUTPUT_FINAL, _correlation(), {"stream": "stdout", "next_sequence": 1}, header_limit=512))
    with pytest.raises(FrameError, match="sequence"):
        conversation.accept(encode_frame(FrameState.OUTPUT, FrameType.OUTPUT, _correlation(), b"late", header_limit=512, stream="stdout", sequence=1))


def test_decode_rejects_old_or_boolean_version_and_non_ascii_digest_as_frame_errors():
    """Metadata scalar types are strict and diagnostics never leak raw HMAC errors."""
    encoded = encode_control(FrameState.HELLO, _correlation(), {"ready": True}, header_limit=512)
    header_size = int.from_bytes(encoded[:4], "big")
    header = canonical_json_load_bytes(encoded[4:4 + header_size])
    payload = encoded[4 + header_size:]
    for key, value in (("version", True), ("version", PROTOCOL_VERSION - 1), ("digest", "é" * 64)):
        altered = dict(header)
        altered[key] = value
        altered_header = canonical_json_bytes(altered)
        wire = len(altered_header).to_bytes(4, "big") + altered_header + payload
        with pytest.raises(FrameError):
            decode_exact_frame(wire, header_limit=512, payload_limit=512)


def test_typed_frame_limit_rejects_from_metadata_before_reading_payload_bytes():
    """A declared oversized payload is rejected before its wire length/body is read."""
    encoded = encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, _correlation(), b"x" * 5, header_limit=512)
    header_size = int.from_bytes(encoded[:4], "big")
    stream = io.BytesIO(encoded)
    limits = {FrameType.CONTROL: 512, FrameType.OWNER: 32, FrameType.PAYLOAD: 4, FrameType.OUTPUT: 64, FrameType.RESULT: 128, FrameType.ERROR: 128}
    with pytest.raises(FrameError, match="configured limit"):
        decode_frame(stream, header_limit=512, payload_limit=limits)
    assert stream.tell() == 4 + header_size


def test_bootstrap_descriptor_is_closed_and_bounded_before_transport():
    """Bootstrap controls carry only correlation and effective wire-limit scalars."""
    descriptor = BootstrapDescriptor(
        _correlation(), "token-1", "127.0.0.1", 43123, 512, 32, 512, 128, 128, 64,
        output_final_timeout=0.0001,
    )
    encoded = encode_bootstrap_descriptor(descriptor)
    assert decode_bootstrap_descriptor(encoded, header_limit=512) == descriptor
    assert "token-1" not in repr(descriptor)
    with pytest.raises(FrameError, match="limit"):
        decode_bootstrap_descriptor(encoded, header_limit=len(encoded) - 1)
    with pytest.raises(FrameError, match="loopback"):
        BootstrapDescriptor(_correlation(), "token-1", "192.0.2.1", 43123, 512, 32, 512, 128, 128, 64)
