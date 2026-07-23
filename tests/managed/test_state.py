from __future__ import annotations

import pytest

from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedMethodDeclaration,
    ManagedOutput,
    ManagedOutputs,
    ManagedStateError,
    OperationKey,
    RealizationState,
    declaration_fingerprint,
)


def _declaration(*, resumable=False, representation="numpy"):
    return ManagedMethodDeclaration(
        outputs=ManagedOutputs(
            ManagedOutput(
                "result",
                primary=True,
                kind="data",
                representations=(representation,),
            )
        ),
        resumable=resumable,
        checkpoint_schema="cursor-v1" if resumable else None,
    )


def test_operation_key_and_declaration_fingerprint_are_stable_and_host_free():
    key = OperationKey(format_cdef_id("a" * 64), "compute")
    first = declaration_fingerprint("compute", _declaration(resumable=True))
    second = declaration_fingerprint("compute", _declaration(resumable=True))

    assert first == second
    assert first.startswith("managed-declaration-v1-")
    assert declaration_fingerprint("compute", _declaration(representation="parquet")) != first
    assert key.to_json() == {
        "method": "compute",
        "producer_cdef_id": format_cdef_id("a" * 64),
    }
    assert "/" not in first and "\\" not in first


def test_realization_state_round_trip_is_versioned_and_strict():
    state = RealizationState(
        realization_id="realization-v1-" + "b" * 32,
        declaration_fingerprint="managed-declaration-v1-" + "c" * 64,
        status="interrupted",
        resumable=True,
        attempt_ids=("attempt-v1-" + "d" * 32,),
        current_attempt_id=None,
        checkpoint_head="checkpoint-v1-test",
        diagnostics=("checkpoint committed",),
    )

    assert RealizationState.from_json(state.to_json()) == state

    malformed = state.to_json()
    malformed["schema_version"] = 2
    with pytest.raises(ManagedStateError, match="schema_version"):
        RealizationState.from_json(malformed)

    malformed = state.to_json()
    malformed["host_path"] = "/tmp/private"
    with pytest.raises(ManagedStateError, match="unknown fields"):
        RealizationState.from_json(malformed)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unknown"},
        {"attempt_ids": []},
        {"diagnostics": ["x"] * 33},
        {"realization_id": "../escape"},
    ],
)
def test_realization_state_rejects_malformed_control_values(changes):
    data = RealizationState(
        realization_id="realization-v1-" + "e" * 32,
        declaration_fingerprint="managed-declaration-v1-" + "f" * 64,
        status="running",
        resumable=False,
        attempt_ids=("attempt-v1-" + "1" * 32,),
        current_attempt_id="attempt-v1-" + "1" * 32,
    ).to_json()
    data.update(changes)

    with pytest.raises(ManagedStateError):
        RealizationState.from_json(data)
