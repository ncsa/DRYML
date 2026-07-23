from __future__ import annotations

import pytest

from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    DataRecord,
    ExecutionRecord,
    ExecutionRecordLink,
    RealizationOutput,
    RealizationRecord,
    RecordValidationError,
    ResolvedRecord,
    StorageRef,
    StoredStateRecord,
    typed_record_from_envelope,
)


def _id(prefix, value):
    return content_id(prefix, 1, {prefix: value})


def _cdef(char="a"):
    return format_cdef_id(char * 64)


REALIZATION_ID = "realization-v1-" + "1" * 32
ATTEMPT_ID = "attempt-v1-" + "2" * 32
DECLARATION = "managed-declaration-v1-" + "3" * 64


def _resolved(record_id=None):
    return ResolvedRecord(
        producer_cdef_id=_cdef("b"),
        method="compute",
        declaration_fingerprint="managed-declaration-v1-" + "4" * 64,
        activation_generation=7,
        realization_id="realization-v1-" + "5" * 32,
        output_slot="result",
        record_id=record_id or _id("record", "input"),
    )


def test_realization_and_managed_ownership_round_trip():
    output_id = _id("record", "output")
    execution_id = _id("record", "execution")
    representation_id = _id("repr", "numpy")
    resolved = _resolved()
    realization = RealizationRecord(
        realization_id=REALIZATION_ID,
        producer_cdef_id=_cdef(),
        method="compute",
        declaration_fingerprint=DECLARATION,
        attempt_ids=(ATTEMPT_ID,),
        outputs=(
            RealizationOutput(
                slot="result",
                record_id=output_id,
                record_kind="data",
                representation_id=representation_id,
                required=True,
            ),
        ),
        primary_output_slot="result",
        primary_representation_id=representation_id,
        execution_record_id=execution_id,
        consumed_records=(resolved,),
        completed_attempt_id=ATTEMPT_ID,
        completion_fence=9,
    )

    envelope = realization.to_envelope()
    assert RealizationRecord.from_envelope(envelope) == realization
    assert isinstance(typed_record_from_envelope(envelope), RealizationRecord)

    data = DataRecord(
        representation_id=representation_id,
        storage=(StorageRef.self_product(role="result"),),
        realization_id=REALIZATION_ID,
        output_slot="result",
    )
    state = StoredStateRecord(
        subject_cdef_id=_cdef("c"),
        representation_id=representation_id,
        storage=(StorageRef.self_product(role="state"),),
        realization_id=REALIZATION_ID,
        output_slot="model",
    )
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=_id("op", "compute"),
        backend={"name": "dryml.fake"},
        status="ok",
        realization_id=REALIZATION_ID,
        consumed_records=(ExecutionRecordLink.from_resolved(resolved),),
        produced_records=(
            ExecutionRecordLink(
                output_id,
                role="result",
                realization_id=REALIZATION_ID,
                output_slot="result",
                required=True,
            ),
        ),
    )

    assert DataRecord.from_envelope(data.to_envelope()).realization_id == REALIZATION_ID
    assert StoredStateRecord.from_envelope(state.to_envelope()).output_slot == "model"
    assert ExecutionRecord.from_envelope(execution.to_envelope()).consumed_records[0].to_resolved() == resolved


@pytest.mark.parametrize(
    "change",
    [
        {"realization_id": "record-v1-" + "0" * 64},
        {"method": "bad/method"},
        {"attempt_ids": ("attempt-v2-" + "0" * 32,)},
        {"completion_fence": 0},
        {"primary_output_slot": "missing"},
    ],
)
def test_realization_record_rejects_malformed_identity_and_lineage(change):
    values = dict(
        realization_id=REALIZATION_ID,
        producer_cdef_id=_cdef(),
        method="compute",
        declaration_fingerprint=DECLARATION,
        attempt_ids=(ATTEMPT_ID,),
        outputs=(
            RealizationOutput(
                "result",
                _id("record", "output"),
                "data",
                _id("repr", "numpy"),
            ),
        ),
        primary_output_slot="result",
        primary_representation_id=_id("repr", "numpy"),
        execution_record_id=_id("record", "execution"),
        completed_attempt_id=ATTEMPT_ID,
        completion_fence=1,
    )
    values.update(change)
    with pytest.raises(RecordValidationError):
        RealizationRecord(**values)


def test_exact_consumed_vector_is_all_or_none_and_output_ownership_is_paired():
    with pytest.raises(RecordValidationError, match="exact consumed vector"):
        ExecutionRecordLink(
            _id("record", "input"),
            activation_generation=1,
            realization_id=REALIZATION_ID,
        )
    with pytest.raises(RecordValidationError, match="ownership"):
        DataRecord(
            representation_id=_id("repr", "numpy"),
            storage=(StorageRef.self_product(),),
            realization_id=REALIZATION_ID,
        )


def test_independent_realizations_and_adapted_representations_remain_distinct():
    source = DataRecord(
        representation_id=_id("repr", "numpy"),
        storage=(StorageRef.self_product(),),
        realization_id=REALIZATION_ID,
        output_slot="result",
    )
    adapted = DataRecord(
        representation_id=_id("repr", "parquet"),
        storage=(StorageRef.self_product(),),
        realization_id=REALIZATION_ID,
        output_slot="result",
        derived_from=(_id("record", "source"),),
    )
    recomputed = DataRecord(
        representation_id=source.representation_id,
        storage=(StorageRef.self_product(),),
        realization_id="realization-v1-" + "6" * 32,
        output_slot="result",
    )

    assert adapted.realization_id == source.realization_id
    assert recomputed.realization_id != source.realization_id
