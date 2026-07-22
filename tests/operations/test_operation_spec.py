import pytest

from dryml.core import Object
from dryml.core.store.dir import DirStore
from dryml.formats.refs import format_cdef_id, format_ref_cdef
from dryml.operations import (
    OperationSpecError,
    attach_operation_id,
    compute_operation_id,
    make_function_call_spec,
    make_method_call_spec,
    operation_payload_for_id,
    validate_operation_spec,
)
from dryml.records import RecordStoreIO


def cdef(char="a"):
    return format_cdef_id(char * 64)


def test_function_call_shape_and_stable_op_id():
    left = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[cdef()], kwargs={"b": 2, "a": 1}, metadata={"x": 1}))
    right = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[cdef()], kwargs={"a": 1, "b": 2}, metadata={"x": 2}))

    assert left["schema"] == "dryml.operation.v1"
    assert left["schema_version"] == 1
    assert left["kind"] == "function_call"
    assert left["payload"] == {"function": "pkg.mod:run", "args": [cdef()], "kwargs": {"a": 1, "b": 2}}
    assert left["id"].startswith("op-v1-")
    assert left["id"] == right["id"]
    assert "metadata" not in operation_payload_for_id(left)


def test_function_call_id_changes_with_semantic_payload():
    base = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[1], kwargs={"x": 1}))
    changed_function = attach_operation_id(make_function_call_spec("pkg.mod:other", args=[1], kwargs={"x": 1}))
    changed_args = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[2], kwargs={"x": 1}))
    changed_kwargs = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[1], kwargs={"x": 2}))

    assert len({base["id"], changed_function["id"], changed_args["id"], changed_kwargs["id"]}) == 4


def test_function_call_defaults_and_validation_errors():
    spec = make_function_call_spec("pkg.mod:run")
    assert spec["payload"]["args"] == []
    assert spec["payload"]["kwargs"] == {}

    with pytest.raises(OperationSpecError):
        make_function_call_spec("not-an-import-path")
    with pytest.raises(OperationSpecError):
        make_function_call_spec("pkg.mod:run", kwargs=[])
    with pytest.raises(OperationSpecError):
        make_function_call_spec("pkg.mod:run", kwargs={1: "x"})
    with pytest.raises(OperationSpecError):
        make_function_call_spec("pkg.mod:run", args=["cdef-v4-nothex"])
    with pytest.raises(OperationSpecError):
        make_function_call_spec("pkg.mod:run", kwargs={"x": "op-v1-short"})


def test_method_call_shape_stable_id_and_validation_errors():
    left = attach_operation_id(make_method_call_spec(cdef(), "train.step", kwargs={"epochs": 1}))
    right = attach_operation_id(make_method_call_spec(cdef(), "train.step", kwargs={"epochs": 1}))
    changed_subject = attach_operation_id(make_method_call_spec(cdef("b"), "train.step", kwargs={"epochs": 1}))
    changed_method = attach_operation_id(make_method_call_spec(cdef(), "train.other", kwargs={"epochs": 1}))

    assert left["kind"] == "method_call"
    assert left["payload"]["subject"] == cdef()
    assert left["payload"]["method"] == "train.step"
    assert left["id"].startswith("op-v1-")
    assert left["id"] == right["id"]
    assert left["id"] != changed_subject["id"]
    assert left["id"] != changed_method["id"]

    with pytest.raises(OperationSpecError):
        make_method_call_spec(format_ref_cdef(cdef()), "train")
    with pytest.raises(OperationSpecError):
        make_method_call_spec("cdef-v4-bad", "train")
    with pytest.raises(OperationSpecError):
        make_method_call_spec(cdef(), "train()")
    with pytest.raises(OperationSpecError):
        make_method_call_spec(cdef(), "bad name")


def test_existing_wrong_ids_are_rejected():
    spec = make_function_call_spec("pkg.mod:run", args=[1])
    wrong_op = "op-v1-" + "a" * 64
    wrong_prefix = "repr-v1-" + "a" * 64

    with pytest.raises(OperationSpecError):
        validate_operation_spec(dict(spec, id=wrong_op))
    with pytest.raises(OperationSpecError):
        attach_operation_id(dict(spec, id=wrong_op))
    with pytest.raises(OperationSpecError):
        validate_operation_spec(dict(spec, id=wrong_prefix))


def test_operation_spec_writes_under_operation_family_and_is_not_object(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    spec = make_function_call_spec("pkg.mod:run", args=[cdef()])
    located = io.write_spec(spec, family="operation")

    assert located.spec_id == compute_operation_id(spec)
    assert located.kind == "operation"
    assert (io.spec_family_dir("operation") / f"{located.spec_id}.json").exists()
    assert isinstance(spec, dict)
    assert not isinstance(spec, Object)
