import pytest

from dryml.formats.refs import format_cdef_id, format_ref_cdef
from dryml.operations import CDefRefArg, MaterializeCDefArg, OperationResolutionError, make_function_call_spec, make_method_call_spec, resolve_call_arguments


def cdef(char="a"):
    return format_cdef_id(char * 64)


def test_function_call_resolution_defaults_and_literals():
    spec = make_function_call_spec(
        "pkg.mod:run",
        args=[cdef(), format_ref_cdef(cdef("b")), {"$literal": cdef("c")}, [cdef("d")], {"nested": format_ref_cdef(cdef("e"))}],
        kwargs={"model": cdef("f"), "record": "record-v1-" + "a" * 64},
    )

    call = resolve_call_arguments(spec)

    assert call.function == "pkg.mod:run"
    assert call.args[0] == MaterializeCDefArg(cdef())
    assert call.args[1] == CDefRefArg(cdef("b"))
    assert call.args[2] == cdef("c")
    assert call.args[3] == [MaterializeCDefArg(cdef("d"))]
    assert call.args[4] == {"nested": CDefRefArg(cdef("e"))}
    assert call.kwargs["model"] == MaterializeCDefArg(cdef("f"))
    assert call.kwargs["record"] == "record-v1-" + "a" * 64


def test_method_subject_and_callbacks_are_resolved():
    spec = make_method_call_spec(cdef(), "train", args=[format_ref_cdef(cdef("b"))])
    call = resolve_call_arguments(spec, materialize_cdef=lambda value: ("mat", value), make_cdef_ref=lambda value: ("ref", value))

    assert call.method == "train"
    assert call.subject == ("mat", cdef())
    assert call.args == (("ref", cdef("b")),)


def test_method_call_identity_has_no_launch_context():
    plain = make_method_call_spec(cdef(), "compute", args=[1], kwargs={"value": "x"})
    launched = dict(plain)
    launch = {
        "store": "/tmp/store",
        "realization_id": "realization-v1-" + "a" * 32,
        "fence_epoch": 1,
    }

    assert resolve_call_arguments(plain) == resolve_call_arguments(launched)
    assert "launch" not in launched
    assert launch["fence_epoch"] == 1


def test_resolution_rejects_malformed_escapes_and_refs():
    with pytest.raises(OperationResolutionError):
        resolve_call_arguments({"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {"function": "pkg.mod:run", "args": [{"$literal": "x", "extra": True}], "kwargs": {}}})
    with pytest.raises(OperationResolutionError):
        resolve_call_arguments({"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {"function": "pkg.mod:run", "args": ["ref(cdef-v4-bad)"], "kwargs": {}}})
