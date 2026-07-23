import importlib
import sys

from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.formats.refs import format_cdef_id
from dryml.operations import attach_operation_id, make_function_call_spec, make_method_call_spec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_method_call_materializes_subject(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(7)
    repo.save(box, store=store, record_policy="descriptive")
    cdef_id = format_cdef_id(box.definition.stable_hash())
    op = attach_operation_id(make_method_call_spec(cdef_id, "plus", args=[5]))

    result = Dispatcher(store=store).run(op, environment=_env(target_module))

    assert result.status == "ok"
    assert result.result_canonical == 12


def test_function_call_cdef_materialize_ref_and_literal(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(stores=[store])
    box = mod.Box(9)
    repo.save(box, store=store, record_policy="descriptive")
    cdef_id = format_cdef_id(box.definition.stable_hash())

    arguments = attach_operation_id(
        make_function_call_spec(
            "dispatch_target:argument_values",
            args=[cdef_id, f"ref({cdef_id})", {"$literal": cdef_id}],
        )
    )

    result = Dispatcher(store=store).run(arguments, environment=_env(target_module))
    assert result.result_canonical[0] == 9
    assert result.result_canonical[1] == cdef_id
    assert result.result_canonical[2] == cdef_id
