import operator

import pytest

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, PickledCallable
from dryml.dispatch.errors import DispatchPlanningError
from dryml.environments import PythonExecutableSpec


def test_pickled_callable_same_python_success(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")

    result = Dispatcher(store=store).run(PickledCallable(operator.add), allow_pickle=True, args=(2, 5))

    assert result.status == "ok"
    assert result.result_canonical == 7


def test_pickled_callable_rejects_mismatched_python(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec("/not/current/python").to_data()

    with pytest.raises(DispatchPlanningError, match="same Python"):
        Dispatcher(store=store).plan(PickledCallable(operator.add), allow_pickle=True, args=(1, 2), environment=env)


def test_pickled_callable_identity_includes_pickle_payload(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    dispatcher = Dispatcher(store=store)

    left_plan = dispatcher.plan(PickledCallable(operator.add), allow_pickle=True, args=(2, 3))
    right_plan = dispatcher.plan(PickledCallable(operator.mul), allow_pickle=True, args=(2, 3))

    assert left_plan.envelope.operation_spec["id"] != right_plan.envelope.operation_spec["id"]
    assert left_plan.dispatch_spec["id"] != right_plan.dispatch_spec["id"]
