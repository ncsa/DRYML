import operator

import pytest

from dryml.core2.store.dir import DirStore
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
