import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from dryml.core import ObjectId, object_namespace


def test_object_namespace_nests_restores_and_allows_explicit_empty():
    assert ObjectId().namespace == ()
    with object_namespace("team"):
        assert ObjectId().namespace == ("team",)
        assert ObjectId(()).namespace == ()
        with object_namespace("experiment"):
            assert ObjectId().namespace == ("team", "experiment")
        assert ObjectId().namespace == ("team",)
    assert ObjectId().namespace == ()


def test_object_namespace_restores_after_exception():
    with pytest.raises(RuntimeError):
        with object_namespace("temporary"):
            raise RuntimeError("stop")
    assert ObjectId().namespace == ()


def test_object_namespace_is_thread_and_task_local():
    with object_namespace("main"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(lambda: ObjectId().namespace).result() == ()

        async def child():
            with object_namespace("task"):
                return ObjectId().namespace

        assert asyncio.run(child()) == ("main", "task")
        assert ObjectId().namespace == ("main",)
