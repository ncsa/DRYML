"""Run importable functions and stored CDef methods without manual OperationSpecs.

For advanced canonical operation-IR construction, see ``docs/operations.md``.
"""

from __future__ import annotations

import importlib
import sys
import tempfile

import dryml
from dryml.core.object import Object
from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.environments import PythonExecutableSpec


def add(left: int, right: int) -> int:
    """Return a deterministic function-dispatch result."""

    return left + right


class Counter(Object):
    """A small stored object with one dispatchable method."""

    def __init__(self, start: int):
        super().__init__()
        self.start = start

    def plus(self, value: int) -> int:
        """Return the stored starting value plus *value*."""

        return self.start + value


def main() -> None:
    """Run both Python-shaped dispatch forms with an owned temporary Store."""

    environment = PythonExecutableSpec(
        sys.executable, pythonpath_policy="inherit"
    ).to_data()
    example_module = importlib.import_module("examples.dispatch.python_shaped_dispatch")
    with tempfile.TemporaryDirectory(prefix="dryml-dispatch-example-") as directory:
        store = DirStore(f"{directory}/store", query_index="none")
        function_result = dryml.dispatch.run(
            example_module.add, store=store, args=(2, 3), environment=environment
        )
        assert function_result.status == "ok"
        assert function_result.result_canonical == 5

        counter = example_module.Counter(10)
        Repo(stores=[store]).save(counter, store=store, record_policy="none")
        method_result = dryml.dispatch.run(
            counter.definition,
            "plus",
            store=store,
            args=(7,),
            environment=environment,
        )
        assert method_result.status == "ok"
        assert method_result.result_canonical == 17


if __name__ == "__main__":
    main()
