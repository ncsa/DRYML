"""Small runtime/world/dispatch dogfooding example."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import dryml
from dryml.core.store.dir import DirStore
from dryml.environments import PythonExecutableSpec


@dryml.env.req(packages={"numpy": ">=1.26"})
@dryml.world.req(cpus={"min": 1})
@dryml.world.default(cpus=1, memory="1GiB")
@dryml.runtime.default(mode="worker", device_visibility={"policy": "assigned"})
def summarize_values(values):
    """Run inside a worker allocation and return a simple summary."""

    allocation = dryml.runtime.require_worker_allocation("summarize_values needs a worker allocation")
    return {"count": len(values), "sum": sum(values), "cpus": tuple(allocation.cpus)}


def make_operation():
    """Build the canonical operation spec for local subprocess dispatch."""

    return dryml.operations.attach_operation_id(
        dryml.operations.make_function_call_spec(
            "runtime_world_dispatch_demo:summarize_values",
            args=[[1, 2, 3]],
        )
    )


if __name__ == "__main__":
    environment = PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(Path(__file__).resolve().parent),),
    ).to_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DirStore(Path(tmpdir) / "store", query_index="none")
        result = dryml.dispatch.run(make_operation(), backend="local_subprocess", environment=environment, store=store)
        print(result.status, result.result_canonical)
