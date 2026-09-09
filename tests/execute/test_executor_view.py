from __future__ import annotations

import pytest

from dryml.execute.output import ExecutionOutput

from .test_executor import FakeBackend, config


def test_view_forwards_all_workload_keywords_without_control_collisions(tmp_path):
    """View keywords remain ordinary workload arguments, including control-shaped names."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    view = executor.with_options(execution_timeout=None)
    future = view.submit(
        lambda **kwargs: kwargs,
        environment="workload-environment",
        world="workload-world",
        output="workload-output",
        backend="workload-backend",
        kwargs="workload-kwargs",
        done_callbacks="workload-callbacks",
    )
    assert future.result(timeout=2) == {
        "environment": "workload-environment",
        "world": "workload-world",
        "output": "workload-output",
        "backend": "workload-backend",
        "kwargs": "workload-kwargs",
        "done_callbacks": "workload-callbacks",
    }
    assert backend.calls[0][0].execution_timeout is None
    executor.close()


def test_reused_view_creates_distinct_outputs_but_rejects_bound_output_reuse(tmp_path):
    """Views share lifecycle only; each unbound call owns an independent output object."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    view = executor.with_options()
    first = view.submit(lambda: "first")
    second = view.submit(lambda: "second")
    assert first.result(timeout=2) == "first"
    assert second.result(timeout=2) == "second"
    assert first.output is not second.output
    output = ExecutionOutput()
    bound = executor.with_options(output=output)
    assert bound.submit(lambda: 1).result(timeout=2) == 1
    with pytest.raises(RuntimeError, match="already bound"):
        bound.submit(lambda: 2)
    executor.close()


def test_parent_closure_rejects_retained_view_submissions(tmp_path):
    """A view has no independent lifecycle and cannot reopen a closed parent."""
    from dryml.execute.executor import Executor

    executor = Executor(config(tmp_path, FakeBackend()))
    view = executor.with_options()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.submit(lambda: 1)
