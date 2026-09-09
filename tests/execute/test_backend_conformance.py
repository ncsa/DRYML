"""Cross-backend common Execute conformance scenarios."""

from __future__ import annotations

import asyncio
from threading import Event
from pathlib import Path

import pytest

from dryml.execute.executor import Executor
from dryml.execute.output import ExecutionOutput
from dryml.execute.ray import RayBackendConfig
from dryml.execute.subprocess import SubProcessConfig
from .conftest import require_ray_integration


def _emit(value: str) -> str:
    """Write a bounded worker output record and return its ordinary value."""
    import os

    os.write(1, b"fractional-final-timeout\n")
    return value


def _sum_values(left: int, right: int) -> int:
    """Return the ordinary common-backend conformance value."""
    return left + right


@pytest.fixture(params=("subprocess", "ray"), ids=str)
def backend_config(request: pytest.FixtureRequest, tmp_path: Path):
    """Create an actual local or explicitly enabled actual-Ray backend configuration."""
    spool_directory = tmp_path / str(request.param)
    spool_directory.mkdir()
    if request.param == "subprocess":
        return SubProcessConfig(spool_directory=spool_directory)
    return RayBackendConfig(
        address=require_ray_integration(),
        spool_directory=spool_directory,
        admission_timeout=90,
        connect_timeout=60,
    )


def test_subprocess_preserves_a_positive_submillisecond_output_final_timeout(tmp_path: Path):
    """A valid fractional final-fence timeout reaches the worker without rounding to zero."""
    output = ExecutionOutput()
    executor = Executor(
        SubProcessConfig(
            spool_directory=tmp_path,
            output_final_timeout=0.0001,
        )
    )
    try:
        future = executor.submit(_emit, "result", output=output)
        assert future.result(timeout=10) == "result"
        assert "fractional-final-timeout" in output.snapshot().stdout
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_backend_configuration_alone_switches_run_submit_view_future_and_output(
    backend_config, tmp_path: Path,
):
    """Common public routes retain values, callbacks, awaits, and output on each real backend."""
    callback = Event()
    output = ExecutionOutput()
    executor = Executor(backend_config)
    try:
        assert executor.run(_sum_values, 2, 3) == 5
        view = executor.with_options(
            stream_output=False,
            output=output,
            done_callbacks=(lambda value: callback.set(),),
        )
        future = view.submit(_emit, "view-result")
        assert future.result(timeout=30) == "view-result"
        assert callback.wait(5)

        async def await_result() -> str:
            return await future

        assert asyncio.run(await_result()) == "view-result"
        snapshot = output.snapshot()
        assert "fractional-final-timeout" in snapshot.stdout
        assert snapshot.complete
        future.cleanup(timeout=30)
        assert "fractional-final-timeout" in output.snapshot().stdout
    finally:
        executor.close(cancel=True, timeout=30)
