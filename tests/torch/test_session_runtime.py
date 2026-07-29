"""Opt-in fresh-process checks against an installed PyTorch runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


_RUN_REAL = os.environ.get("DRYML_RUN_REAL_FRAMEWORK_TESTS") == "1"
_RUN_GPU = os.environ.get("DRYML_RUN_GPU_TESTS") == "1"

if not _RUN_REAL:
    pytest.skip(
        "set DRYML_RUN_REAL_FRAMEWORK_TESTS=1 to run real PyTorch checks",
        allow_module_level=True,
    )


def _run_fresh_process(script: str, *, gpu: bool = False) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if gpu:
        environment.pop("JAX_PLATFORMS", None)
    else:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
            }
        )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script)],
        capture_output=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        env=environment,
        text=True,
        timeout=180,
    )
    records = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    context = (
        f"fresh PyTorch process exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if completed.returncode == 0 and records and "skip" in records[-1]:
        pytest.skip(str(records[-1]["skip"]))
    assert completed.returncode == 0, context
    assert records and "versions" in records[-1], context
    print("real-framework context: " + json.dumps(records[-1]["versions"], sort_keys=True))
    return records[-1]


def test_torch_cpu_raw_import_enforces_visibility_and_threads():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        if importlib.util.find_spec("torch") is None:
            print(json.dumps({"skip": "PyTorch is not installed"}))
            raise SystemExit(0)

        from dryml import session

        session.manage(cpus=1)
        import torch

        version = str(getattr(torch, "__version__", ""))
        print(json.dumps({"versions": {"torch": version}}), flush=True)
        statuses = session.current().statuses
        assert version
        assert torch.cuda.device_count() == 0
        assert torch.get_num_threads() == 1
        assert statuses["torch:torch:visibility"] == "visibility-enforced"
        assert statuses["torch:torch:threads"] == "framework-configured"
        print(json.dumps({"versions": {"torch": version}, "gpu_count": 0}))
        """
    )

    assert result["gpu_count"] == 0


@pytest.mark.skipif(
    not _RUN_GPU,
    reason="set DRYML_RUN_GPU_TESTS=1 on a provisioned host to run PyTorch GPU checks",
)
def test_torch_gpu_raw_import_enforces_assigned_visibility_and_threads():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        if importlib.util.find_spec("torch") is None:
            print(json.dumps({"skip": "PyTorch is not installed"}))
            raise SystemExit(0)

        from dryml import session
        from dryml.worlds import local_inventory

        if not local_inventory().accelerators.get("gpu"):
            print(json.dumps({"skip": "no provisioned GPU is visible to DRYML"}))
            raise SystemExit(0)
        session.manage(cpus=1, gpus=1)
        import torch

        version = str(getattr(torch, "__version__", ""))
        print(json.dumps({"versions": {"torch": version}}), flush=True)
        statuses = session.current().statuses
        count = torch.cuda.device_count()
        assert version
        assert count == 1
        assert torch.get_num_threads() == 1
        assert statuses["torch:torch:visibility"] == "visibility-enforced"
        assert statuses["torch:torch:threads"] == "framework-configured"
        print(json.dumps({"versions": {"torch": version}, "gpu_count": count}))
        """,
        gpu=True,
    )

    assert result["gpu_count"] == 1
