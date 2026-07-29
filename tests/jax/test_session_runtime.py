"""Opt-in fresh-process checks against installed JAX and JAXLIB runtimes."""

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
        "set DRYML_RUN_REAL_FRAMEWORK_TESTS=1 to run real JAX/JAXLIB checks",
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
        f"fresh JAX/JAXLIB process exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if completed.returncode == 0 and records and "skip" in records[-1]:
        pytest.skip(str(records[-1]["skip"]))
    assert completed.returncode == 0, context
    assert records and "versions" in records[-1], context
    print("real-framework context: " + json.dumps(records[-1]["versions"], sort_keys=True))
    return records[-1]


def test_jaxlib_then_jax_cpu_raw_import_reports_supported_controls():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        missing = [name for name in ("jax", "jaxlib") if importlib.util.find_spec(name) is None]
        if missing:
            print(json.dumps({"skip": "missing installed framework roots: " + ", ".join(missing)}))
            raise SystemExit(0)

        from dryml import session

        session.manage(cpus=1)
        import jaxlib

        pending = session.current().statuses
        assert pending["jax:jaxlib:visibility"] == "visibility-enforced"
        assert pending["jax:jaxlib:threads"] == "pending-import"

        import jax

        versions = {
            "jax": str(getattr(jax, "__version__", "")),
            "jaxlib": str(getattr(jaxlib, "__version__", "")),
        }
        print(json.dumps({"versions": versions}), flush=True)
        statuses = session.current().statuses
        devices = jax.devices()
        assert all(versions.values())
        assert devices and all(device.platform == "cpu" for device in devices)
        assert statuses["jax:jax:visibility"] == "visibility-enforced"
        assert statuses["jax:jax:threads"] == "unsupported"
        assert statuses["jax:jaxlib:visibility"] == "visibility-enforced"
        assert statuses["jax:jaxlib:threads"] == "pending-import"
        print(json.dumps({"versions": versions, "platforms": sorted({device.platform for device in devices})}))
        """
    )

    assert result["platforms"] == ["cpu"]


@pytest.mark.skipif(
    not _RUN_GPU,
    reason="set DRYML_RUN_GPU_TESTS=1 on a provisioned host to run JAX GPU checks",
)
def test_jaxlib_then_jax_gpu_raw_import_reports_supported_controls():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        missing = [name for name in ("jax", "jaxlib") if importlib.util.find_spec(name) is None]
        if missing:
            print(json.dumps({"skip": "missing installed framework roots: " + ", ".join(missing)}))
            raise SystemExit(0)

        from dryml import session
        from dryml.worlds import local_inventory

        if not local_inventory().accelerators.get("gpu"):
            print(json.dumps({"skip": "no provisioned GPU is visible to DRYML"}))
            raise SystemExit(0)
        session.manage(cpus=1, gpus=1)
        import jaxlib
        import jax

        versions = {
            "jax": str(getattr(jax, "__version__", "")),
            "jaxlib": str(getattr(jaxlib, "__version__", "")),
        }
        print(json.dumps({"versions": versions}), flush=True)
        statuses = session.current().statuses
        devices = jax.devices("gpu")
        assert all(versions.values())
        assert len(devices) == 1
        assert statuses["jax:jax:visibility"] == "visibility-enforced"
        assert statuses["jax:jax:threads"] == "unsupported"
        assert statuses["jax:jaxlib:visibility"] == "visibility-enforced"
        assert statuses["jax:jaxlib:threads"] == "pending-import"
        print(json.dumps({"versions": versions, "gpu_count": len(devices)}))
        """,
        gpu=True,
    )

    assert result["gpu_count"] == 1
