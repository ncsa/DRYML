"""Opt-in fresh-process checks against an installed TensorFlow runtime."""

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
        "set DRYML_RUN_REAL_FRAMEWORK_TESTS=1 to run real TensorFlow checks",
        allow_module_level=True,
    )


def _run_fresh_process(script: str, *, gpu: bool = False) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
        f"fresh TensorFlow process exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if completed.returncode == 0 and records and "skip" in records[-1]:
        pytest.skip(str(records[-1]["skip"]))
    assert completed.returncode == 0, context
    assert records and "versions" in records[-1], context
    print("real-framework context: " + json.dumps(records[-1]["versions"], sort_keys=True))
    return records[-1]


def test_tensorflow_cpu_raw_import_enforces_visibility_and_threads():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        if importlib.util.find_spec("tensorflow") is None:
            print(json.dumps({"skip": "TensorFlow is not installed"}))
            raise SystemExit(0)

        from dryml import session

        session.manage(cpus=1)
        import tensorflow as tf

        version = str(getattr(tf, "__version__", ""))
        print(json.dumps({"versions": {"tensorflow": version}}), flush=True)
        statuses = session.current().statuses
        assert version
        assert tf.config.get_visible_devices("GPU") == []
        assert tf.config.threading.get_intra_op_parallelism_threads() == 1
        assert statuses["tensorflow:tensorflow:visibility"] == "visibility-enforced"
        assert statuses["tensorflow:tensorflow:threads"] == "framework-configured"
        print(json.dumps({"versions": {"tensorflow": version}, "gpu_count": 0}))
        """
    )

    assert result["gpu_count"] == 0


@pytest.mark.skipif(
    not _RUN_GPU,
    reason="set DRYML_RUN_GPU_TESTS=1 on a provisioned host to run TensorFlow GPU checks",
)
def test_tensorflow_gpu_raw_import_enforces_assigned_visibility_and_threads():
    result = _run_fresh_process(
        """
        import importlib.util
        import json

        if importlib.util.find_spec("tensorflow") is None:
            print(json.dumps({"skip": "TensorFlow is not installed"}))
            raise SystemExit(0)

        from dryml import session
        from dryml.worlds import local_inventory

        if not local_inventory().accelerators.get("gpu"):
            print(json.dumps({"skip": "no provisioned GPU is visible to DRYML"}))
            raise SystemExit(0)
        session.manage(cpus=1, gpus=1)
        import tensorflow as tf

        version = str(getattr(tf, "__version__", ""))
        print(json.dumps({"versions": {"tensorflow": version}}), flush=True)
        statuses = session.current().statuses
        visible = tf.config.get_visible_devices("GPU")
        assert version
        assert len(visible) == 1
        assert tf.config.threading.get_intra_op_parallelism_threads() == 1
        assert statuses["tensorflow:tensorflow:visibility"] == "visibility-enforced"
        assert statuses["tensorflow:tensorflow:threads"] == "framework-configured"
        print(json.dumps({"versions": {"tensorflow": version}, "gpu_count": len(visible)}))
        """,
        gpu=True,
    )

    assert result["gpu_count"] == 1
