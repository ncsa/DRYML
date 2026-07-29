"""Opt-in fresh-process ordering checks for all supported real frameworks."""

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
        "set DRYML_RUN_REAL_FRAMEWORK_TESTS=1 to run real multi-framework checks",
        allow_module_level=True,
    )


def _run_fresh_process(order: tuple[str, ...], *, gpu: bool = False) -> dict[str, object]:
    environment = os.environ.copy()
    environment["DRYML_REAL_FRAMEWORK_IMPORT_ORDER"] = ",".join(order)
    environment["DRYML_REAL_FRAMEWORK_GPU_COUNT"] = "1" if gpu else "0"
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
        [sys.executable, "-I", "-c", textwrap.dedent(_SCRIPT)],
        capture_output=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        env=environment,
        text=True,
        timeout=300,
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
        f"fresh multi-framework process exited {completed.returncode}; order={order!r}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if completed.returncode == 0 and records and "skip" in records[-1]:
        pytest.skip(str(records[-1]["skip"]))
    assert completed.returncode == 0, context
    assert records and "versions" in records[-1], context
    print("real-framework context: " + json.dumps(records[-1]["versions"], sort_keys=True))
    return records[-1]


_SCRIPT = """
import importlib
import importlib.util
import json
import os

roots = ("tensorflow", "torch", "jax", "jaxlib")
missing = [name for name in roots if importlib.util.find_spec(name) is None]
if missing:
    print(json.dumps({"skip": "missing installed framework roots: " + ", ".join(missing)}))
    raise SystemExit(0)

from dryml import session

gpu_count = int(os.environ["DRYML_REAL_FRAMEWORK_GPU_COUNT"])
if gpu_count:
    from dryml.worlds import local_inventory
    if not local_inventory().accelerators.get("gpu"):
        print(json.dumps({"skip": "no provisioned GPU is visible to DRYML"}))
        raise SystemExit(0)
session.manage(cpus=1, gpus=gpu_count)

order = tuple(os.environ["DRYML_REAL_FRAMEWORK_IMPORT_ORDER"].split(","))
for root in order:
    importlib.import_module(root)

import jax
import jaxlib
import tensorflow as tf
import torch

versions = {
    "jax": str(getattr(jax, "__version__", "")),
    "jaxlib": str(getattr(jaxlib, "__version__", "")),
    "tensorflow": str(getattr(tf, "__version__", "")),
    "torch": str(getattr(torch, "__version__", "")),
}
print(json.dumps({"versions": versions}), flush=True)
statuses = session.current().statuses
assert all(versions.values())
assert len(tf.config.get_visible_devices("GPU")) == gpu_count
assert torch.cuda.device_count() == gpu_count
assert tf.config.threading.get_intra_op_parallelism_threads() == 1
assert torch.get_num_threads() == 1
assert statuses["tensorflow:tensorflow:visibility"] == "visibility-enforced"
assert statuses["tensorflow:tensorflow:threads"] == "framework-configured"
assert statuses["torch:torch:visibility"] == "visibility-enforced"
assert statuses["torch:torch:threads"] == "framework-configured"
assert statuses["jax:jax:visibility"] == "visibility-enforced"
assert statuses["jax:jax:threads"] == "unsupported"
assert statuses["jax:jaxlib:visibility"] == "visibility-enforced"
assert statuses["jax:jaxlib:threads"] == "pending-import"
if gpu_count:
    assert len(jax.devices("gpu")) == gpu_count
else:
    assert jax.devices() and all(device.platform == "cpu" for device in jax.devices())
print(json.dumps({"versions": versions, "order": order, "gpu_count": gpu_count}))
"""


@pytest.mark.parametrize(
    "order",
    [
        ("tensorflow", "torch", "jaxlib", "jax"),
        ("jax", "jaxlib", "torch", "tensorflow"),
    ],
)
def test_cpu_raw_import_ordering_preserves_framework_controls(order):
    result = _run_fresh_process(order)

    assert tuple(result["order"]) == order
    assert result["gpu_count"] == 0


@pytest.mark.skipif(
    not _RUN_GPU,
    reason="set DRYML_RUN_GPU_TESTS=1 on a provisioned host to run multi-framework GPU checks",
)
def test_gpu_raw_import_ordering_preserves_framework_controls():
    order = ("jaxlib", "jax", "torch", "tensorflow")
    result = _run_fresh_process(order, gpu=True)

    assert tuple(result["order"]) == order
    assert result["gpu_count"] == 1
