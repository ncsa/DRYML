import os

import pytest

import dryml.execute as execute
from dryml.runtime import BOOTSTRAP_MARKER_ENV, active_runtime_mode


def runtime_mode_value():
    return active_runtime_mode().value


def process_visibility_env():
    return {
        "bootstrap": os.environ.get(BOOTSTRAP_MARKER_ENV),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hip": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "xla": os.environ.get("XLA_VISIBLE_DEVICES"),
    }


def test_inline_execute_does_not_enter_worker_runtime():
    assert execute.run(runtime_mode_value, backend="inline") == "orchestrator"


def test_inline_execute_rejects_legacy_resource_requirements():
    with pytest.raises(execute.ExecutionError):
        execute.run(runtime_mode_value, backend="inline", requirements={"torch": {"num_gpus": 1}})


def test_process_execute_applies_runtime_visibility_before_call():
    env = execute.run(process_visibility_env, backend="process", requirements={"torch": {"num_gpus": 1}})

    assert env == {"bootstrap": "1", "cuda": "0", "hip": "", "rocr": "", "xla": ""}
