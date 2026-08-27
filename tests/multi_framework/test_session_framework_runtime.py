"""Real-framework lifecycle evidence for hosted heavy jobs."""

from __future__ import annotations

import os
import subprocess
import sys


def test_real_framework_imports_publish_adapter_statuses():
    """Activate orchestration before real imports and verify every adapter."""

    code = r'''
from dryml import session

session.set_mode("orchestrator")
import jax
import jaxlib
import torch
import tensorflow

statuses = session.current().statuses
for framework in ("jax", "torch", "tensorflow"):
    assert statuses[f"{framework}:visibility"] == "visibility-enforced", statuses
'''
    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        timeout=120,
    )
