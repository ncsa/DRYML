"""Fresh-process session contracts that cannot safely share framework state."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _run(source: str, *args: str, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", source, *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("policy", ("strict", "warn"))
def test_startup_override_is_visible_then_a_facade_mutation_supersedes_it(policy):
    environment = dict(os.environ, DRYML_RUNTIME_ENFORCEMENT=policy)
    _run(
        "import dryml.runtime as runtime\n"
        "from dryml import session\n"
        f"assert runtime.active_runtime().enforcement.value == {policy!r}\n"
        "assert session.mode() == 'python'\n"
        "session.set_mode(mode='python')\n"
        "assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.OFF\n"
        "with runtime.enter_runtime(runtime.RuntimeMode.ORCHESTRATOR, enforcement='warn'):\n"
        "    assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.WARN\n"
        "assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.OFF\n",
        env=environment,
    )


def test_facade_raw_fake_import_finalizes_status_and_identical_followup_is_a_noop(tmp_path):
    package = tmp_path / "tensorflow"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import os\n"
        "SEEN = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "class _Config:\n"
        "    @staticmethod\n"
        "    def get_physical_devices(kind): return ()\n"
        "    @staticmethod\n"
        "    def set_visible_devices(devices, kind): pass\n"
        "    @staticmethod\n"
        "    def get_visible_devices(kind): return ()\n"
        "config = _Config()\n",
        encoding="utf-8",
    )
    _run(
        "import importlib, sys\n"
        "from dryml import session\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "pending = session.set_mode(mode='orchestrator')\n"
        "assert pending.statuses['tensorflow:tensorflow:visibility'] == 'pending-import'\n"
        "module = importlib.import_module('tensorflow')\n"
        "assert module.SEEN == ''\n"
        "settled = session.current()\n"
        "assert settled.statuses['tensorflow:tensorflow:visibility'] == 'visibility-enforced'\n"
        "number = settled.generation\n"
        "assert session.set_mode('orchestrator').generation == number\n"
        "assert session.require_env('dryml>=0').statuses['tensorflow:tensorflow:visibility'] == 'visibility-enforced'\n",
        str(tmp_path),
    )


def test_failed_mandatory_raw_fake_import_poison_the_session(tmp_path):
    package = tmp_path / "tensorflow"
    package.mkdir()
    (package / "__init__.py").write_text("class _Config: pass\nconfig = _Config()\n", encoding="utf-8")
    _run(
        "import importlib, sys\n"
        "from dryml import session\n"
        "from dryml.runtime.errors import FrameworkImportSafetyError\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "session.set_mode('orchestrator')\n"
        "try:\n"
        "    importlib.import_module('tensorflow')\n"
        "except FrameworkImportSafetyError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('mandatory import unexpectedly succeeded')\n"
        "assert session.current().health == 'failed'\n",
        str(tmp_path),
    )
