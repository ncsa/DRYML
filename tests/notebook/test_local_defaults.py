from __future__ import annotations

import json
import linecache
from pathlib import Path
import subprocess
import sys
import types

import dryml
import pytest

from dryml.dispatch import Dispatcher
from dryml.core2.store.dir import DirStore
from dryml.environments import CurrentEnvironmentSpec, EnvironmentRegistry
from dryml.operations import make_function_call_spec
from dryml.runtime import active_runtime
from dryml.runtime.allocation import is_no_allocation
from dryml.worlds import LocalResourceInventory, WorldSpec


def test_notebook_context_defaults_are_selected_and_restored_without_allocation():
    requested_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}})
    previous_environment = dryml.environments.set_current(CurrentEnvironmentSpec())
    previous_world = dryml.worlds.set_current(requested_world)
    try:
        explanation = Dispatcher().explain(lambda: None, allow_pickle=True, requirement_policy="ignore", inventory=LocalResourceInventory((0, 1)))
        assert explanation.resolution.environment_selection.source == "current"
        assert explanation.resolution.world_selection.source == "current"
        assert explanation.resolution.world_allocation_summary is None
        assert "environment_attempts=" in str(explanation)
        assert "environment=current" in str(explanation)
        assert "world=current" in str(explanation)
        assert "launchable=" in str(explanation)
        with dryml.environments.use(CurrentEnvironmentSpec()), dryml.worlds.use(WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {}}}})):
            assert dryml.worlds.current() != requested_world
        assert dryml.worlds.current() == requested_world
    finally:
        dryml.environments.set_current(previous_environment) if previous_environment is not None else dryml.environments.reset_current()
        dryml.worlds.set_current(previous_world) if previous_world is not None else dryml.worlds.reset_current()


def test_notebook_registry_explain_is_explicit_repeatable_and_allocation_free(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    registry = EnvironmentRegistry()
    entry = registry.register("current", CurrentEnvironmentSpec())

    explanation = Dispatcher(store=store, environment_registry=registry).explain(
        make_function_call_spec("operator:add", args=[1, 2]),
    )
    repeated = Dispatcher(store=store, environment_registry=registry).explain(
        make_function_call_spec("operator:add", args=[1, 2]),
    )

    assert explanation.resolution.environment_selection.source == "resolver"
    assert registry.list() == (entry,)
    assert repeated.to_data() == explanation.to_data()
    with pytest.raises(Exception):
        registry.register("current", CurrentEnvironmentSpec())
    assert is_no_allocation(active_runtime().allocation)
    assert not store.records.specs_dir.exists()


def test_notebook_explain_does_not_import_framework_modules():
    command = (
        "import json, sys; "
        "from dryml.dispatch import Dispatcher; "
        "from dryml.operations import make_function_call_spec; "
        "Dispatcher().explain(make_function_call_spec('operator:add', args=[1, 2])); "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {'torch', 'tensorflow', 'jax', 'jaxlib', 'keras', 'cupy', 'pynvml', 'py3nvml', 'nvidia'})))"
    )
    output = subprocess.check_output([sys.executable, "-c", command], text=True)

    assert json.loads(output) == []


def test_documented_local_defaults_notebook_executes_without_jupyter_or_frameworks():
    """Execute the tracked notebook's code cells in one standard-library namespace."""

    notebook = Path(__file__).resolve().parents[2] / "examples/notebooks/local_defaults_and_plain_mode.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in document["cells"] if cell["cell_type"] == "code")

    before_environment = dryml.environments.current()
    before_world = dryml.worlds.current()
    before_runtime = active_runtime()
    before_frameworks = {
        name for name in sys.modules if name.split(".", 1)[0] in {"torch", "tensorflow", "jax", "jaxlib", "keras", "cupy"}
    }
    module_name = "dryml_documentation_notebook"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                filename = f"{notebook}::cell-{index}"
                module.__file__ = filename
                linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
                exec(compile(source, filename, "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)

    after_frameworks = {
        name for name in sys.modules if name.split(".", 1)[0] in {"torch", "tensorflow", "jax", "jaxlib", "keras", "cupy"}
    }
    assert dryml.environments.current() is before_environment
    assert dryml.worlds.current() is before_world
    assert active_runtime() is before_runtime
    assert after_frameworks == before_frameworks
