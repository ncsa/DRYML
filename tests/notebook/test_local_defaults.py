from __future__ import annotations

import json
import operator
import subprocess
import sys

import dryml
import pytest

from dryml.dispatch import Dispatcher
from dryml.core.store.dir import DirStore
from dryml.environments import CurrentEnvironmentSpec, EnvironmentRegistry
from dryml.runtime import active_runtime
from dryml.runtime.allocation import is_no_allocation
from dryml.worlds import WorldSpec


def test_notebook_session_defaults_are_persistent_and_reset_semantically():
    dryml.session.reset()
    try:
        fresh = dryml.session.current()
        assert fresh.mode == "python"
        assert fresh.allocation is None
        assert is_no_allocation(fresh.runtime.allocation)

        managed = dryml.session.manage(cpus=1)
        requested = dryml.session.worker_world_request(cpus=1)
        assert managed.mode == "managed"
        assert requested.allocation == managed.allocation
        assert requested.requested_world is not None
        assert requested.controls["memory"] in {"undeclared", "declarative"}
        assert requested.statuses["visibility"] == "visibility-enforced"

        reset = dryml.session.reset()
        assert reset.mode == "python"
        assert reset.allocation is None
        assert reset.requested_world is None
        assert reset.environment.requirements == ()
        assert is_no_allocation(reset.runtime.allocation)
    finally:
        dryml.session.reset()


def test_advanced_context_defaults_are_selected_and_restored_without_allocation():
    requested_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}})
    before_runtime = active_runtime()
    previous_environment = dryml.environments.set_current(CurrentEnvironmentSpec())
    previous_world = dryml.worlds.set_current(requested_world)
    try:
        assert active_runtime() is before_runtime
        assert is_no_allocation(active_runtime().allocation)
        with dryml.environments.use(CurrentEnvironmentSpec()), dryml.worlds.use(WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {}}}})):
            assert dryml.worlds.current() != requested_world
            assert active_runtime() is before_runtime
            assert is_no_allocation(active_runtime().allocation)
        assert dryml.worlds.current() == requested_world
        assert active_runtime() is before_runtime
    finally:
        dryml.environments.set_current(previous_environment) if previous_environment is not None else dryml.environments.reset_current()
        dryml.worlds.set_current(previous_world) if previous_world is not None else dryml.worlds.reset_current()


def test_notebook_registry_explain_is_explicit_repeatable_and_allocation_free(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    registry = EnvironmentRegistry()
    entry = registry.register("current", CurrentEnvironmentSpec())

    explanation = Dispatcher(store=store, environment_registry=registry).explain(
        operator.add,
        args=(1, 2),
    )
    repeated = Dispatcher(store=store, environment_registry=registry).explain(
        operator.add,
        args=(1, 2),
    )
    summary = {
        "launchable": explanation.launchable,
        "environment_source": explanation.resolution.environment_selection.source,
        "world_source": explanation.resolution.world_selection.source,
        "diagnostic_count": len(explanation.resolution.diagnostics),
    }
    repeated_summary = {
        "launchable": repeated.launchable,
        "environment_source": repeated.resolution.environment_selection.source,
        "world_source": repeated.resolution.world_selection.source,
        "diagnostic_count": len(repeated.resolution.diagnostics),
    }

    assert explanation.resolution.environment_selection.source == "resolver"
    assert registry.list() == (entry,)
    assert repeated_summary == summary
    with pytest.raises(Exception):
        registry.register("current", CurrentEnvironmentSpec())
    assert is_no_allocation(active_runtime().allocation)
    assert not store.records.records_dir.exists()


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
