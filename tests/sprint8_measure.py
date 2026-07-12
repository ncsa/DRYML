"""Reproducibly measure Sprint 8 inventory and managed probe activity.

Run ``python tests/sprint8_measure.py`` from the repository root. The script
executes the prescribed Sprint 8 matrix and prints JSON counters after pytest.
"""

from __future__ import annotations

import functools
import importlib
import json
import time

import pytest


COUNTERS = {
    "inventory": {"calls": 0, "seconds": 0.0},
    "code_probe": {"calls": 0, "seconds": 0.0},
    "environment_probe": {"calls": 0, "seconds": 0.0},
    "managed_subprocess_probe": {"calls": 0, "seconds": 0.0},
}


def _instrument(module, attribute: str, counter: str) -> None:
    original = getattr(module, attribute)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        started = time.monotonic()
        COUNTERS[counter]["calls"] += 1
        try:
            return original(*args, **kwargs)
        finally:
            COUNTERS[counter]["seconds"] += time.monotonic() - started

    setattr(module, attribute, wrapped)


def main() -> int:
    import dryml.dispatch.requirements as dispatch_requirements
    import dryml.dispatch.local_world as local_world
    import dryml.environments as environments
    import dryml.environments.registry as registry
    import dryml.environments.resolution as resolution
    import dryml.worlds as worlds
    import dryml.worlds.synthesis as synthesis

    code_probe = importlib.import_module("dryml.code.probe")
    environment_probe = importlib.import_module("dryml.environments.probe")
    inventory = importlib.import_module("dryml.worlds.inventory")

    _instrument(inventory, "local_inventory", "inventory")
    _instrument(worlds, "local_inventory", "inventory")
    _instrument(synthesis, "local_inventory", "inventory")
    _instrument(local_world, "local_inventory", "inventory")
    _instrument(code_probe, "probe_target", "code_probe")
    _instrument(dispatch_requirements, "probe_target", "code_probe")
    _instrument(environment_probe, "probe", "environment_probe")
    _instrument(environments, "probe", "environment_probe")
    _instrument(registry, "probe", "environment_probe")
    _instrument(resolution, "probe", "environment_probe")
    # These aliases are the actual bounded worker-launch seam. Counting them
    # excludes injected fake probe runners used by unit tests.
    _instrument(environment_probe, "_run_bounded_command", "managed_subprocess_probe")
    _instrument(code_probe, "_run_bounded_command", "managed_subprocess_probe")
    status = pytest.main([
        "tests/worlds/test_local_inventory.py",
        "tests/worlds/test_synthesize.py",
        "tests/environments/test_environment_registry.py",
        "tests/environments/test_resolve.py",
        "tests/dispatch/test_no_override_candidate_resolution.py",
        "tests/dispatch/test_environment_selection.py",
        "tests/dispatch/test_world_selection.py",
        "tests/dispatch/test_environment_probe_check.py",
        "tests/dispatch/test_local_world.py",
        "tests/dispatch/test_explain.py",
        "tests/notebook/test_local_defaults.py",
    ])
    print(json.dumps(COUNTERS, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
