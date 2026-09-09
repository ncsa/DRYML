"""Dependency-free contract checks for the optional Execute Ray backend."""

from __future__ import annotations

import sys
from time import monotonic

import pytest

from dryml.execute.output import ExecutionOutput
from dryml.execute.admission import _admit_observed_logical
from dryml.execute.ray import RayBackendConfig, RayFuture, _logical_world, _native_options, _requested_amounts, _resource_amounts
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement


def test_ray_config_and_future_are_inert_without_importing_ray(monkeypatch):
    """Construction creates no SDK object and native access starts empty."""
    monkeypatch.delitem(sys.modules, "ray", raising=False)

    config = RayBackendConfig(address="auto")
    backend = config.create_backend()
    future = backend.create_future("submission", ExecutionOutput())

    assert isinstance(future, RayFuture)
    assert future.object_ref is None
    assert future.server_address is None
    assert future.node_id is None
    assert "ray" not in sys.modules


@pytest.mark.parametrize("address", ["ray://127.0.0.1:10001", "http://127.0.0.1:6379", "127.0.0.1", "127.0.0.1:0"])
def test_ray_config_rejects_non_endpoint_or_provisioning_forms(address):
    """Only the closed existing-server endpoint grammar is accepted."""
    with pytest.raises(ValueError):
        RayBackendConfig(address=address)


def test_ray_native_options_omit_unconstrained_defaults():
    """An unconstrained task leaves scheduler defaults untouched."""
    assert _native_options(None) == {}
    assert _requested_amounts(None).cpus == 1.0


def test_ray_resource_observation_preserves_unknown_dimensions():
    """Native observations never manufacture memory, CPU, or device identifiers."""
    amounts = _resource_amounts({"CPU": 2.0, "GPU": 1.0})

    assert amounts.cpus == 2.0
    assert amounts.memory_bytes is None
    assert amounts.accelerators == {"gpu": 1.0}
    assert _resource_amounts({}).cpus is None


def test_ray_resource_observation_keeps_named_scheduler_keys_out_of_accelerators():
    """Only Ray GPU is an accelerator; object-store and node keys stay named."""
    amounts = _resource_amounts({"CPU": 2.0, "node:192.168.2.31": 1.0, "object_store_memory": 4.0, "TPU": 3.0})

    assert amounts.accelerators == {}
    assert amounts.named == {"node:192.168.2.31": 1.0, "object_store_memory": 4.0, "TPU": 3.0}


def test_ray_logical_cpu_grant_passes_owner_checks_without_cpu_ids():
    """Ray logical grants admit count constraints without fabricating affinity IDs."""
    requirement = WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
    })
    observed, controls = _logical_world(requirement, {
        "assigned_resources": {"CPU": 1.0},
        "accelerator_ids": {},
    })

    decision = _admit_observed_logical(
        world=requirement, observed_world=observed, controls=controls,
        deadline=monotonic() + 1,
    )

    assert decision.go
    assert decision.allocation is None
