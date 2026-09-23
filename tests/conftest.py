from .fixtures import store_resource_factory, create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, primary_store_set, ray
import builtins
import os
import sys

import pytest

pytest_plugins = ("tests.timing_plugin",)

try:
    from mk_ic import install
    from mk_ic import pytest_wrapper_elimination as _pwe
except ImportError:
    builtins.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
else:
    install()
    ics.configureOutput(frame_filters=[_pwe])


def _synthetic_environment_record():
    """Build the small environment record shared by opt-in test fixtures."""

    from dryml.environments import (
        DrymlRuntimeRecord,
        EnvironmentRecord,
        PackageRecord,
        PlatformRecord,
        PythonRecord,
    )

    return EnvironmentRecord(
        python=PythonRecord(
            version=".".join(str(value) for value in sys.version_info[:3]),
            implementation=sys.implementation.name,
            executable=sys.executable,
            prefix=sys.prefix,
            base_prefix=getattr(sys, "base_prefix", sys.prefix),
        ),
        platform=PlatformRecord(
            system="test",
            release="1",
            version="1",
            machine="test",
            platform="test-policy-environment",
        ),
        distributions={
            "dryml": PackageRecord("dryml", "0.3.0"),
            "policy-fixture": PackageRecord("policy-fixture", "1.0.0"),
        },
        dryml=DrymlRuntimeRecord(
            version="0.3.0",
            execution_protocol="test",
            schema_versions={"environment_record": "1.1"},
            features=("dryml.environments.v1.1",),
        ),
        kind="test",
    )


@pytest.fixture
def synthetic_environment_record(monkeypatch):
    """Return policy evidence and reject installed-distribution inventory reads."""

    from dryml.environments import introspection

    inventory_reads = []

    def reject_inventory(*_args, **_kwargs):
        inventory_reads.append(True)
        raise AssertionError(
            "policy-only test must not inspect installed distributions"
        )

    monkeypatch.setattr(introspection.metadata, "distributions", reject_inventory)
    yield _synthetic_environment_record()
    assert not inventory_reads, "policy-only test attempted host inventory"


@pytest.fixture
def fixed_snapshot_environment(monkeypatch):
    """Use fixed evidence only when snapshot capture omits its observer.

    The opt-in fixture leaves explicit observers and clocks intact and delegates
    lineage and requirement capture to the production implementation. Unlike
    :func:`synthetic_environment_record`, it does not guard unrelated environment
    admission or introspection calls made by an integration test.

    Returns:
        The fixed environment record supplied to default snapshot observations.
    """

    from dryml.core import snapshot_capture

    environment = _synthetic_environment_record()
    original_capture = snapshot_capture._capture_snapshot_evidence

    def capture(lineages, classes, *, observer=None,
                clock=snapshot_capture.current_utc_time):
        if observer is None:
            observer = lambda: environment
        return original_capture(
            lineages, classes, observer=observer, clock=clock,
        )

    monkeypatch.setattr(snapshot_capture, "_capture_snapshot_evidence", capture)
    return environment


@pytest.fixture
def fixed_managed_snapshot_environment(
        monkeypatch, fixed_snapshot_environment):
    """Use fixed evidence for managed lifecycle snapshot pre-observation.

    This explicit opt-in extends :func:`fixed_snapshot_environment` only at the
    managed runtime's pre-observation seam. Requirement collection, clocks,
    persistence, and lifecycle failure handling continue through production
    implementations.

    Returns:
        The fixed environment record supplied to managed snapshot observations.
    """

    from dryml.managed import runtime

    def preobserve():
        return lambda: fixed_snapshot_environment

    monkeypatch.setattr(runtime, "_preobserve_snapshot_environment", preobserve)
    return fixed_snapshot_environment


def pytest_sessionstart(session):
    if os.environ.get("DRYML_TEST_BOOTSTRAP_CONTEXTS") != "1":
        return

    from dryml.context.context_tracker import add_context

    # import jax needs to go before tensorflow
    # Enforce special loading order to prevent crash
    # https://github.com/pytorch/pytorch/issues/101152
    #import torch  # noqa: F401
    #import tensorflow as tf  # noqa: F401

    for ctx_name in ("jax", "torch", "tf"):
        try:
            add_context(ctx_name)
        except Exception:
            sys.modules.pop(ctx_name, None)
            pass


__all__ = [
    store_resource_factory,
    primary_store_set,
    create_name,
    create_temp_file,
    create_temp_dir,
    create_temp_named_file,
    ray,
]
