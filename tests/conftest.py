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


@pytest.fixture
def synthetic_environment_record(monkeypatch):
    """Return policy evidence and reject installed-distribution inventory reads."""

    from dryml.environments import (
        DrymlRuntimeRecord,
        EnvironmentRecord,
        PackageRecord,
        PlatformRecord,
        PythonRecord,
    )
    from dryml.environments import introspection

    inventory_reads = []

    def reject_inventory(*_args, **_kwargs):
        inventory_reads.append(True)
        raise AssertionError(
            "policy-only test must not inspect installed distributions"
        )

    monkeypatch.setattr(introspection.metadata, "distributions", reject_inventory)
    yield EnvironmentRecord(
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
    assert not inventory_reads, "policy-only test attempted host inventory"


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
