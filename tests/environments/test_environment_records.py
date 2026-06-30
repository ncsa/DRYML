import importlib.metadata as metadata
import sys
import types

import pytest

import dryml.environments as envs
from dryml.environments import introspection


def sample_record(**kwargs):
    data = {
        "python": envs.PythonRecord("3.11.8", "CPython", executable="/usr/bin/python"),
        "platform": envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        "distributions": {
            "DryML": envs.PackageRecord("DryML", "0.3.0"),
            "torch": envs.PackageRecord("torch", "2.5.1"),
        },
        "dryml": envs.DrymlRuntimeRecord(
            version="0.3.0-dev",
            execution_protocol="1",
            schema_versions={"environment_record": 1},
            features=("dryml.environments.v1", "custom.capability"),
        ),
        "kind": "venv",
        "tags": ("dev", "torch"),
        "details": {"virtual_env": "/tmp/venv"},
    }
    data.update(kwargs)
    return envs.EnvironmentRecord(**data)


def test_environment_record_roundtrip_and_id_stability():
    record = sample_record()
    clone = envs.EnvironmentRecord.from_data(record.to_data())
    assert clone.to_data() == record.to_data()
    assert clone.id == record.id


def test_environment_record_distribution_keys_are_normalized_and_order_stable():
    first = sample_record(distributions={"Foo_Bar": envs.PackageRecord("Foo_Bar", "1")})
    second = sample_record(distributions={"foo-bar": envs.PackageRecord("Foo_Bar", "1")})
    assert tuple(first.distributions) == ("foo-bar",)
    assert first.id == second.id


def test_environment_record_details_are_deeply_immutable():
    input_details = {"nested": {"items": ["a"]}}
    record = sample_record(details=input_details)
    before = record.id

    input_details["nested"]["items"].append("b")

    assert record.to_data()["details"] == {"nested": {"items": ["a"]}}
    assert record.id == before
    with pytest.raises(AttributeError):
        record.details["nested"]["items"].append("c")


def test_environment_requirement_details_are_deeply_immutable():
    input_details = {"sources": ["base"], "nested": {"enabled": True}}
    requirement = envs.EnvironmentRequirement(details=input_details)
    before = requirement.id

    input_details["sources"].append("mutated")

    assert requirement.to_data()["details"] == {"nested": {"enabled": True}, "sources": ["base"]}
    assert requirement.id == before
    with pytest.raises(AttributeError):
        requirement.details["sources"].append("x")


def test_non_json_details_are_rejected():
    with pytest.raises(envs.EnvironmentSerializationError):
        sample_record(details={"bad": object()})


def test_environment_intern_table_reuses_record_and_requirement_instances():
    table = envs.EnvironmentInternTable()
    record = sample_record()
    req = envs.EnvironmentRequirement(requirements=("dryml>=0.3",))
    assert table.intern_record(record) is record
    assert table.intern_record(envs.EnvironmentRecord.from_data(record.to_data())) is record
    assert table.intern_requirement(req) is req
    assert table.intern_requirement(envs.EnvironmentRequirement.from_data(req.to_data())) is req


class FakeDist:
    version = "1.2.3"
    files = ()
    metadata = {"Name": "Fake_Pkg"}

    def read_text(self, name):
        if name == "INSTALLER":
            return "pip\n"
        return None


def test_inspect_current_uses_importlib_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "distributions", lambda: [FakeDist()])
    monkeypatch.setattr(metadata, "version", lambda name: "0.3.0-dev")
    record = introspection.inspect_current()
    assert record.python.version
    assert record.platform.system
    assert record.distributions["fake-pkg"].version == "1.2.3"
    assert record.dryml.features == ("dryml.environments.v1",)


def test_inspect_current_does_not_import_heavy_modules(monkeypatch):
    before = {name: sys.modules.get(name) for name in ("tensorflow", "torch", "jax", "ray")}
    monkeypatch.setattr(metadata, "distributions", lambda: [])
    record = introspection.inspect_current()
    assert record.kind in {"conda", "venv", "system"}
    for name, module in before.items():
        assert sys.modules.get(name) is module


def test_environment_kind_detection(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/test")
    assert introspection._environment_kind() == "conda"
    monkeypatch.delenv("CONDA_PREFIX")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/venv")
    assert introspection._environment_kind() == "venv"
    monkeypatch.delenv("VIRTUAL_ENV")
    monkeypatch.setattr(introspection, "sys", types.SimpleNamespace(prefix="/a", base_prefix="/a"))
    assert introspection._environment_kind() == "system"
