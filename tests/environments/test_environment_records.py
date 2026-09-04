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
            schema_versions={"environment_record": "1.1"},
            features=("dryml.environments.v1.1", "custom.capability"),
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

    assert record.to_data()["payload"]["details"] == {"nested": {"items": ["a"]}}
    assert record.id == before
    with pytest.raises(AttributeError):
        record.details["nested"]["items"].append("c")


def test_environment_requirement_details_are_deeply_immutable():
    input_details = {"sources": ["base"], "nested": {"enabled": True}}
    requirement = envs.EnvironmentRequirement(details=input_details)
    before = requirement.id

    input_details["sources"].append("mutated")

    assert requirement.to_data()["payload"]["details"] == {"nested": {"enabled": True}, "sources": ["base"]}
    assert requirement.id == before
    with pytest.raises(AttributeError):
        requirement.details["sources"].append("x")


def test_non_json_details_are_rejected():
    with pytest.raises(Exception, match="JSON compatible"):
        sample_record(details={"bad": object()})


def test_environment_record_rejects_non_string_detail_keys():
    with pytest.raises(Exception, match="mapping keys must be strings"):
        sample_record(details={"1": "string-key", 1: "integer-key"})


def test_environment_requirement_rejects_non_string_detail_keys():
    with pytest.raises(Exception, match="mapping keys must be strings"):
        envs.EnvironmentRequirement(details={"1": "string-key", 1: "integer-key"})


def test_environment_record_rejects_non_finite_detail_floats():
    with pytest.raises(Exception, match="floats must be finite"):
        sample_record(details={"value": float("nan")})


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
    assert record.dryml.features == ("dryml.environments.v1.1",)
    assert "environment_fragment" not in record.dryml.schema_versions


def test_fresh_record_capability_drop_changes_only_the_fragment_entry():
    """Fresh introspection no longer advertises the retired fragment format."""

    current = envs.DrymlRuntimeRecord(
        version="0.3.0",
        schema_versions={
            "environment_record": "1.1",
            "environment_requirement": "1.1",
            "environment_spec": "1.1",
            "environment_lock_ref": "1.1",
            "compatibility_report": 1,
        },
        features=("dryml.environments.v1.1",),
    )
    legacy = envs.DrymlRuntimeRecord(
        version=current.version,
        schema_versions={**current.schema_versions, "environment_fragment": "1.1"},
        features=current.features,
    )
    common = {
        "python": envs.PythonRecord("3.12.0", "CPython"),
        "platform": envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        "distributions": {},
        "kind": "system",
    }
    fresh = envs.EnvironmentRecord(dryml=current, **common)
    old = envs.EnvironmentRecord(dryml=legacy, **common)
    assert fresh.dryml.schema_versions == {key: value for key, value in legacy.schema_versions.items() if key != "environment_fragment"}
    assert fresh.id == "envrec-v1.1-743ee40a7e5c30f4578f750f67ab48af545d654800e719face81475a2c46746c"
    assert fresh.id != old.id


def test_deterministic_fresh_inspection_has_the_reduced_capability_id(monkeypatch):
    """The fresh inspection path emits the pinned reduced-capability record ID."""

    monkeypatch.setattr(metadata, "distributions", lambda: [])
    monkeypatch.setattr(metadata, "version", lambda name: "0.3.0")
    monkeypatch.setattr(introspection, "_environment_kind", lambda: "system")
    monkeypatch.setattr(introspection.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(introspection.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(introspection.platform, "system", lambda: "Linux")
    monkeypatch.setattr(introspection.platform, "release", lambda: "1")
    monkeypatch.setattr(introspection.platform, "version", lambda: "v")
    monkeypatch.setattr(introspection.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(introspection.platform, "platform", lambda: "Linux-x86_64")

    record = introspection.inspect_current()

    assert "environment_fragment" not in record.dryml.schema_versions
    assert record.id == "envrec-v1.1-5118a37db47900df6c688dae3f74e8f5caa97a671f0960245ce83e0b82741350"


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
