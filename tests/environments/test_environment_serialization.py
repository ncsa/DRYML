import pytest
from pathlib import Path

import dryml.environments as envs
from dryml.environments.ids import content_id
from dryml.environments.serialization import canonical_json_bytes, canonical_json_dumps, deep_freeze_json, json_ready
from dryml.formats.ids import content_id as format_content_id


def test_schema_constants_are_visible():
    assert envs.ENVIRONMENT_RECORD_SCHEMA_VERSION == 1
    assert envs.ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION == 1
    assert envs.ENVIRONMENT_SPEC_SCHEMA_VERSION == 1
    assert envs.ENVIRONMENT_LOCK_REF_SCHEMA_VERSION == 1
    assert envs.COMPATIBILITY_REPORT_SCHEMA_VERSION == 1
    assert envs.ENVIRONMENT_FRAGMENT_SCHEMA_VERSION == 1


def test_canonical_serialization_stable_under_dict_order():
    left = {"b": [2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [2, 1]}
    assert canonical_json_dumps(left) == canonical_json_dumps(right)
    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_deep_freeze_json_canonicalizes_sets_and_lists():
    frozen = deep_freeze_json({"items": {"b", "a"}, "nested": [1, {"x": True}]})

    assert canonical_json_dumps(frozen) == '{"items":["a","b"],"nested":[1,{"x":true}]}'


def test_deep_freeze_rejects_non_string_mapping_keys():
    with pytest.raises(envs.EnvironmentSerializationError, match="mapping keys must be strings"):
        deep_freeze_json({1: "integer-key"})


def test_deep_freeze_rejects_string_key_collision():
    with pytest.raises(envs.EnvironmentSerializationError, match="mapping keys must be strings"):
        deep_freeze_json({"1": "string-key", 1: "integer-key"})


def test_json_ready_does_not_silently_collapse_keys():
    with pytest.raises(envs.EnvironmentSerializationError, match="mapping keys must be strings"):
        json_ready({"1": "string-key", 1: "integer-key"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected_during_freeze(value):
    with pytest.raises(envs.EnvironmentSerializationError, match="floats must be finite"):
        deep_freeze_json({"value": value})


def test_content_id_changes_with_schema_version_and_data():
    first = content_id("envrec", 1, {"value": 1})
    assert first == content_id("envrec", 1, {"value": 1})
    assert first != content_id("envrec", 2, {"value": 1})
    assert first != content_id("envrec", 1, {"value": 2})


def test_environment_ids_match_generic_format_ids():
    record = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.11.8", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={"dryml": envs.PackageRecord("dryml", "0.3.0")},
        dryml=envs.DrymlRuntimeRecord(schema_versions={"environment_record": 1}),
    )
    requirement = envs.EnvironmentRequirement(requirements=("dryml>=0.3",))
    spec = envs.PythonExecutableSpec("/usr/bin/python", env={"A": "B"})
    lock = envs.EnvironmentLockRef("conda-lock", "file:///tmp/conda-lock.yml", digest="sha256:abc")

    assert record.id == format_content_id("envrec", record.schema_version, record.to_data())
    assert requirement.id == format_content_id("envreq", requirement.schema_version, requirement.to_data())
    assert spec.id == format_content_id("envspec", spec.schema_version, spec.to_data())
    assert lock.id == format_content_id("envlock", lock.schema_version, lock.to_data())


def test_from_data_ignores_unknown_fields_and_preserves_schema_version():
    data = envs.EnvironmentRequirement(requirements=("dryml",)).to_data()
    data["unknown_future_field"] = {"ok": True}
    data["schema_version"] = 7
    req = envs.EnvironmentRequirement.from_data(data)
    assert req.schema_version == 7
    assert req.requirements == ("dryml",)
    assert req.id.startswith("envreq-v7-")


def test_distribution_name_and_requirement_normalization():
    assert envs.normalize_distribution_name("Foo_Bar.baz") == "foo-bar-baz"
    assert envs.normalize_requirement_string("  Foo_Bar >= 1 ; python_version >= '3.10' ") == "foo-bar>=1; python_version >= \"3.10\""


def test_invalid_requirement_string_has_context():
    with pytest.raises(envs.EnvironmentRequirementError) as excinfo:
        envs.normalize_requirement_string("not a valid req !!!")
    assert "requirement" in excinfo.value.context


def test_errors_carry_structured_context():
    err = envs.DrymlEnvironmentError("boom", context={"code": "x"})
    assert str(err) == "boom"
    assert err.context == {"code": "x"}


def test_environment_docs_page_exists_and_is_linked():
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    text = (docs_dir / "environments.md").read_text()
    toc = (docs_dir / "table_of_content.md").read_text()
    assert "inspect_current" in text
    assert "EnvironmentRequirement" in text
    assert "probe_python" in text
    assert "CondaEnvironmentSpec" in text
    assert "req" in text and "add_req" in text
    assert "environments.md" in toc
