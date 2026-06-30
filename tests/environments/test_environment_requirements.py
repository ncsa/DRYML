import pytest

import dryml.environments as envs


def record(packages=None, *, python="3.11.8", features=("dryml.environments.v1",), tags=("dev",)):
    packages = packages or {"dryml": "0.3.0", "torch": "2.5.1"}
    return envs.EnvironmentRecord(
        python=envs.PythonRecord(python, "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={name: envs.PackageRecord(name, version) for name, version in packages.items()},
        dryml=envs.DrymlRuntimeRecord(
            version="0.3.0",
            execution_protocol="1",
            schema_versions={"environment_record": 1},
            features=features,
        ),
        tags=tags,
    )


def issue_codes(report):
    return [issue.code for issue in report.issues]


def test_requirement_roundtrip_and_reordered_id_stability():
    left = envs.EnvironmentRequirement(
        requirements=("torch>=2", "dryml>=0.3"),
        excludes=("tensorflow", "jax"),
        capabilities=("b", "a"),
        tags=("dev", "torch"),
    )
    right = envs.EnvironmentRequirement(
        requirements=("dryml>=0.3", "torch>=2"),
        excludes=("jax", "tensorflow"),
        capabilities=("a", "b"),
        tags=("torch", "dev"),
    )
    assert left.to_data() == envs.EnvironmentRequirement.from_data(left.to_data()).to_data()
    assert left.id == right.id


def test_requirement_check_compatible():
    req = envs.EnvironmentRequirement(
        python=">=3.10,<3.13",
        requirements=("dryml>=0.3", "torch>=2.4,<2.7"),
        excludes=("tensorflow",),
        capabilities=("dryml.environments.v1",),
        tags=("dev",),
        schema_versions={"environment_record": "==1"},
    )
    report = req.check(record())
    assert report.status == "compatible"
    assert report.ok


def test_requirement_check_missing_package_and_warn_policy():
    req = envs.EnvironmentRequirement(requirements=("tensorflow>=2",))
    report = req.check(record())
    assert report.status == "incompatible"
    assert issue_codes(report) == ["package_missing"]
    warn = req.check(record(), policy="warn")
    assert warn.status == "warning"
    assert warn.issues[0].severity == "warning"


def test_requirement_check_version_mismatch_and_unknown_version():
    mismatch = envs.EnvironmentRequirement(requirements=("torch>=3",)).check(record())
    assert mismatch.status == "incompatible"
    assert "package_version_mismatch" in issue_codes(mismatch)
    unknown = envs.EnvironmentRequirement(requirements=("mystery>=1",)).check(record({"mystery": None}))
    assert unknown.status == "unknown"
    assert "package_version_unknown" in issue_codes(unknown)


def test_requirement_check_markers_excludes_python_and_capabilities():
    marker_false = envs.EnvironmentRequirement(requirements=("missing-pkg>=1; python_version < '1'",))
    assert marker_false.check(record()).status == "compatible"
    marker_true = envs.EnvironmentRequirement(requirements=("missing-pkg>=1; python_version >= '3'",))
    assert marker_true.check(record()).issues[0].code == "package_missing"
    excluded = envs.EnvironmentRequirement(excludes=("torch",)).check(record())
    assert issue_codes(excluded) == ["package_excluded_present"]
    py = envs.EnvironmentRequirement(python=">=9").check(record())
    assert issue_codes(py) == ["python_version_mismatch"]
    cap = envs.EnvironmentRequirement(capabilities=("custom.cap",)).check(record())
    assert issue_codes(cap) == ["capability_missing"]


def test_requirement_check_schema_and_dryml_runtime():
    schema = envs.EnvironmentRequirement(schema_versions={"missing": "==1"}).check(record())
    assert issue_codes(schema) == ["schema_missing"]
    no_runtime_record = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.11.8", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={},
        dryml=None,
    )
    strict = envs.EnvironmentRequirement().check(no_runtime_record, policy="strict")
    assert strict.status == "incompatible"
    assert issue_codes(strict) == ["dryml_runtime_missing"]
    protocol = envs.EnvironmentRequirement(dryml_protocol=">=2").check(record())
    assert issue_codes(protocol) == ["dryml_protocol_mismatch"]


def test_tag_semantics_are_warning_by_default_and_error_in_strict():
    req = envs.EnvironmentRequirement(tags=("gpu",))
    compatible = req.check(record())
    assert compatible.status == "warning"
    assert compatible.issues[0].severity == "warning"
    strict = req.check(record(), policy="strict")
    assert strict.status == "incompatible"
    assert strict.issues[0].severity == "error"


def test_invalid_policy_and_requirement_fail_clearly():
    with pytest.raises(envs.EnvironmentCompatibilityError):
        envs.EnvironmentRequirement().check(record(), policy="bad")
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.EnvironmentRequirement(requirements=("not valid !!!",))


def test_policy_ignore_skips_checks():
    report = envs.EnvironmentRequirement(requirements=("missing-package",)).check(record(), policy="ignore")
    assert report.status == "compatible"
    assert report.issues == ()
