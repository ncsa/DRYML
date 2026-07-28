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


def test_semantic_requirement_merge_intersects_and_deduplicates():
    merged = envs.EnvironmentRequirement(
        requirements=("torch>=2", "dryml>=0.3"), python=">=3.10"
    ).merge(
        envs.EnvironmentRequirement(requirements=("torch<3", "dryml>=0.3"), python="<3.13"),
        sources=("test",),
    )

    assert merged.requirements == ("dryml>=0.3", "torch<3,>=2")
    assert merged.python == "<3.13,>=3.10"
    assert merged.details["sources"] == ("test",)


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


def test_marker_uses_record_sys_platform_not_current_process():
    remote = record()
    remote = envs.EnvironmentRecord(
        python=remote.python,
        platform=envs.PlatformRecord("Windows", "1", "v", "AMD64", "Windows", sys_platform="win32"),
        distributions=remote.distributions,
        dryml=remote.dryml,
    )

    assert envs.EnvironmentRequirement(requirements=("missing-pkg; sys_platform == 'linux'",)).check(remote).status == "compatible"
    report = envs.EnvironmentRequirement(requirements=("missing-pkg; sys_platform == 'win32'",)).check(remote)
    assert issue_codes(report) == ["package_missing"]


def test_marker_uses_record_platform_machine():
    remote = record()
    remote = envs.EnvironmentRecord(
        python=remote.python,
        platform=envs.PlatformRecord("Linux", "1", "v", "aarch64", "Linux-aarch64"),
        distributions=remote.distributions,
        dryml=remote.dryml,
    )

    report = envs.EnvironmentRequirement(requirements=("missing-pkg; platform_machine == 'aarch64'",)).check(remote)

    assert issue_codes(report) == ["package_missing"]


def test_marker_uses_record_implementation_name():
    remote = record()
    remote = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.11.8", "PyPy"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions=remote.distributions,
        dryml=remote.dryml,
    )

    assert envs.marker_environment_from_record(remote)["implementation_name"] == "pypy"
    report = envs.EnvironmentRequirement(requirements=("missing-pkg; implementation_name == 'pypy'",)).check(remote)

    assert issue_codes(report) == ["package_missing"]


def test_marker_with_unknown_platform_field_reports_unknown():
    remote = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.11.8", "CPython"),
        platform=envs.PlatformRecord("", "", "", "", ""),
        distributions={},
        dryml=envs.DrymlRuntimeRecord(),
    )

    report = envs.EnvironmentRequirement(requirements=("missing-pkg; sys_platform == 'linux'",)).check(remote)

    assert report.status == "unknown"
    assert issue_codes(report) == ["marker_environment_unknown"]


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


def test_invalid_observed_dryml_protocol_reports_unknown():
    bad = record()
    bad = envs.EnvironmentRecord(
        python=bad.python,
        platform=bad.platform,
        distributions=bad.distributions,
        dryml=envs.DrymlRuntimeRecord(execution_protocol="not-a-version", schema_versions={}),
    )

    report = envs.EnvironmentRequirement(dryml_protocol=">=1").check(bad)

    assert report.status == "unknown"
    assert issue_codes(report) == ["dryml_protocol_mismatch"]


def test_invalid_required_dryml_protocol_reports_requirement_error():
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.EnvironmentRequirement(dryml_protocol="not a spec")


def test_invalid_observed_schema_version_reports_unknown():
    bad = record()
    bad = envs.EnvironmentRecord(
        python=bad.python,
        platform=bad.platform,
        distributions=bad.distributions,
        dryml=envs.DrymlRuntimeRecord(schema_versions={"environment_record": "bad"}),
    )

    report = envs.EnvironmentRequirement(schema_versions={"environment_record": ">=1"}).check(bad)

    assert report.status == "unknown"
    assert issue_codes(report) == ["schema_version_mismatch"]


def test_invalid_required_schema_version_reports_requirement_error():
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.EnvironmentRequirement(schema_versions={"environment_record": "not a spec"})


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
