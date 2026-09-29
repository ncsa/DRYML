import pytest

import dryml.environments as envs


def test_family_metadata_is_non_identifying_and_cross_family_ids_fail():
    left = envs.PythonExecutableSpec("/one/python", metadata={"observed": "first"})
    right = envs.PythonExecutableSpec("/one/python", metadata={"observed": "second"})
    assert left.semantic_id == right.semantic_id
    data = left.to_data()
    data["id"] = envs.EnvironmentLockRef("lock", "https://example.test/lock").semantic_id
    with pytest.raises(Exception, match="attached semantic ID"):
        envs.spec_from_data(data)


def test_record_identity_excludes_observation_only_fields():
    def record(path, location, revision, details):
        return envs.EnvironmentRecord(
            python=envs.PythonRecord("3.12", "CPython", executable=path),
            platform=envs.PlatformRecord("Linux", "x", "x", "x", "x"),
            distributions={"demo": envs.PackageRecord("demo", "1", location=location)},
            dryml=envs.DrymlRuntimeRecord(version="1", git_revision=revision),
            details=details,
        )

    assert record("/one", "/a", "one", {"trace": 1}).semantic_id == record("/two", "/b", "two", {"trace": 2}).semantic_id


def test_environment_record_distribution_exception_does_not_relax_ordinary_maps():
    packages = {f"package-{index}": envs.PackageRecord(f"package-{index}", "1") for index in range(65)}
    record = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.12", "CPython"),
        platform=envs.PlatformRecord("Linux", "x", "x", "x", "x"),
        distributions=packages,
    )
    assert len(record.distributions) == 65

    with pytest.raises(Exception, match="entry bound"):
        envs.EnvironmentRecord(
            python=record.python,
            platform=record.platform,
            details={f"key-{index}": index for index in range(65)},
        )
