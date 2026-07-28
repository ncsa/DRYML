import pytest

from dryml.worlds.errors import ResourceValidationError
from dryml.worlds.resources import CountConstraint, ResourceRequirement, ResourceSpec, parse_byte_size


def test_byte_size_parsing_and_canonical_resource_spec():
    assert parse_byte_size("16GiB") == 16 * 1024**3
    assert parse_byte_size("512MiB") == 512 * 1024**2
    assert parse_byte_size("1000000000B") == 1_000_000_000

    left = ResourceSpec.from_data({"memory": 1024**3, "accelerators": {"gpu": 1}, "cpus": 2})
    right = ResourceSpec.from_data({"accelerators": {"gpu": 1}, "memory": "1GiB", "cpus": 2})
    assert left.to_data() == right.to_data()


def test_invalid_byte_units_are_rejected():
    with pytest.raises(ResourceValidationError):
        parse_byte_size("1GB")
    with pytest.raises(ResourceValidationError):
        parse_byte_size("10")


def test_count_constraint_exact_min_max_and_conflicts():
    assert CountConstraint.from_data({"exact": 2}).to_data() == {"exact": 2}
    assert CountConstraint.from_data({"min": 1, "max": 4}).satisfied_by(3)
    assert not CountConstraint.from_data({"min": 2}).satisfied_by(1)

    merged = CountConstraint.from_data({"min": 1}).merge(CountConstraint.from_data({"max": 2}))
    assert merged.to_data() == {"min": 1, "max": 2}
    with pytest.raises(ResourceValidationError):
        CountConstraint.from_data({"min": 4}).merge(CountConstraint.from_data({"max": 2}))


def test_resource_requirement_canonicalization_and_merge():
    req = ResourceRequirement.from_data(
        {"memory": {"min": "1GiB"}, "accelerators": {"gpu": {"exact": 1}}, "cpus": {"min": 2}}
    )
    assert req.to_data()["memory"] == {"min": "1GiB"}
    assert req.to_data()["accelerators"]["gpu"] == {"exact": 1}

    merged = req.merge(ResourceRequirement.from_data({"cpus": {"min": 4}}))
    assert merged.to_data()["cpus"] == {"min": 4}


def test_accelerator_memory_is_canonical_and_requires_one_limit_per_accelerator():
    spec = ResourceSpec.from_data(
        {"accelerators": {"gpu": 2}, "accelerator_memory": {"gpu": [1024**3, "512MiB"]}}
    )

    assert spec.to_data()["accelerator_memory"] == {"gpu": ["1GiB", "512MiB"]}
    with pytest.raises(ResourceValidationError):
        ResourceSpec.from_data({"accelerators": {"gpu": 2}, "accelerator_memory": {"gpu": ["1GiB"]}})
