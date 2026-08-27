import pytest

from dryml.worlds import CountConstraint, ResourceRequirement, ResourceValidationError


def test_constraints_intersect_and_accelerator_memory_is_per_device():
    assert CountConstraint.from_data({"min": 1, "max": None}).merge(CountConstraint.from_data({"min": None, "max": 2})).to_data() == {"min": 1, "max": 2}
    resources = ResourceRequirement.from_data({"accelerators": {"gpu": {"min": 2, "max": 2}}, "accelerator_memory": {"gpu": {"min": "1GiB", "max": None}}})
    assert resources.to_data()["accelerator_memory"]["gpu"] == {"min": "1GiB", "max": None}


@pytest.mark.parametrize("data", ({"min": -1, "max": None}, {"min": 2, "max": 1}, {"exact": 1}, {"min": 1}))
def test_invalid_count_constraints_fail(data):
    with pytest.raises(ResourceValidationError):
        CountConstraint.from_data(data)


def test_count_constraint_payload_is_explicit_and_closed():
    assert CountConstraint().to_data() == {"min": None, "max": None}
    assert CountConstraint(2, 2).to_data() == {"min": 2, "max": 2}
