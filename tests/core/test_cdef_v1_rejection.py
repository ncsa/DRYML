"""Prove retired CDef authority fails before it can become a usable CDef."""

import pytest

from dryml.core import ConcreteDefinition


@pytest.mark.parametrize(
    "state",
    [
        (object, (), {}),
        {"cls": object, "args": (), "kwargs": {}},
        {"identity_version": 1, "cls": object, "args": (), "kwargs": {}, "stable_hash_cache": None},
    ],
)
def test_pre_v2_identity_authority_never_hydrates_a_cdef(state):
    """Raw and V1 identity records reject before CDef restoration."""

    target = object.__new__(ConcreteDefinition)

    with pytest.raises(ValueError, match="V2|version|pre-V2"):
        target.__setstate__(state)


def test_missing_identity_version_never_defaults_to_a_legacy_reader():
    """A mapping without the explicit V2 version is rejected, not inferred."""

    target = object.__new__(ConcreteDefinition)

    with pytest.raises(ValueError, match="V2"):
        target.__setstate__(
            {
                "cls": object,
                "parameters": {},
                "stateful_role": False,
                "stable_hash_cache": None,
            }
        )
