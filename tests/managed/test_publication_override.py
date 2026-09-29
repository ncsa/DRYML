"""Declared managed Store publication-control tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo, StateRef
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.managed import ManagedConfig, ManagedDeclarationError, managed_operation


class OverrideValue(Pickleable):
    """Resumable receiver that records the declared borrowed Store handle."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(
        resumable=True, return_state_ref=True, store_parameter="store",
    )
    def advance(self, amount, *, store: Store | None = None, managed) -> None:
        """Save via the supplied Store while retaining ordinary managed controls."""

        self.value += amount if store is not None else amount + 1


def test_store_parameter_requires_an_optional_keyword_only_declared_slot():
    """Invalid declared Store controls fail during declaration rather than invocation."""

    def positional(self, store, *, managed) -> None:
        return None

    def non_none_default(self, *, store=object(), managed) -> None:
        return None

    def missing(self, *, managed) -> None:
        return None

    def reserved(self, *, managed) -> None:
        return None

    for target, parameter in (
            (positional, "store"), (non_none_default, "store"),
            (missing, "store"), (reserved, "managed"),
    ):
        with pytest.raises(ManagedDeclarationError, match="store_parameter"):
            managed_operation(store_parameter=parameter)(target)


def test_declared_store_override_is_borrowed_and_returns_its_final_state(tmp_path):
    """A non-None declared Store governs final publication without mutating routing."""

    source = DirStore(tmp_path / "source")
    override = DirStore(tmp_path / "override")
    control = DirStore(tmp_path / "control")
    repo = Repo((source,))
    value = OverrideValue(repo=repo)

    result = value.advance(
        4, store=override,
        managed=ManagedConfig(state_repo=repo, control_store=control),
    )

    assert type(result) is StateRef
    assert value.value == 4
    assert override.read_state_ref_record(result.digest()).state_ref == result
    assert source.read_state_ref_record(result.digest()) is None
    assert value.advance.status(
        state_repo=override, control_store=control,
    ).final_state_ref == result


def test_none_store_parameter_preserves_current_repo_routing(tmp_path):
    """The declared None control leaves ordinary current-Repo publication intact."""

    source = DirStore(tmp_path / "source")
    repo = Repo((source,))
    value = OverrideValue(repo=repo)

    result = value.advance(4, store=None, managed=ManagedConfig(state_repo=repo))

    assert source.read_state_ref_record(result.digest()).state_ref == result
    assert value.value == 5
