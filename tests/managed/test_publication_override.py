"""Declared Store controls reach the authored checkpoint publisher (U16)."""

import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.managed import ManagedConfig, ManagedControlError, ManagedDeclarationError, managed_operation
from tests.managed.u16_fixtures import Value


class OverrideValue(Value):
    @managed_operation(store_parameter="store")
    def advance(self, amount, *, store: Store | None = None, managed):
        self.value += amount if store is not None else amount + 1
        managed.final_state(managed.checkpoint())


def test_store_parameter_requires_optional_keyword_only_slot():
    def positional(self, store, *, managed): pass
    def default(self, *, store=object(), managed): pass
    def missing(self, *, managed): pass
    for target, parameter in ((positional, "store"), (default, "store"), (missing, "store"), (missing, "managed")):
        with pytest.raises(ManagedDeclarationError, match="store_parameter"):
            managed_operation(store_parameter=parameter)(target)


@pytest.mark.parametrize("override", [False, True])
def test_declared_store_override_is_borrowed_and_governs_checkpoint(tmp_path, override):
    source = DirStore(tmp_path / "source", query_index="none")
    other = DirStore(tmp_path / "override", query_index="none")
    control = DirStore(tmp_path / "control", query_index="none")
    repo = Repo(source)
    value = OverrideValue(repo=repo)
    result = value.advance(4, store=other if override else None,
        managed=ManagedConfig(state_repo=repo, control_store=control))
    assert result.value is None
    status = value.advance.status(state_repo=repo, control_store=control)
    selected = other if override else source
    assert status.checkpoint is not None
    with selected.read_state(status.checkpoint.state.key) as state:
        assert state is not None
    if override:
        from dryml.core.store.store import StoreAuthorityError
        with pytest.raises(StoreAuthorityError, match="unavailable"):
            with source.read_state(status.checkpoint.state.key):
                pytest.fail("override checkpoint leaked into default Store")
    assert value.value == (4 if override else 5) and repo.default_store is source
    selected._assert_state_owner_open()
    control._assert_state_owner_open()
