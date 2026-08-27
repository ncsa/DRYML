import pytest

from dryml.worlds import current, reset_current, set_current, use


def test_current_scope_restores_after_exception_without_allocation():
    reset_current()
    assert set_current("outer") is None
    with pytest.raises(RuntimeError):
        with use("inner"):
            assert current() == "inner"
            raise RuntimeError()
    assert current() == "outer"
