import pytest

from dryml.runtime import RuntimeEnforcement, materialization_action, materialization_scope


def test_nested_materialization_scope_uses_innermost_and_restores_after_exception():
    assert materialization_action() == "strict"
    with materialization_scope("warn") as action:
        assert action is RuntimeEnforcement.WARN
        assert materialization_action() is RuntimeEnforcement.WARN
        with pytest.raises(RuntimeError):
            with materialization_scope("off"):
                assert materialization_action() == "off"
                raise RuntimeError("stop")
        assert materialization_action() == "warn"
    assert materialization_action() == "strict"
