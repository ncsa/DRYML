import pytest

import dryml.dispatch as dispatch
import dryml.runtime as runtime


pytestmark = pytest.mark.current_limitation


def test_runtime_enforcement_strict_exists():
    assert runtime.RuntimeEnforcement.STRICT.value == "strict"


def test_runtime_enforcement_warn_exists():
    assert runtime.RuntimeEnforcement.WARN.value == "warn"


def test_runtime_enforcement_off_context_exists():
    with runtime.disabled():
        assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.OFF


def test_runtime_plain_context_uses_local_python_like_execution():
    with runtime.plain():
        assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.OFF


@pytest.mark.xfail(
    reason=(
        "currently unsupported: runtime_enforcement OFF does not bypass "
        "dispatch planning and launch requirements"
    ),
    strict=True,
)
def test_dispatch_respects_runtime_enforcement_off():
    with runtime.disabled():
        assert dispatch.submit(lambda: 1).result() == 1
