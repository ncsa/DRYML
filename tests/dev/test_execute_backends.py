import pytest

from dryml.execute.backend import InlineBackend, SubProcessBackend, ExecutionError


def test_inline_backend_runs_function_directly():
    be = InlineBackend()

    def add(x, y):
        return x + y

    res = be.run(add, 2, 3)
    assert res == 5


def test_subprocess_backend_runs_function_in_subprocess():
    be = SubProcessBackend()

    def mul(x, y):
        return x * y

    res = be.run(mul, 4, 5)
    assert res == 20


def test_subprocess_backend_propagates_exceptions():
    be = SubProcessBackend()

    def boom():
        raise ValueError("boom")

    with pytest.raises(ExecutionError) as excinfo:
        be.run(boom)
    assert "boom" in str(excinfo.value)
    assert isinstance(excinfo.value.original, ValueError)
