import dryml.execute as execute


def add(x, y):
    return x + y


def boom():
    raise ValueError("expected failure")


def test_inline_execute_returns_result():
    assert execute.run(add, 2, 3, backend="inline") == 5


def test_process_execute_returns_result():
    assert execute.run(add, 2, 3, backend="process") == 5


def test_process_execute_propagates_error():
    future = execute.submit(boom, backend="process")
    err = future.exception(timeout=10)

    assert isinstance(err, execute.RemoteExecutionError)
    assert "expected failure" in str(err)
