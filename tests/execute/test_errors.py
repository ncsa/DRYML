from __future__ import annotations

from dryml.execute.errors import CleanupError, ExecutionError, RemoteExecutionError


def test_execution_errors_retain_typed_reports_without_sensitive_expansion():
    """Errors keep structured recovery data while ordinary text stays bounded."""
    report = object()
    error = ExecutionError("admission rejected", report=report)
    cleanup = CleanupError("cleanup incomplete", report=report, execution=object())
    remote = RemoteExecutionError("failed", remote_type="ValueError", remote_traceback="trace")

    assert error.report is report
    assert cleanup.report is report
    assert cleanup.execution is not None
    assert not hasattr(cleanup, "cleanup_owner")
    assert remote.remote_type == "ValueError"
    assert remote.remote_traceback == "trace"
