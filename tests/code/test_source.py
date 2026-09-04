"""Tests for static, file-backed source retrieval."""

from __future__ import annotations

import ast

import pytest

from dryml.code import InvalidTargetError, SourceInfo, SourceTarget, SourceUnavailableError, extract_source, get_source_info


def decorated_source_subject() -> str:
    """Provide a file-backed subject for source extraction."""

    return "subject"


def function_with_lambda_default(callback=lambda: 1) -> object:
    """Provide a named function whose first line also contains a lambda."""

    return callback


def one_line_lambda_body() -> object: return lambda: 1


def test_get_source_info_reads_file_backed_function() -> None:
    """File-backed functions expose dedented source and one-based origin lines."""

    source = get_source_info(decorated_source_subject)

    assert source is not None
    assert source.source.startswith("def decorated_source_subject")
    assert source.start_line is not None and source.start_line > 0
    assert source.filename is not None


def test_named_function_source_ignores_same_line_lambdas() -> None:
    """Inline lambdas do not make their containing function ambiguous."""

    default_source = get_source_info(function_with_lambda_default)
    body_source = get_source_info(one_line_lambda_body)

    assert default_source is not None and default_source.source.startswith("def function_with_lambda_default")
    assert body_source is not None and body_source.source.startswith("def one_line_lambda_body")


def test_extract_source_accepts_static_source_target_without_execution() -> None:
    """Source targets are parsed but never compiled or evaluated."""

    target = SourceTarget("lambda value: forbidden_name(value)", filename="/secret/input.py")
    source = extract_source(target)

    assert source == SourceInfo(target.source, "/secret/input.py", None)
    assert ast.parse(source.source)


def test_source_errors_are_typed_and_redacted() -> None:
    """Missing and malformed source report stable categories without raw paths."""

    with pytest.raises(InvalidTargetError) as unavailable:
        extract_source(len)
    with pytest.raises(SourceUnavailableError) as invalid:
        extract_source(SourceTarget("def broken(:\n", filename="/secret/token.py"))

    assert unavailable.value.code == "target.invalid"
    assert invalid.value.code == "source.invalid"
    assert "/secret" not in str(invalid.value)


def test_get_source_info_does_not_call_arbitrary_loader_hooks() -> None:
    """Unsupported objects return no source without calling a loader protocol."""

    class Loader:
        def get_source(self, name: str) -> str:
            raise AssertionError("loader hook invoked")

    class Subject:
        __loader__ = Loader()

    assert get_source_info(Subject()) is None
