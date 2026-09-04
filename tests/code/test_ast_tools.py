"""Tests for tolerant AST parsing and deterministic access collection."""

from __future__ import annotations

import ast
import sys

import pytest

from dryml.code import SourceInfo, SourceUnavailableError
from dryml.code.ast_tools import collect_accesses_from_source, parse_source


def test_collect_accesses_preserves_unicode_byte_columns_and_absolute_lines() -> None:
    """AST offsets remain UTF-8 byte offsets while lines honor source origins."""

    source = SourceInfo("é = 1\nobj.method(é)\nobj.value = é\n", "input.py", 40)
    accesses = collect_accesses_from_source(source)

    assert accesses.method_calls[0].root == "obj"
    assert accesses.method_calls[0].chain == ("method",)
    assert accesses.method_calls[0].lineno == 41
    assert accesses.method_calls[0].col_offset == 0
    assert accesses.attr_accesses[-1].ctx == "store"
    assert accesses.attr_accesses[-1].lineno == 42


def test_parse_source_accepts_maintained_syntax_without_version_specific_nodes() -> None:
    """Parsing uses the running interpreter rather than newer AST node imports."""

    tree = parse_source("match subject:\n    case {'key': value}:\n        result = value\n")

    assert isinstance(tree, ast.Module)


def test_parse_source_accepts_version_gated_grammar() -> None:
    """New grammar is tested only where the running interpreter supports it."""

    if sys.version_info >= (3, 11):
        assert isinstance(parse_source("try:\n    pass\nexcept* ValueError:\n    pass\n"), ast.Module)
    if sys.version_info >= (3, 12):
        assert isinstance(parse_source("def identity[T](value: T) -> T:\n    return value\n"), ast.Module)
    if sys.version_info >= (3, 14):
        assert isinstance(parse_source("value = t'{name}'\n"), ast.Module)


def test_parse_source_reports_malformed_input_with_a_stable_error() -> None:
    """Syntax failures never expose parser text at the public boundary."""

    with pytest.raises(SourceUnavailableError) as error:
        parse_source("def invalid(:\n")

    assert error.value.code == "source.invalid"
    assert "invalid syntax" not in str(error.value).lower()
