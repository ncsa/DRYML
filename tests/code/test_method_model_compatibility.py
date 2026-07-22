from __future__ import annotations

import importlib
import subprocess
import sys

import dryml.code as code
from dryml.core2.methods import Method, traits


class CoreImportedMethod(Method):
    @traits(backend=None)
    def call_default(self, value):
        return value


def test_old_code_imports_are_warning_free_and_identity_compatible():
    source = """
import dryml.code as code
import dryml.core2.methods as methods
from dryml.code.compiler_info import CompilerInfo
from dryml.code.method import Method, traits
from dryml.code.traits import BatchMode, Traits

assert code.Method is methods.Method
assert code.Traits is methods.Traits
assert code.CompilerInfo is methods.CompilerInfo
assert code.traits is methods.traits
assert Method is methods.Method
assert traits is methods.traits
assert Traits is methods.Traits
assert BatchMode is methods.BatchMode
assert CompilerInfo is methods.CompilerInfo
"""
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", source],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_method_contract_analyzer_recognizes_core_and_compat_methods():
    old_method = importlib.import_module("dryml.code.method")

    class OldImportedMethod(code.Method):
        @old_method.traits(backend=None)
        def call_default(self, value):
            return value

    core_result = code.analyze(
        CoreImportedMethod, algorithms="method_contracts"
    )
    old_result = code.analyze(OldImportedMethod, algorithms="method_contracts")

    for result in (core_result, old_result):
        fact = result.facts_of_kind("method_contract")[0]
        assert fact.data["method_contract_detected"] is True
        assert fact.data["trait_impls"][0]["name"] == "call_default"
        assert set(fact.data) == {
            "method_contract_detected",
            "class_module",
            "class_qualname",
            "trait_impls",
            "has_user_call",
        }


def test_method_contract_analyzer_respects_include_method_contracts_false():
    result = code.analyze(
        CoreImportedMethod,
        algorithms="method_contracts",
        context=code.CodeAnalysisContext(include_method_contracts=False),
    )

    assert result.facts == ()
    assert result.diagnostics == ()
