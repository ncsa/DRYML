from dataclasses import dataclass
from typing import Any

import pytest

from dryml.execute.orchestrator import ExecutionOrchestrator
from dryml.execute.backend import InlineBackend, SubProcessBackend


def test_select_backend_prefers_subprocess_when_required():
    orch = ExecutionOrchestrator(
        backends=(InlineBackend(), SubProcessBackend()),
        default_backend_name="inline",
    )

    # explicitly demand subprocess
    requirements = {"subprocess": True}
    backend = orch.select_backend(requirements)
    assert isinstance(backend, SubProcessBackend)


def test_select_backend_uses_default_when_no_requirement():
    orch = ExecutionOrchestrator(
        backends=(InlineBackend(), SubProcessBackend()),
        default_backend_name="inline",
    )

    backend = orch.select_backend({})
    assert isinstance(backend, InlineBackend)


def test_run_function_uses_selected_backend(monkeypatch):
    orch = ExecutionOrchestrator(
        backends=(InlineBackend(), SubProcessBackend()),
        default_backend_name="inline",
    )

    calls = {}

    # Fake backend that records that it was used
    class RecordingBackend(InlineBackend):
        name = "recording"

        def run(self, fn, *args, **kwargs):
            calls["used"] = True
            return super().run(fn, *args, **kwargs)

    rb = RecordingBackend()
    orch.register_backend(rb)

    # Monkeypatch select_backend to always return our recording backend
    monkeypatch.setattr(orch, "select_backend", lambda requirements, backend_hint=None: rb)

    # Monkeypatch plan_for_call to return empty plan + empty requirements
    @dataclass
    class DummyPlan:
        def aggregate_requirements(self):
            return {}

    monkeypatch.setattr(orch, "plan_for_call", lambda fn, *a, **kw: (DummyPlan(), {}))

    def add(x, y):
        return x + y

    res = orch.run_function(add, 2, 3)
    assert res == 5
    assert calls.get("used", False)
