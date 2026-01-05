# tests/test_graph_ast_analysis.py

from dataclasses import dataclass
from typing import Any

import pytest

from dryml.graph.ast_analysis import collect_call_specs


# Minimal compute_spec stub for testing
@dataclass
class DummySpec:
    name: str
    compute_reqs: dict[str, Any] | None = None


def compute_spec(*, name: str, compute_reqs=None):
    def dec(f):
        f.__dry_compute_spec__ = DummySpec(name=name, compute_reqs=compute_reqs or {})
        return f
    return dec


class Model:
    @compute_spec(name="fit_spec", compute_reqs={"tf": {"gpu": 1}})
    def fit(self, data):
        pass

    @compute_spec(name="eval_spec", compute_reqs={"cpu": {"threads": 4}})
    def evaluate(self, data):
        pass


@compute_spec(name="log_spec", compute_reqs={"logging": {}})
def log_metrics(mdl, data):
    pass


def test_ast_collects_method_and_function_calls():
    def train_fn(mdl: Model, data, do_eval=True):
        mdl.fit(data)
        if do_eval:
            mdl.evaluate(data)
        log_metrics(mdl, data)

    specs = collect_call_specs(train_fn)

    # Expect 3 call sites with specs: fit, evaluate, log_metrics
    names = {s.spec.name for s in specs}
    assert names == {"fit_spec", "eval_spec", "log_spec"}

    # Check that receiver_name is set for method calls
    method_specs = [s for s in specs if s.receiver_name is not None]
    assert {s.receiver_name for s in method_specs} == {"mdl"}

    # Global function call has receiver_name=None
    func_specs = [s for s in specs if s.receiver_name is None]
    assert any(s.qualname.endswith("log_metrics") for s in func_specs)


def test_ast_respects_simple_aliases():
    def train_fn(mdl: Model, data):
        x = mdl
        x.fit(data)

    specs = collect_call_specs(train_fn)
    names = {s.spec.name for s in specs}
    assert names == {"fit_spec"}
    # Ensure it recognized x as alias of mdl
    assert specs[0].receiver_name in {"mdl", "x"}
