import os

import dryml.annotations as ann
import dryml.runtime as runtime


def test_runtime_default_merge_and_override(monkeypatch):
    before = dict(os.environ)

    @runtime.default(torch={"num_threads": 4}, env={"OMP_NUM_THREADS": "4"})
    @ann.default(namespace="runtime", fragment={"frameworks": {"torch": {"deterministic": True}}})
    def train():
        pass

    spec = ann.resolve_runtime_default(train, overrides={"runtime": {"frameworks": {"torch": {"num_threads": 8}}}})
    assert isinstance(spec, runtime.RuntimeContextSpec)
    assert spec.frameworks["torch"] == {"deterministic": True, "num_threads": 8}
    assert dict(os.environ) == before
    assert runtime.active_runtime().mode is runtime.RuntimeMode.ORCHESTRATOR


def test_runtime_default_convenience_accepts_direct_override_payload():
    @runtime.default(torch={"num_threads": 4})
    def train():
        pass

    spec = ann.resolve_runtime_default(train, overrides={"frameworks": {"torch": {"num_threads": 2}}})
    assert spec.frameworks["torch"]["num_threads"] == 2
