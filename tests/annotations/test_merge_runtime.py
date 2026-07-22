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


def test_runtime_default_fragment_is_mode_neutral_until_resolution():
    @runtime.default(torch={"num_threads": 4})
    def train():
        pass

    fragment = ann.fragments_for(train, namespace="runtime", kind="default")[0]
    assert "mode" not in fragment.fragment
    assert ann.resolve_runtime_default(train).mode is runtime.RuntimeMode.ORCHESTRATOR


def test_runtime_override_can_clear_framework_defaults():
    @runtime.default(torch={"num_threads": 4})
    def train():
        pass

    spec = ann.resolve_runtime_default(train, overrides={"frameworks": {}})
    assert spec.frameworks == {}


def test_runtime_merge_policy_replace():
    @ann.default(namespace="runtime", fragment={"frameworks": {"torch": {"num_threads": 2}}}, merge_policy="replace", priority=1)
    @ann.default(namespace="runtime", fragment={"frameworks": {"plain": {}}}, priority=0)
    def train():
        pass

    spec = ann.resolve_runtime_default(train)
    assert set(spec.frameworks) == {"torch"}


def test_runtime_requirement_fragments_are_collected_but_not_enforced_yet():
    @ann.require(namespace="runtime", fragment={"frameworks": {"torch": {"num_threads": 8}}})
    @runtime.default(torch={"num_threads": 2})
    def train():
        pass

    result = ann.resolve(train)
    assert result.report.ok
    assert result.requirements.runtime == {"frameworks": {"torch": {"num_threads": 8}}}
