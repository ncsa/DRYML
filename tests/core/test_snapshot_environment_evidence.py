from datetime import datetime, timezone

import pytest

import dryml.environments as envs
from dryml.environments import combination
from dryml.core import Object, Repo, Serializable
from dryml.core.repo_plan import build_save_plan
from dryml.core.snapshot_capture import capture_snapshot
from dryml.core.store.dir import DirStore
from dryml.environments.combination import _requirements_for_classes


@envs.req(requirements=("root>=1",), source="root")
class EvidenceRoot(Object):
    def __init__(self, child, ref_only=None):
        self.child = child
        self.ref_only = ref_only


@envs.req(requirements=("child<2",), source="child")
class EvidenceLeaf(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


@envs.req(requirements=("conflict<1",), source="first")
class ConflictingFirst:
    pass


@envs.req(requirements=("conflict>=1",), source="second")
class ConflictingSecond:
    pass


class EmptyEvidence:
    pass


def _record():
    return envs.EnvironmentRecord(
        python=envs.PythonRecord("3.12.0", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
        distributions={},
        dryml=envs.DrymlRuntimeRecord(),
    )


def test_capture_uses_materializing_live_classes_once_and_observes_once(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = EvidenceLeaf("child", repo=repo)
    root = EvidenceRoot([child, child], ref_only="ignored", repo=repo)
    observed = []

    capture = capture_snapshot(
        build_save_plan(repo, root),
        observer=lambda: observed.append(True) or _record(),
        clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert observed == [True]
    assert capture.environment_status == "known"
    assert capture.requirements_status == "value"
    assert capture.requirements_coverage == "complete"
    assert capture.requirements.requirements == ("child<2", "root>=1")


def test_requirement_capture_keeps_partial_conflicts_and_unavailable_distinct():
    conflict = _requirements_for_classes((ConflictingFirst, ConflictingSecond))
    partial = _requirements_for_classes((EvidenceRoot, object()))
    unavailable = _requirements_for_classes((object(),))
    empty = _requirements_for_classes((EmptyEvidence,))

    assert (conflict.status, conflict.coverage, conflict.value) == ("conflict", "complete", None)
    assert conflict.diagnostics
    assert (partial.status, partial.coverage) == ("value", "incomplete")
    assert partial.value.requirements == ("root>=1",)
    assert (unavailable.status, unavailable.coverage, unavailable.value) == (
        "unavailable", "incomplete", None,
    )
    assert (empty.status, empty.coverage, empty.value) == ("empty", "complete", None)


def test_requirement_capture_rejects_aggregate_limits_and_propagates_cancellation(monkeypatch):
    monkeypatch.setattr(combination, "_MAX_DECLARATIONS", 1)
    limited = _requirements_for_classes((EvidenceRoot, EvidenceLeaf))

    assert (limited.status, limited.coverage, limited.value) == (
        "unavailable", "incomplete", None,
    )
    assert limited.diagnostics == ((
        "dryml.environments.requirement_collection_unavailable",
        "environment requirement collection unavailable",
    ),)

    monkeypatch.setattr(
        combination,
        "collect_declarations",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _requirements_for_classes((EvidenceRoot,))


def test_capture_reports_observer_failure_but_propagates_cancellation(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    root = EvidenceRoot(EvidenceLeaf("child", repo=repo), repo=repo)
    plan = build_save_plan(repo, root)

    unavailable = capture_snapshot(plan, observer=lambda: (_ for _ in ()).throw(OSError("offline")))
    assert unavailable.environment_status == "unavailable"
    assert unavailable.environment is None
    assert unavailable.diagnostics == (("dryml.environments.observation_unavailable", "current environment observation unavailable"),)

    with pytest.raises(KeyboardInterrupt):
        capture_snapshot(plan, observer=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
