import os

import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.publication import EffectPlan, PublicationService, SessionGeneration, publication


def _service(runtime_state=None, **kwargs):
    service = PublicationService(**kwargs)
    service.initialize(runtime_state or runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.OFF))
    return service


def _candidate(service, *, runtime_state=None):
    before = service.current()
    return before, service.stage(
        before,
        SessionGeneration(before.number + 1, runtime_state or before.runtime),
    )


def test_fresh_process_baseline_is_unchecked_python_without_effects():
    state = runtime.active_runtime()

    assert state.mode is runtime.RuntimeMode.NONE
    assert state.allocation is runtime.NoAllocation
    assert state.enforcement is runtime.RuntimeEnforcement.OFF
    assert state.requirement_axes.to_data() == []


def test_publication_commits_and_restores_owned_environment(monkeypatch):
    original = os.environ.get("DRYML_U2_PUBLICATION_TEST")
    before = publication.current()
    candidate = SessionGeneration(
        number=before.number + 1,
        runtime=runtime.RuntimeState(enforcement=runtime.RuntimeEnforcement.STRICT),
    )

    published = publication.commit(
        publication.stage(before, candidate),
        EffectPlan(environment={"DRYML_U2_PUBLICATION_TEST": "owned"}),
    )
    assert published.number == before.number + 1
    assert os.environ["DRYML_U2_PUBLICATION_TEST"] == "owned"

    restored_generation = SessionGeneration(number=published.number + 1, runtime=before.runtime)
    restored = publication.commit(
        publication.stage(published, restored_generation),
        EffectPlan(environment={"DRYML_U2_PUBLICATION_TEST": original}),
    )
    assert restored.runtime is before.runtime
    assert os.environ.get("DRYML_U2_PUBLICATION_TEST") == original


def test_publication_stale_candidate_and_reader_writer_overlap_fail_closed():
    current = publication.current()
    candidate = SessionGeneration(number=current.number + 1, runtime=current.runtime)
    staged = publication.stage(current, candidate)

    with publication.reader():
        with pytest.raises(RuntimeTransitionError, match="upgrade"):
            publication.commit(staged)

    publication.commit(staged)
    with pytest.raises(RuntimeTransitionError) as exc_info:
        publication.commit(staged)
    assert exc_info.value.context["reason"] == "stale_candidate"


def test_effect_failure_rolls_back_owned_environment(monkeypatch):
    before = publication.current()
    candidate = SessionGeneration(number=before.number + 1, runtime=before.runtime)
    key = "DRYML_U2_ROLLBACK_TEST"
    monkeypatch.delenv(key, raising=False)
    plan = EffectPlan(environment={key: "owned"}, process_limits={1: (1, 1)})

    with pytest.raises(RuntimeTransitionError, match="process limits"):
        publication.commit(publication.stage(before, candidate), plan)

    assert publication.current() is before
    assert key not in os.environ


def test_effect_changing_transition_rejects_active_generation_lease():
    before = publication.current()
    candidate = SessionGeneration(number=before.number + 1, runtime=before.runtime)

    with publication.lease() as leased:
        assert leased is before
        with pytest.raises(RuntimeTransitionError, match="active generation lease"):
            publication.commit(
                publication.stage(before, candidate),
                EffectPlan(environment={"DRYML_U2_LEASE_TEST": "blocked"}),
            )


def test_effect_changing_transition_rejects_a_lease_on_a_declarative_descendant():
    service = _service(environ={})
    _, declarative = _candidate(service)
    published = service.commit(declarative)

    with service.lease() as leased:
        assert leased is published
        _, later = _candidate(service)
        service.commit(later)
        _, effectful = _candidate(service)
        with pytest.raises(RuntimeTransitionError, match="active generation lease"):
            service.commit(effectful, EffectPlan(environment={"DRYML_U2_OLD_LEASE": "blocked"}))


def test_effect_journal_carries_environment_ownership_across_declarative_commit_and_restore():
    environ = {"DRYML_U2_JOURNAL": "inherited"}
    service = _service(environ=environ)
    before, candidate = _candidate(service)
    published = service.commit(candidate, EffectPlan(environment={"DRYML_U2_JOURNAL": "owned"}))

    _, declarative = _candidate(service, runtime_state=published.runtime)
    service.commit(declarative)
    journal = service.effect_journal()

    assert len(journal) == 1
    assert journal[0].previous == "inherited"
    assert journal[0].written == "owned"

    _, restore = _candidate(service)
    service.commit(restore, EffectPlan(environment={"DRYML_U2_JOURNAL": "inherited"}))
    assert environ["DRYML_U2_JOURNAL"] == "inherited"
    assert service.effect_journal() == ()


def test_ownership_drift_during_rollback_never_overwrites_external_environment(monkeypatch):
    environ = {}
    service = _service(environ=environ)
    _, candidate = _candidate(service)

    original_publish = service._publish

    def drift_then_fail(generation):
        environ["DRYML_U2_DRIFT"] = "external"
        raise RuntimeError("publication failed")

    monkeypatch.setattr(service, "_publish", drift_then_fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_DRIFT": "owned"}))

    assert environ["DRYML_U2_DRIFT"] == "external"
    assert service.current().health == "failed"
    monkeypatch.setattr(service, "_publish", original_publish)


def test_failure_before_an_irreversible_outcome_does_not_poison_generation(monkeypatch):
    class FailingEnvironment(dict):
        def __setitem__(self, key, value):
            raise KeyboardInterrupt("interrupted before effect")

    service = _service(environ=FailingEnvironment())
    before, candidate = _candidate(service)

    with pytest.raises(KeyboardInterrupt, match="interrupted before effect"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_INTERRUPT": "owned"}, irreversible_outcome="native-init"))

    assert service.current() is before


def test_windows_environment_aliases_are_one_owned_logical_key_without_real_windows(monkeypatch):
    environ = {"Path": "inherited"}
    service = _service(environ=environ, windows=True)
    _, candidate = _candidate(service)
    service.commit(candidate, EffectPlan(environment={"PATH": "owned"}))

    assert environ == {"Path": "owned"}
    with pytest.raises(RuntimeTransitionError, match="duplicate logical keys"):
        _, duplicate = _candidate(service)
        service.commit(duplicate, EffectPlan(environment={"PATH": "one", "path": "two"}))


def test_dedicated_nonrestoring_limit_is_explicit_and_reusable_limit_rejects_before_effects():
    class Limits:
        def __init__(self):
            self.value = (10, 20)
            self.calls = []

        def get(self, kind):
            return self.value

        def set(self, kind, value):
            self.calls.append((kind, value))
            self.value = value

    limits = Limits()
    environ = {}
    service = _service(environ=environ, limit_getter=limits.get, limit_setter=limits.set)
    _, candidate = _candidate(service)
    with pytest.raises(RuntimeTransitionError, match="dedicated non-restoring"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_LIMIT": "must-not-write"}, process_limits={7: (5, 15)}))
    assert environ == {}
    assert limits.calls == []

    _, dedicated = _candidate(service)
    service.commit(dedicated, EffectPlan(process_limits={7: (5, 15)}, dedicated_process=True))
    assert limits.value == (5, 15)
    assert any(record.kind == "irreversible" for record in service.effect_journal())


def test_process_limit_setter_is_journaled_before_a_failed_readback():
    class Limits:
        def __init__(self):
            self.value = (10, 10)
            self.readback_failure = False

        def get(self, kind):
            if self.readback_failure:
                self.readback_failure = False
                return (9, 9)
            return self.value

        def set(self, kind, value):
            self.value = value
            self.readback_failure = value == (5, 10)

    limits = Limits()
    service = _service(environ={}, limit_getter=limits.get, limit_setter=limits.set)
    _, candidate = _candidate(service)

    with pytest.raises(RuntimeTransitionError, match="readback"):
        service.commit(candidate, EffectPlan(process_limits={7: (5, 10)}))

    assert limits.value == (10, 10)
    assert service.current().health == "healthy"


def test_environment_record_precedes_a_mutate_then_raise_setter():
    class InterruptingEnvironment(dict):
        interrupted = False

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("after environment mutation")

    environ = InterruptingEnvironment(DRYML_U2_ENV_RECORD="inherited")
    service = _service(environ=environ)
    before, candidate = _candidate(service)

    with pytest.raises(KeyboardInterrupt, match="after environment mutation"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_ENV_RECORD": "managed"}))

    assert environ["DRYML_U2_ENV_RECORD"] == "inherited"
    assert service.current() is before


def test_affinity_record_precedes_a_mutate_then_raise_setter():
    affinity = {0, 1}
    interrupted = False

    def set_affinity(cpus):
        nonlocal interrupted
        affinity.clear()
        affinity.update(cpus)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after affinity mutation")

    service = _service(affinity_getter=lambda: affinity, affinity_setter=set_affinity)
    before, candidate = _candidate(service)

    with pytest.raises(KeyboardInterrupt, match="after affinity mutation"):
        service.commit(candidate, EffectPlan(cpu_affinity=(0,)))

    assert affinity == {0, 1}
    assert service.current() is before


def test_affinity_is_normalized_and_restored_exactly_after_publication_failure(monkeypatch):
    affinity = {3, 1}

    def get_affinity():
        return affinity

    def set_affinity(cpus):
        affinity.clear()
        affinity.update(cpus)

    service = _service(affinity_getter=get_affinity, affinity_setter=set_affinity)
    before, candidate = _candidate(service)
    monkeypatch.setattr(service, "_publish", lambda generation: (_ for _ in ()).throw(RuntimeError("publish failed")))

    with pytest.raises(RuntimeError, match="publish failed"):
        service.commit(candidate, EffectPlan(cpu_affinity=(3, 1, 3)))

    assert affinity == {1, 3}
    assert service.current() is before


def test_interruption_after_generation_assignment_restores_prior_generation(monkeypatch):
    environ = {}
    service = _service(environ=environ)
    before, candidate = _candidate(service)

    def publish_then_interrupt(generation):
        service._generation = generation
        raise KeyboardInterrupt("interrupted after publication")

    monkeypatch.setattr(service, "_publish", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupted after publication"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_PUBLISH_INTERRUPT": "owned"}))

    assert service.current() is before
    assert environ == {}


def test_interceptor_rollback_uses_identity_and_retained_position(monkeypatch):
    neighbor = object()
    meta_path = [neighbor]
    interceptor = object()
    service = _service(meta_path=meta_path)
    before, candidate = _candidate(service)
    monkeypatch.setattr(service, "_publish", lambda generation: (_ for _ in ()).throw(RuntimeError("publish failed")))

    with pytest.raises(RuntimeError, match="publish failed"):
        service.commit(candidate, EffectPlan(interceptor=interceptor, interceptor_position=1))

    assert meta_path == [neighbor]
    assert interceptor not in meta_path
    assert service.current() is before


def test_failed_environment_readback_fails_closed_without_claiming_rollback():
    class UnreadableEnvironment(dict):
        def get(self, key, default=None):
            if key == "DRYML_U2_READBACK":
                return "not-owned"
            return super().get(key, default)

    environ = UnreadableEnvironment()
    service = _service(environ=environ)
    _, candidate = _candidate(service)

    with pytest.raises(RuntimeTransitionError, match="failed closed"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_READBACK": "owned"}))

    failed = service.current()
    assert failed.health == "failed"
    assert failed.runtime.mode is runtime.RuntimeMode.ORCHESTRATOR
    assert failed.runtime.allocation is runtime.NoAllocation
    assert failed.runtime.enforcement is runtime.RuntimeEnforcement.STRICT
    assert failed.runtime.requirement_axes.to_data() == ["environment", "world", "runtime"]


def test_effect_write_reentry_fails_before_state_lock_and_outer_transaction_rolls_back():
    service = _service()
    before, candidate = _candidate(service)
    reentry_errors = []

    class AuditedEnvironment(dict):
        def __setitem__(self, key, value):
            try:
                service.snapshot()
            except RuntimeTransitionError as exc:
                reentry_errors.append(exc)
            super().__setitem__(key, value)

    service._environ = AuditedEnvironment()
    original_publish = service._publish
    service._publish = lambda generation: (_ for _ in ()).throw(RuntimeError("publish failed"))
    with pytest.raises(RuntimeError, match="publish failed"):
        service.commit(candidate, EffectPlan(environment={"DRYML_U2_AUDIT": "owned"}))

    assert reentry_errors
    assert service._environ == {}
    assert service.current() is before
    service._publish = original_publish
