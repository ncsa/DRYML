"""Concurrency boundaries for watched framework loader lifecycles."""

from __future__ import annotations

import importlib.util
import threading
import uuid

import pytest

from dryml.runtime import EffectPlan, PublicationBusyError, RuntimeMode, RuntimeState, publication
from dryml.runtime.frameworks import FrameworkRegistration, framework_registry
from dryml.runtime.publication import PublicationService
from dryml.worlds import LocalResourceInventory


@pytest.fixture(autouse=True)
def _isolate_registry_freeze():
    """Keep synthetic registrations isolated from another test's frozen epoch."""
    with framework_registry._lock:
        registrations = dict(framework_registry._registrations)
        framework_registry._frozen = False
    yield
    with framework_registry._lock:
        framework_registry._registrations = registrations
        framework_registry._frozen = False


def test_create_exec_gap_holds_a_publication_lease(tmp_path, monkeypatch):
    """Incompatible publication cannot cross a watched module's creation gap."""
    name = "dryml_fake_" + uuid.uuid4().hex
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
    framework_registry.register(FrameworkRegistration(name, (name,), object()))
    monkeypatch.syspath_prepend(str(tmp_path))
    publication.publish(RuntimeState(RuntimeMode.ORCHESTRATOR))
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(PublicationBusyError):
        publication.publish(RuntimeState(RuntimeMode.NONE))
    spec.loader.exec_module(module)


def test_reader_to_writer_upgrade_is_bounded():
    """A publication writer cannot upgrade its own framework-reader admission."""
    from dryml._framework_imports import ImportEpochCoordinator, ImportEpochBusyError

    coordinator = ImportEpochCoordinator()
    with coordinator.reader():
        with pytest.raises(ImportEpochBusyError, match="upgrade"):
            with coordinator.writer():
                pass


def test_active_publication_closes_registration_before_process_effects():
    """A registration cannot enter while an active generation is staged."""
    effect_started = threading.Event()
    release_effect = threading.Event()

    class BlockingEnvironment(dict):
        def __setitem__(self, key, value):
            effect_started.set()
            assert release_effect.wait(timeout=2)
            super().__setitem__(key, value)

    service = PublicationService(environ=BlockingEnvironment())
    service.initialize(RuntimeState())
    inventory = LocalResourceInventory((0,), {})
    failure: list[BaseException] = []

    def publish():
        try:
            service.publish(
                RuntimeState(RuntimeMode.ORCHESTRATOR),
                inventory=inventory,
                effects=EffectPlan(environment={"DRYML_TEST_EFFECT": "1"}),
            )
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=publish)
    thread.start()
    assert effect_started.wait(timeout=2)
    try:
        with pytest.raises(RuntimeError, match="import-busy|frozen"):
            root = "dryml_fake_" + uuid.uuid4().hex
            framework_registry.register(FrameworkRegistration(root, (root,), object()))
    finally:
        release_effect.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert not failure
