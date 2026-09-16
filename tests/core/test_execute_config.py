"""Focused configuration proofs for the deferred core execution adapter."""

from __future__ import annotations

import pytest

import dryml.core.execute as execute_module
from dryml.core import Repo
from dryml.core.execute import CoreOptions, PreparedCoreCall, resolve_core_options
from dryml.core.session import config, get_config
from dryml.core.store.dir import DirStore
from dryml.runtime import RuntimeMode


def test_core_options_resolve_call_executor_and_session_layers(tmp_path):
    """Per-call values override executor values, which override session defaults."""

    session_repo = Repo(DirStore(tmp_path / "session", query_index="none"))
    executor_repo = Repo(DirStore(tmp_path / "executor", query_index="none"))
    call_repo = Repo(DirStore(tmp_path / "call", query_index="none"))
    control = DirStore(tmp_path / "control", query_index="none")

    with config(repo=session_repo, cache="strong"):
        inherited = resolve_core_options(
            None,
            executor=CoreOptions(repo=executor_repo, control_store=control, cache="none"),
        )
        assert inherited.repo is executor_repo
        assert inherited.control_store is control
        assert inherited.cache == "none"

        cleared = resolve_core_options(
            CoreOptions(control_store=None),
            executor=CoreOptions(repo=executor_repo, control_store=control),
        )
        assert cleared.repo is executor_repo
        assert cleared.control_store is None

        selected = resolve_core_options(
            CoreOptions(repo=call_repo, cache="weak"),
            executor=CoreOptions(repo=executor_repo, cache="none"),
        )
        assert selected.repo is call_repo
        assert selected.cache == "weak"


def test_core_options_keep_inert_construction_and_reject_unsupported_strategy(tmp_path):
    """Options neither export Stores nor accept a future strategy class prematurely."""

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    exports = []
    original_export = repo.to_definition

    def observe_export():
        exports.append(True)
        return original_export()

    repo.to_definition = observe_export

    options = CoreOptions(repo=repo)
    assert options.repo is repo
    assert exports == []

    class UnsupportedStrategy:
        pass

    with pytest.raises(ValueError, match="SharedDirStoreStrategy"):
        CoreOptions(marshalling=UnsupportedStrategy)


def test_orchestration_floor_rejects_live_results_and_updates_before_export(tmp_path, monkeypatch):
    """Prohibited local materialization fails without changing caller configuration."""

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    before = get_config()
    monkeypatch.setattr(
        execute_module,
        "active_runtime",
        lambda: type("Runtime", (), {"mode": RuntimeMode.ORCHESTRATOR})(),
    )
    monkeypatch.setattr(
        repo,
        "to_definition",
        lambda: (_ for _ in ()).throw(AssertionError("must not export")),
    )

    with config(repo=repo):
        configured = get_config()
        with pytest.raises(ValueError, match="orchestration"):
            resolve_core_options(CoreOptions(return_objects=True))
        assert get_config() is configured
        with pytest.raises(ValueError, match="orchestration"):
            resolve_core_options(CoreOptions(update_args=True))
        assert get_config() is configured

    assert get_config() is before


def test_prepared_core_call_owns_only_detached_frozen_data():
    """Prepared call storage remains deeply immutable after caller mutation."""

    storage = {"repo": {"stores": [{"path": "state"}]}}
    prepared = PreparedCoreCall(b"invocation", storage)
    storage["repo"]["stores"][0]["path"] = "mutated"

    assert prepared.invocation == b"invocation"
    assert prepared.storage_setup["repo"]["stores"][0]["path"] == "state"
    with pytest.raises(TypeError):
        prepared.storage_setup["new"] = "value"
