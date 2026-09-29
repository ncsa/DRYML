"""Whole-call transport coverage for managed configuration values."""

from __future__ import annotations

import shutil

import dill
import pytest

from dryml.core import Repo
from dryml.core.symbol import ImportRef
from dryml.core.execute_codec import (
    CoreCallCodecError,
    _outcome,
    _result_graph,
    decode_outcome,
    encode_invocation,
    invoke_invocation,
)
from dryml.core.session import config as core_config
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store, StoreAuthorityError
from dryml.core.store.zip import ZipStore
from dryml.core.object import Pickleable
from dryml.managed import ManagedConfig, managed_operation
from dryml import session


def _config_facts(config):
    """Project portable config policy without returning its live resources."""

    return config.rerun, len(config.callbacks or [])


def _same_config_facts(first, second):
    """Expose config and callback graph aliases without returning resources."""

    return first is second, first.callbacks[0] is second.callbacks[0]


def _nested_config_facts(value):
    """Read a config from a nested ordinary argument container."""

    return _config_facts(value["config"][0])


class _ConfigFieldCallable:
    """Callable instance fixture with a transported config instance field."""

    def __init__(self, config):
        self.config = config

    def __call__(self):
        return _config_facts(self.config)


class _CallbackMarker(Exception):
    """Test-only callback failure proving worker-side callback execution."""


class _ManagedValue(Pickleable):
    """Minimal checkpointing receiver for transported callback coverage."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def advance(self, *, managed):
        """Publish one checkpoint before returning the updated value."""

        self.value += 1
        managed.checkpoint()
        return self.value


def _raise_callback_marker(object_, context):
    """Stop after checkpoint association with a recognizable worker failure."""

    raise _CallbackMarker()


def _nested_managed_call(value, options):
    """Invoke managed normally from an ordinary transported function."""

    return value.advance(managed=options["config"])


def _resource_facts(config):
    """Compare decoded config resources with the active worker cache selection."""

    from dryml.core.session import current_repo

    selected = current_repo()
    return (
        config.state_repo is selected,
        config.control_store is selected.stores[0],
    )


def _store_setting_facts(config):
    """Expose a decoded Store policy without returning its live handle."""

    from dryml.core.session import current_repo

    return config.state_repo.query_index_policy, config.state_repo is current_repo().stores[0]


def _worker_constructed_config():
    """Construct a normal local config after transport reaches the worker."""

    from dryml.managed import ManagedConfig

    return _config_facts(ManagedConfig(rerun=True))


def _callback_shape(config):
    """Expose the public ``None`` versus empty-list callback distinction."""

    return config.callbacks is None, config.callbacks == [], type(config.callbacks) is list


def _direct_config_result(config):
    """Return one config directly to exercise result-graph rejection."""

    return config


def _nested_config_result(config):
    """Return one config nested in ordinary result containers."""

    return {"config": [config]}


def test_managed_config_policy_and_callbacks_round_trip_detached_from_caller(tmp_path):
    """A transported config retains the capture-time policy and callback members."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    callbacks = [lambda object_, context: None]
    config = ManagedConfig(state_repo=repo, rerun=True, callbacks=callbacks)

    with core_config(repo=repo), session.resource_cache():
        invocation = encode_invocation(_config_facts, (config,), {}, repo=repo)
        graph = dill.loads(invocation)
        object.__setattr__(config, "rerun", False)
        callbacks.clear()
        outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)

    assert outcome["success"]
    assert outcome["result"] == (True, 1)
    assert graph["version"] == 2
    assert {node["tag"] for node in graph["nodes"]} >= {"managed_config", "function"}


@pytest.mark.parametrize(
    "location", ("positional", "keyword", "nested", "default", "captured", "field"),
)
def test_managed_config_round_trips_at_every_supported_call_graph_location(tmp_path, location):
    """Configs remain ordinary exact values in argument, capture, and field graphs."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    config = ManagedConfig(rerun=True, callbacks=[lambda object_, context: None])
    if location == "positional":
        target, args, kwargs = _config_facts, (config,), {}
    elif location == "keyword":
        target, args, kwargs = _config_facts, (), {"config": config}
    elif location == "nested":
        target, args, kwargs = _nested_config_facts, ({"config": [config]},), {}
    elif location == "default":
        def target(config=config):
            return _config_facts(config)

        args, kwargs = (), {}
    elif location == "captured":
        def target():
            return _config_facts(config)

        args, kwargs = (), {}
    else:
        target, args, kwargs = _ConfigFieldCallable(config), (), {}

    outcome = decode_outcome(
        invoke_invocation(encode_invocation(target, args, kwargs, repo=repo), repo=repo), repo=repo,
    )

    assert outcome["success"]
    assert outcome["result"] == (True, 1)


def test_repeated_config_and_callback_references_preserve_aliases(tmp_path):
    """The invocation memo spans repeated config values and callback references."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    callback = lambda object_, context: None
    config = ManagedConfig(callbacks=[callback])

    outcome = decode_outcome(
        invoke_invocation(
            encode_invocation(_same_config_facts, (config, config), {}, repo=repo), repo=repo,
        ),
        repo=repo,
    )

    assert outcome["success"]
    assert outcome["result"] == (True, True)


@pytest.mark.parametrize(
    ("callbacks", "encoded_callbacks", "expected"),
    (
        (None, None, (True, False, False)),
        ([], [], (False, True, True)),
    ),
)
def test_managed_config_callback_shape_round_trips_exactly(
        tmp_path, callbacks, encoded_callbacks, expected):
    """Invocation transport preserves absent callbacks separately from an empty list."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    invocation = encode_invocation(
        _callback_shape, (ManagedConfig(callbacks=callbacks),), {}, repo=repo,
    )
    graph = dill.loads(invocation)
    callback_nodes = [
        node["callbacks"] for node in graph["nodes"] if node["tag"] == "managed_config"
    ]
    outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)

    assert callback_nodes == [encoded_callbacks]
    assert outcome["success"]
    assert outcome["result"] == expected


@pytest.mark.parametrize("target", (_direct_config_result, _nested_config_result))
def test_managed_config_results_are_rejected_by_the_result_encoder(target):
    """Result encoding denies configs at direct and nested locations."""

    with pytest.raises(CoreCallCodecError, match="managed config result"):
        _result_graph(target(ManagedConfig()), limit_bytes=1_000_000, automatic_references=set())


@pytest.mark.parametrize("nested", (False, True))
def test_managed_config_result_decode_rejects_before_opening_resources(
        tmp_path, nested, monkeypatch):
    """A result config is denied during graph validation before any authority opens."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    config = {
        "tag": "managed_config",
        "state_repo": {"role": "repo", "definition": repo.to_definition().to_data()},
        "control_store": {
            "role": "store", "definition": repo.stores[0].to_definition(),
        },
        "rerun": False,
        "callbacks": [],
    }
    graph = {
        "version": 2,
        "root": 1 if nested else 0,
        "nodes": [config] if not nested else [config, {"tag": "list", "items": [0]}],
    }
    opened = []
    monkeypatch.setattr(
        Repo, "from_definition",
        classmethod(lambda cls, definition: opened.append((cls, definition))),
    )
    monkeypatch.setattr(
        Store, "from_definition",
        classmethod(lambda cls, definition: opened.append((cls, definition))),
    )

    with pytest.raises(CoreCallCodecError, match="managed config result"):
        decode_outcome(_outcome(
            True, result=dill.dumps(graph, protocol=5), limit_bytes=1_000_000,
        ), repo=repo)

    assert opened == []


@pytest.mark.parametrize(
    ("tag", "fields", "reference"),
    (
        (
            "managed_declaration",
            {"authored": 0, "executable": 0, "owner": 0, "member": "operation", "resumable": True},
            "authored",
        ),
        (
            "managed_composite",
            {"declaration": 0, "outer": 0, "owner": 0, "member": "operation"},
            "declaration",
        ),
        (
            "managed_target",
            {"receiver": 0, "declaration": 0},
            "declaration",
        ),
        (
            "managed_composite_target",
            {"receiver": 0, "composite": 0},
            "composite",
        ),
    ),
)
@pytest.mark.parametrize("malformation", ("field", "reference", "type"))
def test_malformed_managed_graph_nodes_fail_before_import_or_resource_open(
        tmp_path, tag, fields, reference, malformation, monkeypatch):
    """Managed node grammar is closed before imports, materialization, or opens occur."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    node = {"tag": tag, **fields}
    if malformation == "field":
        node["unexpected"] = None
    elif malformation == "reference":
        node[reference] = True
    graph = {
        "version": 2,
        "root": 1,
        "nodes": [
            {"tag": "import", "module": "math", "qualname": None},
            node,
        ],
    }
    monkeypatch.setattr(ImportRef, "resolve", lambda self: pytest.fail("import resolved"))
    monkeypatch.setattr(
        Repo, "from_definition",
        classmethod(lambda cls, definition: pytest.fail("resource opened")),
    )
    monkeypatch.setattr(
        Repo, "materialize_boundary",
        lambda self, *args, **kwargs: pytest.fail("value materialized"),
    )

    with pytest.raises(CoreCallCodecError, match="malformed (graph node|graph reference|managed)"):
        invoke_invocation(dill.dumps(graph, protocol=5), repo=repo)


def test_transported_callback_runs_inside_nested_worker_managed_call(
        tmp_path, fixed_managed_snapshot_environment):
    """Checkpoint callbacks run in the in-process worker lifecycle, not capture."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = _ManagedValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    config = ManagedConfig(state_repo=repo, callbacks=[_raise_callback_marker])

    with core_config(repo=repo), session.resource_cache():
        outcome = decode_outcome(
            invoke_invocation(
                encode_invocation(_nested_managed_call, (value, {"config": config}), {}, repo=repo),
                repo=repo,
            ),
            repo=repo,
        )

    assert not outcome["success"]
    assert outcome["reason"] == "_CallbackMarker"


def test_matching_config_resources_reuse_worker_cache_handles(tmp_path):
    """Matching Repo and control roles resolve to the worker's selected handles."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    config = ManagedConfig(state_repo=repo, control_store=repo.stores[0])

    with core_config(repo=repo), session.resource_cache() as cache:
        outcome = decode_outcome(
            invoke_invocation(encode_invocation(_resource_facts, (config,), {}, repo=repo), repo=repo),
            repo=repo,
        )
        assert cache.repos == (repo,)
        assert cache.stores == (repo.stores[0],)

    assert outcome["success"]
    assert outcome["result"] == (True, True)


def test_differing_config_store_settings_do_not_reuse_worker_handle(tmp_path):
    """The cache keeps direct Store requests with different opening policy distinct."""

    path = tmp_path / "state"
    worker_store = DirStore(path, query_index="auto")
    configured_store = DirStore(path, query_index="none")
    repo = Repo(worker_store)
    config = ManagedConfig(state_repo=configured_store)

    with core_config(repo=repo), session.resource_cache() as cache:
        outcome = decode_outcome(
            invoke_invocation(encode_invocation(_store_setting_facts, (config,), {}, repo=repo), repo=repo),
            repo=repo,
        )
        assert len(cache.stores) == 2

    configured_store.close()
    assert outcome["success"]
    assert outcome["result"] == ("none", False)


def test_live_resources_outside_managed_config_remain_rejected(tmp_path):
    """The config role is the only core-resource exception in the call grammar."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))

    with pytest.raises(CoreCallCodecError, match="live core resource"):
        encode_invocation(_nested_config_facts, ({"resource": repo},), {}, repo=repo)


def test_managed_config_rejects_zipstore_transport(tmp_path):
    """The worker config role remains limited to direct DirStore authority."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    archive = ZipStore(tmp_path / "state.zip")

    try:
        with pytest.raises(CoreCallCodecError, match="unsupported managed config resource"):
            encode_invocation(_config_facts, (ManagedConfig(state_repo=archive),), {}, repo=repo)
    finally:
        archive.close()


def test_unsupported_callback_and_mutated_config_policy_fail_before_invocation(tmp_path):
    """Transport revalidates mutable callback membership and exact bool policy."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    config = ManagedConfig(callbacks=[])
    config.callbacks.append(object())
    with pytest.raises(CoreCallCodecError, match="unsupported managed callback"):
        encode_invocation(_config_facts, (config,), {}, repo=repo)

    config = ManagedConfig()
    object.__setattr__(config, "rerun", 1)
    with pytest.raises(CoreCallCodecError, match="malformed managed config"):
        encode_invocation(_config_facts, (config,), {}, repo=repo)


@pytest.mark.parametrize(
    ("graph", "reason"),
    (
        ({"version": 1, "root": 0, "nodes": []}, "malformed call graph"),
        ({"version": 3, "root": 0, "nodes": []}, "malformed call graph"),
        ({"version": 2, "root": 0, "nodes": [], "extra": None}, "malformed call graph"),
        ({"version": 2, "root": 0, "nodes": [{"tag": "unknown"}]}, "malformed graph node"),
        ({
            "version": 2, "root": 0,
            "nodes": [{
                "tag": "managed_config", "state_repo": None, "control_store": None,
                "rerun": True, "callbacks": [], "extra": None,
            }],
        }, "malformed graph node"),
        ({
            "version": 2, "root": 0,
            "nodes": [{
                "tag": "managed_config", "state_repo": None, "control_store": None,
                "rerun": 1, "callbacks": [],
            }],
        }, "malformed managed config"),
        ({
            "version": 2, "root": 0,
            "nodes": [{
                "tag": "managed_config", "state_repo": None, "control_store": None,
                "rerun": True, "callbacks": [0] * 65,
            }],
        }, "malformed managed config"),
    ),
)
def test_malformed_managed_config_graphs_fail_before_effectful_decode(tmp_path, graph, reason):
    """Unknown versions, tags, fields, types, and bounds have closed failures."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))

    with pytest.raises(CoreCallCodecError, match=reason):
        invoke_invocation(dill.dumps(graph, protocol=5), repo=repo)


def test_malformed_config_resource_fails_before_opening_any_store(tmp_path, monkeypatch):
    """Resource role validation completes before a malformed graph can open authority."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    graph = {
        "version": 2, "root": 0,
        "nodes": [{
            "tag": "managed_config",
            "state_repo": {
                "role": "store",
                "definition": {"kind": "zip", "path": str(tmp_path / "missing.zip")},
            },
            "control_store": None, "rerun": True, "callbacks": [],
        }],
    }
    monkeypatch.setattr("dryml.core.execute_codec.Store.from_definition", lambda definition: pytest.fail("opened"))

    with pytest.raises(CoreCallCodecError, match="unsupported managed config resource"):
        invoke_invocation(dill.dumps(graph, protocol=5), repo=repo)


def test_unavailable_config_destination_fails_when_its_node_decodes(tmp_path):
    """An explicit config node opens existing authority only at worker decoding."""

    worker_repo = Repo(DirStore(tmp_path / "worker", query_index="none"))
    destination = DirStore(tmp_path / "destination", query_index="none")
    invocation = encode_invocation(
        _config_facts, (ManagedConfig(state_repo=destination),), {}, repo=worker_repo,
    )
    destination.close()
    shutil.rmtree(destination.base_dir)

    with pytest.raises(StoreAuthorityError, match="missing or inaccessible"):
        invoke_invocation(invocation, repo=worker_repo)


def test_plain_descriptor_in_untaken_branch_does_not_open_resource(tmp_path, monkeypatch):
    """Only config nodes request opening; ordinary captured descriptor data remains inert."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    descriptor = repo.to_definition().to_data()

    def target(take):
        if take:
            return Repo.from_definition(descriptor)
        return "untaken"

    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: pytest.fail("opened")))
    outcome = decode_outcome(
        invoke_invocation(encode_invocation(target, (False,), {}, repo=repo), repo=repo), repo=repo,
    )

    assert outcome["success"]
    assert outcome["result"] == "untaken"


def test_worker_can_construct_an_ordinary_local_managed_config(tmp_path):
    """Config transport does not replace normal worker-local config construction."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    graph = dill.loads(encode_invocation(_worker_constructed_config, (), {}, repo=repo))
    outcome = decode_outcome(invoke_invocation(dill.dumps(graph, protocol=5), repo=repo), repo=repo)

    assert "managed_config" not in {node["tag"] for node in graph["nodes"]}
    assert outcome["success"]
    assert outcome["result"] == (True, 0)
