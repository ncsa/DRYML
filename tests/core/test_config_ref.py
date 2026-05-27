import pytest

import core2_objects as objects
from dryml.core2.config import ConfigError, ConfigRef
from dryml.core2.repo import Repo


def test_config_ref_resolves_during_object_construction():
    repo = Repo(config={"data.root": "/local/data"})

    obj = objects.ConfigConsumer(ConfigRef("data.root"), repo=repo)

    assert obj.value == "/local/data"
    assert obj.definition.args[0] == ConfigRef("data.root")


def test_config_ref_missing_key_raises_during_construction():
    repo = Repo()

    with pytest.raises(ConfigError):
        objects.ConfigConsumer(ConfigRef("missing.path"), repo=repo)


def test_config_ref_default_is_used_when_missing():
    repo = Repo()

    obj = objects.ConfigConsumer(ConfigRef("missing.path", default="fallback"), repo=repo)

    assert obj.value == "fallback"


def test_repo_resolve_config_handles_nested_values():
    repo = Repo(config={"root": "/tmp/root", "nested": {"path": "/tmp/nested"}})
    value = {
        "root": ConfigRef("root"),
        "nested": (ConfigRef("nested.path"), [ConfigRef("missing", default=5)]),
    }

    assert repo.resolve_config(value) == {
        "root": "/tmp/root",
        "nested": ("/tmp/nested", [5]),
    }


def test_config_ref_keeps_identity_stable_across_runtime_config(primary_store_set):
    repo = Repo(stores=primary_store_set.stores, config={"data.root": "/local/data"})
    obj = objects.ConfigConsumer(ConfigRef("data.root"), repo=repo)
    repo.save_object(obj, alias="configured")
    repo.close(flush=True)

    repo2 = Repo(stores=primary_store_set.fresh_stores(), config={"data.root": "/cloud/data"})
    loaded = repo2.load_alias("configured")

    assert loaded.definition == obj.definition
    assert loaded.value == "/cloud/data"


def test_config_ref_hash_uses_reference_not_resolved_value():
    left = objects.ConfigConsumer(ConfigRef("path"), repo=Repo(config={"path": "a"}))
    right = objects.ConfigConsumer(ConfigRef("path"), repo=Repo(config={"path": "b"}))

    assert left.definition == right.definition
    assert left.value == "a"
    assert right.value == "b"
