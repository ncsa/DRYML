import base64
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import core2_objects as objects
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.cdef_identity import V1_IDENTITY_VERSION, V2_IDENTITY_VERSION
from dryml.core2.bound_args import BoundArguments
from dryml.core2.definition import ConcreteDefinition, Definition, stable_hash_function
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.utils.general import unpickler
from dryml.core2.utils.stable_hash import StableHashGraphHasher


_V1_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "manifest.json"
_V1_FIXTURE_PRODUCER = "85ea268860091f96b97fa9031ac813beb369c749"


def _validate_v1_fixture_manifest(manifest):
    assert manifest["format"] == "dill-base64/v1"
    assert manifest["producer"] == {
        "commit": _V1_FIXTURE_PRODUCER,
        "dill": "unrecorded",
        "pickle_protocol": 5,
        "python": "3.8 (declared by dev-env.yaml)",
        "runtime": "unrecorded",
    }
    payload = base64.b64decode(manifest["payload"], validate=True)
    assert hashlib.sha256(payload).hexdigest() == manifest["payload_sha256"]
    assert payload[:2] == b"\x80\x05"
    return payload


def _load_v1_fixture():
    manifest = json.loads(_V1_FIXTURE_PATH.read_text())
    return manifest, unpickler(_validate_v1_fixture_manifest(manifest))


def test_definition_hash_1():
    definition1 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass1,
        10, test='a')

    def_hash_2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash_2


def test_definition_hash_2():
    definition1 = Definition(
        objects.TestClass2,
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_4():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_5():
    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            10,
            test='B'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 != def_hash2


def test_definition_hash_6():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 == def_hash2


def test_definition_hash_7():
    arr = np.random.random((10,10)).astype(np.float32)
    arr2 = np.copy(arr)
    arr2[0,0] = 5.

    definition1 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr,
            test='A'),
        var1='a',
        var2='b',
        var3='c',
        var4='d')

    def_hash1 = stable_hash_function(definition1)

    definition2 = Definition(
        objects.TestClass2,
        Definition(
            objects.TestClass1,
            arr2,
            test='A'),
        var4='d',
        var3='c',
        var2='b',
        var1='a')

    def_hash2 = stable_hash_function(definition2)

    assert def_hash1 != def_hash2


def test_prechange_cdef_fixture_keeps_v1_hashes_paths_and_topology():
    manifest, payload = _load_v1_fixture()

    for name, expected_hash in manifest["hashes"].items():
        assert payload[name].identity_version == V1_IDENTITY_VERSION
        assert payload[name].stable_hash() == expected_hash

    assert payload["nested_shared"][0] is payload["nested_shared"][1]
    assert payload["object_root"] is payload["main_definition"]
    assert payload["aliases"]["primary"] is payload["object_root"]
    assert payload["aliases"]["secondary"] is payload["object_root"]
    for name, path in manifest["object_paths"].items():
        cdef = payload[name]
        assert path == f"objects/{cdef.stable_hash()[:2]}/{cdef.stable_hash()}"

    expected = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (10,),
        {"test": "legacy"},
    )
    assert payload["standalone"] == expected
    assert hash(payload["standalone"]) == hash(expected)


def test_v1_fixture_manifest_validates_bytes_before_unpickling():
    manifest = json.loads(_V1_FIXTURE_PATH.read_text())

    payload = _validate_v1_fixture_manifest(manifest)
    assert payload

    invalid_manifest = dict(manifest, payload_sha256="0" * 64)
    with pytest.raises(AssertionError):
        _validate_v1_fixture_manifest(invalid_manifest)


def test_v1_fixture_decoding_does_not_resolve_or_bind_classes():
    code = """
import base64
import hashlib
import inspect
import json
from pathlib import Path
import dryml.core2.canonical as canonical
import dryml.core2.definition as definition
from dryml.core2.utils.general import unpickler

def fail(*args, **kwargs):
    raise AssertionError('decoding must not resolve, inspect, prepare, or bind')

canonical.resolve_symbol = fail
definition.resolve_symbol = fail
inspect.signature = fail
manifest = json.loads(Path(r'__FIXTURE__').read_text())
payload = base64.b64decode(manifest['payload'], validate=True)
assert hashlib.sha256(payload).hexdigest() == manifest['payload_sha256']
assert payload[:2] == bytes((0x80, 5))
payload = unpickler(payload)
assert payload['standalone'].identity_version == 1
assert payload['symbolic'].stable_hash() == manifest['hashes']['symbolic']
""".replace("__FIXTURE__", str(_V1_FIXTURE_PATH))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        env={"PYTHONPATH": f"{Path(__file__).parents[1]}:{Path(__file__).parent}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_v1_fixture_keeps_symbolic_hydration_import_free_and_raw_classes_explicit():
    manifest, payload = _load_v1_fixture()

    assert payload["symbolic"].cls.module not in sys.modules
    assert payload["symbolic"].stable_hash() == manifest["hashes"]["symbolic"]
    assert isinstance(payload["raw_class"].cls, type)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import base64, hashlib, json; from pathlib import Path; "
            "from dryml.core2.utils.general import unpickler; "
            f"m=json.loads(Path(r'{_V1_FIXTURE_PATH}').read_text()); "
            "p=base64.b64decode(m['payload'], validate=True); "
            "assert hashlib.sha256(p).hexdigest() == m['payload_sha256']; "
            "assert p[:2] == bytes((0x80, 5)); unpickler(p)",
        ],
        cwd=Path(__file__).parents[2],
        env={"PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "core2_objects" in result.stderr


def test_v1_and_private_v2_records_are_distinct_mapping_keys_and_graph_nodes():
    child_v1 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (10,),
        {"test": "child"},
    )
    child_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "child"))),
    )
    root_v1 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (child_v1, child_v1),
        {"test": "root"},
    )
    root_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("values", FrozenTuple((child_v2, child_v2))), ("test", "root"))),
    )

    assert root_v1 != root_v2
    assert root_v1.stable_hash() != root_v2.stable_hash()
    assert len({root_v1: "v1", root_v2: "v2"}) == 2
    assert copy(root_v2) is root_v2
    assert deepcopy(root_v2) is root_v2

    graph = ConcreteDefinitionGraph.from_roots((root_v1, root_v1, root_v2, root_v2))
    assert graph.roots == (root_v1, root_v2)
    assert {node.definition for node in graph.nodes()} == {root_v1, child_v1, root_v2, child_v2}


def test_v2_persisted_hash_cache_must_match_identity_record():
    cdef = Definition(objects.TestClass1, 10, test="cached").concretize()

    with pytest.raises(ValueError, match="hash cache"):
        ConcreteDefinition._from_persisted_record(
            cdef.cls,
            identity_version=V2_IDENTITY_VERSION,
            parameters=cdef._bound_args,
            stable_hash_cache="0" * 64,
        )

    restored = ConcreteDefinition._from_persisted_record(
        cdef.cls,
        identity_version=V2_IDENTITY_VERSION,
        parameters=cdef._bound_args,
        stable_hash_cache=cdef.stable_hash(),
    )
    assert restored == cdef
    assert restored.stable_hash() == cdef.stable_hash()


def test_v2_persisted_hash_cache_validation_rehashes_legacy_children():
    """V2 cache validation does not trust unvalidated V1 child caches."""

    child = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (10,),
        {"test": "legacy-child"},
        stable_hash_cache="0" * 64,
    )
    cdef = ConcreteDefinition._from_persisted_record(
        objects.TestNest2,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("A", child),)),
    )
    state = cdef.__getstate__()
    state["stable_hash_cache"] = cdef.stable_hash()

    restored = object.__new__(ConcreteDefinition)
    restored.__setstate__(state)

    assert restored.stable_hash() == cdef.stable_hash()


@pytest.mark.parametrize("depth", (10, 20, 40))
def test_v2_persisted_hash_cache_validation_reuses_validated_nested_hashes(monkeypatch, depth):
    """Hydrating cached nested V2 records performs linear hash work."""

    cdef = Definition(objects.TestNest2, None).concretize()
    cdef.stable_hash()
    persisted = [cdef]
    for _ in range(depth - 1):
        cdef = Definition(objects.TestNest2, cdef).concretize()
        cdef.stable_hash()
        persisted.append(cdef)

    dispatch = StableHashGraphHasher.dispatch
    cdef_visits = 0

    def count_cdef_visits(self, obj, ctx):
        nonlocal cdef_visits
        if isinstance(obj, ConcreteDefinition):
            cdef_visits += 1
        return dispatch(self, obj, ctx)

    monkeypatch.setattr(StableHashGraphHasher, "dispatch", count_cdef_visits)
    restored = None
    for state_source in persisted:
        state = state_source.__getstate__()
        if restored is not None:
            state = dict(state)
            state["parameters"] = FrozenDict(
                (name, restored if isinstance(value, ConcreteDefinition) else value)
                for name, value in state["parameters"].items()
            )
        next_restored = object.__new__(ConcreteDefinition)
        next_restored.__setstate__(state)
        restored = next_restored

    assert restored == cdef
    assert cdef_visits <= 2 * depth


def test_v2_parameter_order_is_not_part_of_identity():
    first = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 10), ("test", "ordered"))),
    )
    second = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("test", "ordered"), ("value", 10))),
    )

    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert len({first, second}) == 1
