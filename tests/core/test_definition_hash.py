import base64
from copy import copy, deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import core2_objects as objects
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.cdef_identity import V1_IDENTITY_VERSION, V2_IDENTITY_VERSION
from dryml.core2.definition import ConcreteDefinition, Definition, stable_hash_function
from dryml.core2.utils.general import unpickler


_V1_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "cdef_v1" / "manifest.json"


def _load_v1_fixture():
    manifest = json.loads(_V1_FIXTURE_PATH.read_text())
    return manifest, unpickler(base64.b64decode(manifest["payload"]))


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

    expected = ConcreteDefinition(objects.TestClass1, (10,), {"test": "legacy"})
    assert payload["standalone"] == expected
    assert hash(payload["standalone"]) == hash(expected)


def test_v1_fixture_decoding_does_not_resolve_or_bind_classes():
    code = """
import base64
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
payload = unpickler(base64.b64decode(manifest['payload']))
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
            "import base64, json; from pathlib import Path; "
            "from dryml.core2.utils.general import unpickler; "
            f"m=json.loads(Path(r'{_V1_FIXTURE_PATH}').read_text()); "
            "unpickler(base64.b64decode(m['payload']))",
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
    child_v1 = ConcreteDefinition(objects.TestClass1, (10,), {"test": "child"})
    child_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (10,),
        {"test": "child"},
        identity_version=V2_IDENTITY_VERSION,
    )
    root_v1 = ConcreteDefinition(objects.TestClass1, (child_v1, child_v1), {"test": "root"})
    root_v2 = ConcreteDefinition._from_persisted_record(
        objects.TestClass1,
        (child_v2, child_v2),
        {"test": "root"},
        identity_version=V2_IDENTITY_VERSION,
    )

    assert root_v1 != root_v2
    assert root_v1.stable_hash() != root_v2.stable_hash()
    assert len({root_v1: "v1", root_v2: "v2"}) == 2
    assert copy(root_v2) is root_v2
    assert deepcopy(root_v2) is root_v2

    graph = ConcreteDefinitionGraph.from_roots((root_v1, root_v1, root_v2, root_v2))
    assert graph.roots == (root_v1, root_v2)
    assert {node.definition for node in graph.nodes()} == {root_v1, child_v1, root_v2, child_v2}
