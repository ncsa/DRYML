from __future__ import annotations

import subprocess
import sys

import dryml
import pytest
from dryml.core2 import Definition, EdgeKind, Object, Repo
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.store.dir import DirStore
from dryml.managed import ManagedOutputRef


class HeavyProducer(Object):
    builds = 0

    def __init__(self, value):
        type(self).builds += 1
        self.value = value


def test_output_ref_identity_is_stable_and_store_independent():
    producer = Definition(HeavyProducer, 3).concretize()

    first = ManagedOutputRef.defn(
        producer=producer,
        method="compute",
        slot="result",
    ).concretize()
    second = ManagedOutputRef.defn(
        producer=producer,
        method="compute",
        slot="result",
    ).concretize()

    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert set(first.kwargs) == {"producer", "method", "slot"}
    assert not any("record" in key or "store" in key or "representation" in key for key in first.kwargs)


@pytest.mark.parametrize("query_index", ["memory", "sqlite"])
def test_output_ref_has_a_queryable_non_materializing_producer_edge(tmp_path, query_index):
    HeavyProducer.builds = 0
    producer = Definition(HeavyProducer, 5).concretize()
    output = ManagedOutputRef.defn(
        producer=producer,
        method="compute",
        slot="result",
    ).concretize()

    graph = ConcreteDefinitionGraph.from_root(output)
    assert len(graph.edges()) == 1
    assert graph.edges()[0].child == producer
    assert graph.edges()[0].kind is EdgeKind.REF

    store_path = tmp_path / query_index
    with Repo([DirStore(store_path, query_index=query_index)]) as repo:
        repo.save_definition(output, main=True)
        selector = Definition(
            ManagedOutputRef,
            producer=Definition(HeavyProducer, 5).ref(),
            method="compute",
            slot="result",
        )
        assert tuple(repo.query(selector).stored().defs()) == (output,)
        assert repo.query(Definition(HeavyProducer, 5)).nested().definitions().known().count() == 1

    with Repo([DirStore(store_path, query_index=query_index)]) as repo:
        assert repo.query(Definition(HeavyProducer, 5)).nested().definitions().stored().count() == 1
        reloaded_def = repo.load_definition(output)
        reloaded = repo.load(reloaded_def)

    assert reloaded.definition == output
    assert reloaded.producer == producer
    assert HeavyProducer.builds == 0


def test_output_ref_public_exports():
    assert dryml.managed.ManagedOutputRef is ManagedOutputRef


def test_managed_authoring_import_is_lightweight():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dryml.managed, sys; "
            "heavy = {'jax', 'ray', 'tensorflow', 'torch', 'pyarrow'}; "
            "assert not ({name.split('.', 1)[0] for name in sys.modules} & heavy)",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
