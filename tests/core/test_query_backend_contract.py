from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from dryml.core import Definition, Object, Repo, SKIP_ARGS
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.cdef_identity import same_cdef
from dryml.core.query.domain import StoredDomain
from dryml.core.query.fingerprint import target_local_fingerprint
from dryml.core.query.graph_plan import graph_candidate_ids
from dryml.core.query.lowering import LoweringDiagnostics, ScanPolicy
from dryml.core.query.federation import IndexGenerationVector, RepoGenerationVector
from dryml.core.query.model import (
    DefinitionEdgeRecord,
    IndexWriteResult,
    OccurrenceTraversalSnapshot,
    ReconcileReport,
    StoreReplica,
    StoredReplica,
    StoredRootMetadata,
)
from dryml.core.query.memory import MemoryStoreQueryIndex
from dryml.core.query.path import DefinitionPath
from dryml.core.query.selector_graph import compile_selector_graph
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore


class ContractLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class ContractParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class ContractPair(Object):
    def __init__(self, left=None, right=None, *, name="pair"):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name


class ContractVariadic(Object):
    def __init__(self, *values, **labels):
        super().__init__()
        self.values = values
        self.labels = labels


class MemoryContractIndex:
    source_key = "contract-memory-index"

    def __init__(self, store):
        self.repo = Repo(stores=store)
        self.store = store
        self.index = MemoryStoreQueryIndex(self.repo._query_catalog, self.store)

    def read_view(self, *, include_cached=True):
        return self.index.read_view(include_cached=include_cached)

    def current_generation(self):
        return self.index.current_generation()

    def register_stored_roots(self, graph, roots):
        return self.index.register_stored_roots(graph, roots)

    def remove_stored_roots(self, roots):
        return self.index.remove_stored_roots(roots)

    def refresh(self, policy, *, stats=None):
        return self.index.refresh(policy, stats=stats)

    def close(self):
        return self.index.close()


@dataclass(frozen=True)
class FakeRecord:
    did: int
    cdef: object


@dataclass(frozen=True)
class FakeEdge:
    parent_id: int
    path: object
    child_id: int


class FakeContractIndex:
    source_key = "contract-fake-index"

    def __init__(self):
        self._records = ()
        self._edges = ()
        self._stored_ids = frozenset()
        self._generation = 0

    @contextmanager
    def read_view(self, *, include_cached=True):
        view = FakeReadView(
            source_key=self.source_key,
            generation=self._generation,
            records=self._records,
            edges=self._edges,
            stored_ids=self._stored_ids,
        )
        try:
            yield view
        finally:
            view.close()

    def current_generation(self):
        return self._generation

    def register_stored_roots(self, graph, roots):
        roots = tuple(dict.fromkeys(roots))
        records = list(self._records)
        edges = list(self._edges)
        stored_ids = set(self._stored_ids)
        changed = False

        for node in graph.nodes():
            cdef = node.definition
            if _fake_id(records, cdef) is None:
                records.append(FakeRecord(len(records) + 1, cdef))
                changed = True

        for edge in graph.edges():
            parent_id = _fake_id(records, edge.parent)
            child_id = _fake_id(records, edge.child)
            fake_edge = FakeEdge(parent_id, edge.path, child_id)
            if fake_edge not in edges:
                edges.append(fake_edge)
                changed = True

        roots_added = 0
        for root in roots:
            did = _fake_id(records, root)
            if did not in stored_ids:
                stored_ids.add(did)
                roots_added += 1
                changed = True

        if changed:
            self._records = tuple(records)
            self._edges = tuple(edges)
            self._stored_ids = frozenset(stored_ids)
            self._generation += 1
        return IndexWriteResult(generation=self._generation, changed=changed, roots_added=roots_added)

    def remove_stored_roots(self, roots):
        stored_ids = set(self._stored_ids)
        removed = 0
        for root in tuple(dict.fromkeys(roots)):
            did = _fake_id(self._records, root)
            if did in stored_ids:
                stored_ids.remove(did)
                removed += 1
        if removed:
            self._stored_ids = frozenset(stored_ids)
            self._generation += 1
        return IndexWriteResult(generation=self._generation, changed=bool(removed), roots_removed=removed)

    def refresh(self, policy, *, stats=None):
        return None

    def close(self):
        return None


class FakeReadView:
    def __init__(self, *, source_key, generation, records, edges, stored_ids):
        self.source_key = source_key
        self.generation = generation
        self._records = records
        self._edges = edges
        self._stored_ids = stored_ids
        self._active = True

    def close(self):
        self._active = False

    def _check_active(self):
        if not self._active:
            raise RuntimeError("fake read view is closed")

    def all_definition_ids(self):
        self._check_active()
        return {record.did for record in self._records}

    def exact_ids(self, cdef):
        self._check_active()
        return {record.did for record in self._records if same_cdef(record.cdef, cdef)}

    def estimate_exact_ids(self, cdef):
        return len(self.exact_ids(cdef))

    def estimate_local_candidates(self, requirements):
        self._check_active()
        if not requirements:
            return len(self._records)
        return min(len(self._ids_for_requirement(req)) for req in requirements)

    def local_candidates(self, requirements, *, within=None, domain=None, stats=None):
        self._check_active()
        if not requirements:
            candidates = self.all_definition_ids() if within is None else set(within)
        else:
            candidates = None
            ordered = sorted(requirements, key=lambda req: len(self._ids_for_requirement(req)))
            for req in ordered:
                ids = self._ids_for_requirement(req)
                candidates = ids if candidates is None else candidates & ids
            if within is not None:
                candidates &= set(within)
        if domain is not None:
            candidates = domain.filter(candidates)
        if stats is not None:
            stats.candidate_count = len(candidates)
        return candidates

    def parents(self, child_ids, path, *, unordered, within=None):
        self._check_active()
        parents = {
            edge.parent_id
            for edge in self._edges
            if edge.child_id in child_ids and (edge.path == path or (unordered and edge.path.startswith(path)))
        }
        return parents if within is None else parents & set(within)

    def children(self, parent_ids, path, *, unordered, within=None):
        self._check_active()
        children = {
            edge.child_id
            for edge in self._edges
            if edge.parent_id in parent_ids and (edge.path == path or (unordered and edge.path.startswith(path)))
        }
        return children if within is None else children & set(within)

    def is_stored_id(self, did):
        self._check_active()
        return did in self._stored_ids

    def filter_stored_ids(self, ids):
        self._check_active()
        return set(ids) & set(self._stored_ids)

    def all_stored_ids(self):
        self._check_active()
        return set(self._stored_ids)

    def is_cached_id(self, did, *, reuse_weak=True):
        return False

    def all_cached_ids(self, *, reuse_weak=True):
        return set()

    def all_known_ids(self, *, reuse_weak=True):
        return self.all_stored_ids()

    def nested_ids(self):
        nested = set()
        stack = list(self._stored_ids)
        seen = set(stack)
        while stack:
            parent_id = stack.pop()
            for child_id in self.children({parent_id}, _ROOT_PATH, unordered=True):
                if child_id in seen:
                    continue
                seen.add(child_id)
                nested.add(child_id)
                stack.append(child_id)
        return nested

    def filter_nested_ids(self, ids):
        return {did for did in ids if self.has_stored_ancestor(did)}

    def has_stored_ancestor(self, did):
        seen = {did}
        stack = [did]
        while stack:
            child_id = stack.pop()
            for parent_id in self.parents({child_id}, _ROOT_PATH, unordered=True):
                if parent_id in self._stored_ids:
                    return True
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                stack.append(parent_id)
        return False

    def cdefs_by_id(self, ids):
        self._check_active()
        id_set = set(ids)
        return {record.did: record.cdef for record in self._records if record.did in id_set}

    def replica_map(self, ids):
        return {cdef: (self.source_key,) for cdef in self.cdefs_by_id(self.filter_stored_ids(ids)).values()}

    def project_owners(self, ids):
        snapshot = self.occurrence_snapshot_for_nested_ids(ids)
        return snapshot.project_owners(set(ids))

    def occurrence_snapshot_for_nested_ids(self, target_ids):
        target_ids = set(target_ids)
        incoming = {}
        cdefs = self.cdefs_by_id(target_ids)
        stored_ids = set()
        seen = set()
        stack = list(target_ids)
        while stack:
            child_id = stack.pop()
            if child_id in seen:
                continue
            seen.add(child_id)
            for edge in self._incoming_edges(child_id):
                incoming.setdefault(edge.child_id, []).append(edge)
                cdefs.update(self.cdefs_by_id({edge.parent_id, edge.child_id}))
                if edge.parent_id in self._stored_ids:
                    stored_ids.add(edge.parent_id)
                if edge.parent_id not in seen:
                    stack.append(edge.parent_id)
        return OccurrenceTraversalSnapshot(
            targets=target_ids,
            cdefs=cdefs,
            stored_ids=stored_ids,
            incoming={child_id: tuple(edges) for child_id, edges in incoming.items()},
            owner_replicas=self.replica_map(stored_ids),
        )

    def _ids_for_requirement(self, requirement):
        return {
            record.did
            for record in self._records
            if target_local_fingerprint(record.cdef).counts.get(requirement.token, 0) >= requirement.count
        }

    def _incoming_edges(self, child_id):
        return tuple(
            DefinitionEdgeRecord((edge.parent_id, edge.path, edge.child_id), edge.parent_id, edge.path, edge.child_id)
            for edge in self._edges
            if edge.child_id == child_id
        )


def _fake_id(records, cdef):
    for record in records:
        if same_cdef(record.cdef, cdef):
            return record.did
    return None


_ROOT_PATH = DefinitionPath()


@dataclass(frozen=True)
class BackendCase:
    name: str
    index: object


@pytest.fixture(params=["fake", "memory", "sqlite"])
def backend_case(request, tmp_path):
    if request.param == "fake":
        return BackendCase("fake", FakeContractIndex())
    if request.param == "memory":
        return BackendCase("memory", MemoryContractIndex(DirStore(tmp_path / "memory", query_index="memory")))
    if not sqlite_available():
        pytest.skip("sqlite3 is unavailable")
    return BackendCase(
        "sqlite",
        SQLiteStoreQueryIndex(
            source_key="contract-sqlite-store",
            path=tmp_path / "index.sqlite",
            config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
        ),
    )


def test_contract_register_exact_lookup_and_generation(backend_case):
    index = backend_case.index
    obj = ContractLeaf(name="exact")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)

    result = index.register_stored_roots(graph, [obj.definition])
    repeat = index.register_stored_roots(graph, [obj.definition])

    assert result.changed
    assert repeat.changed is False
    assert index.current_generation() == result.generation
    with index.read_view(include_cached=False) as view:
        dids = view.exact_ids(obj.definition)
        assert len(dids) == 1
        did = next(iter(dids))
        assert view.generation == result.generation
        assert view.is_stored_id(did)
        assert view.cdefs_by_id(dids) == {did: obj.definition}


def test_contract_phase0_record_types_are_available():
    replica = StoreReplica(definition_id="def-1", store_id="store-1")
    root = StoredRootMetadata(
        definition_id="def-1",
        store_id="store-1",
        storage_hash="abc123",
        relative_def_path="objects/abc123.dry",
        indexed_generation=2,
    )
    report = ReconcileReport(
        backend="memory",
        store_key="store-1",
        changed=True,
        action="rebuild",
        generation_before=1,
        generation_after=2,
        roots_added=1,
    )
    vector = IndexGenerationVector({"store-1": 2})

    assert StoredReplica is StoreReplica
    assert RepoGenerationVector is IndexGenerationVector
    assert replica.definition_id == root.definition_id
    assert report.generation_after == vector.generations["store-1"]


def test_contract_local_candidates_and_planner(backend_case):
    index = backend_case.index
    wanted = ContractParent(child=ContractLeaf(name="wanted"), name="root")
    other = ContractParent(child=ContractLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector = Definition(ContractParent, SKIP_ARGS, child=Definition(ContractLeaf, SKIP_ARGS, name="wanted"))
    selector_graph = compile_selector_graph(selector)

    with index.read_view(include_cached=False) as view:
        domain = StoredDomain(view)
        candidate_ids = graph_candidate_ids(view, selector_graph, domain)
        cdefs = tuple(view.cdefs_by_id(candidate_ids).values())

    assert cdefs == (wanted.definition,)


def test_memory_variadic_selector_uses_semantic_requirements_without_scanning(tmp_path):
    contract = MemoryContractIndex(DirStore(tmp_path / "memory", query_index="memory"))
    wanted = ContractVariadic("wanted", marker="wanted")
    decoys = tuple(ContractVariadic(f"decoy-{index}", marker=f"decoy-{index}") for index in range(999))
    definitions = (wanted.definition, *(decoy.definition for decoy in decoys))
    contract.register_stored_roots(
        ConcreteDefinitionGraph.from_roots(definitions),
        definitions,
    )
    selector_graph = compile_selector_graph(Definition(ContractVariadic, "wanted", marker="wanted"))

    assert selector_graph is not None
    assert not selector_graph.requires_scan
    with contract.read_view(include_cached=False) as view:
        candidates = graph_candidate_ids(view, selector_graph, StoredDomain(view))
        cdefs = tuple(view.cdefs_by_id(candidates).values())

    assert cdefs == (wanted.definition,)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_variadic_selector_lowers_without_scanning_or_decoding_decoys(tmp_path):
    index = SQLiteStoreQueryIndex(
        source_key="contract-variadic-sqlite-store",
        path=tmp_path / "index.sqlite",
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
    )
    wanted = ContractVariadic("wanted", marker="wanted")
    decoys = tuple(ContractVariadic(f"decoy-{index}", marker=f"decoy-{index}") for index in range(999))
    definitions = (wanted.definition, *(decoy.definition for decoy in decoys))
    index.register_stored_roots(ConcreteDefinitionGraph.from_roots(definitions), definitions)
    selector_graph = compile_selector_graph(Definition(ContractVariadic, "wanted", marker="wanted"))
    diagnostics = LoweringDiagnostics()

    assert selector_graph is not None
    with index.read_view(include_cached=False) as view:
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy("forbid"),
            diagnostics=diagnostics,
        )
        batches = tuple(view.iter_candidate_cdef_batches(plan, batch_size=1000))

    assert not diagnostics.scan_required
    assert tuple(cdef for batch in batches for cdef in batch.cdefs) == (wanted.definition,)
    assert diagnostics.cdef_blobs_decoded == 1


def test_contract_parent_child_and_nested_semantics(backend_case):
    index = backend_case.index
    leaf = ContractLeaf(name="nested")
    owner = ContractPair(left=leaf, right=ContractLeaf(name="other"), name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])
    left_path = next(edge.path for edge in graph.edges() if edge.child == leaf.definition)

    with index.read_view(include_cached=False) as view:
        owner_id = next(iter(view.exact_ids(owner.definition)))
        leaf_id = next(iter(view.exact_ids(leaf.definition)))
        assert view.children({owner_id}, left_path, unordered=False) == {leaf_id}
        assert view.parents({leaf_id}, left_path, unordered=False) == {owner_id}
        assert view.children({owner_id}, left_path.parent, unordered=True) >= {leaf_id}
        assert leaf_id in view.nested_ids()
        assert view.filter_nested_ids({owner_id, leaf_id}) == {leaf_id}
        assert view.has_stored_ancestor(leaf_id)


def test_contract_owner_projection_and_occurrence_capture(backend_case):
    index = backend_case.index
    leaf = ContractLeaf(name="shared")
    owner1 = ContractParent(child=leaf, name="owner1")
    owner2 = ContractParent(child=leaf, name="owner2")
    graph = ConcreteDefinitionGraph.from_roots([owner1.definition, owner2.definition])
    index.register_stored_roots(graph, [owner1.definition, owner2.definition])

    with index.read_view(include_cached=False) as view:
        leaf_id = next(iter(view.exact_ids(leaf.definition)))
        projection = view.project_owners({leaf_id})
        occurrences = tuple(view.occurrence_snapshot_for_nested_ids({leaf_id}).iter_occurrences())

    assert set(projection.cdefs) == {owner1.definition, owner2.definition}
    assert {occ.owner for occ in occurrences} == {owner1.definition, owner2.definition}
    assert {occ.definition for occ in occurrences} == {leaf.definition}
    assert all(str(occ.path) == '$[@param("child")]' for occ in occurrences)


def test_contract_remove_stored_roots_when_supported(backend_case):
    index = backend_case.index
    obj = ContractLeaf(name="remove")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    index.register_stored_roots(graph, [obj.definition])

    result = index.remove_stored_roots([obj.definition])

    assert result.changed
    with index.read_view(include_cached=False) as view:
        did = next(iter(view.exact_ids(obj.definition)))
        assert not view.is_stored_id(did)
