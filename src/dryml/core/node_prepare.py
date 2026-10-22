"""Owned, metadata-only preparation of immutable Node graphs.

One occurrence table serves all requested roots. Exact keys use the v1 hybrid
contract: disjoint child closures compose with a direct-alias partition; joins
use root-relative adjacency. Unordered incidence is canonicalized as a whole,
never by independently ordering graph-bearing members. No external validity
evidence, resolved targets, or process-global cache is retained here.
"""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import permutations
import hashlib
import json
from threading import RLock

from ._content_facts import _primitive_size
from .utils.graph.hasher import fold_closure_masks


EXACT_KEY_VERSION = "dryml-node-exact/v1"


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _digest(domain, value):
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _bytes(value)).hexdigest()


def _refs(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        if value.get("tag") == "node":
            yield value["slot"]
        else:
            for child in value.values():
                yield from _refs(child, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _refs(child, seen)


def _map(value, ref, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, (dict, list, tuple)) and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        if value.get("tag") == "node":
            return ref(value["slot"])
        result = {key: _map(child, ref, memo) for key, child in value.items()}
        memo[id(value)] = result
        return result
    if isinstance(value, (list, tuple)):
        result = [_map(child, ref, memo) for child in value]
        memo[id(value)] = result
        return result
    return value


def _unordered(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return False
        seen.add(id(value))
    if isinstance(value, dict):
        return ((value.get("tag") in ("set", "frozenset") and bool(tuple(_refs(value))))
                or any(_unordered(child, seen) for child in value.values()))
    return isinstance(value, (list, tuple)) and any(_unordered(child, seen) for child in value)


def _incidence(value, path=(), seen=None):
    """Typed incoming positions; unordered member indices are not identities."""
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        if value.get("tag") == "node":
            yield value["slot"], _bytes(path).decode("ascii")
        else:
            role = value.get("role") if value.get("tag") == "link" else value.get("tag")
            for key, child in value.items():
                if key == "items" and value.get("tag") in ("set", "frozenset"):
                    for member in child:
                        yield from _incidence(member, (*path, role, "member"), seen)
                else:
                    yield from _incidence(child, (*path, role, key), seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _incidence(child, (*path, index), seen)


def _value_signature(row, ref, scalars, charge):
    """Fold a local value DAG without expanding shared symbolic operands."""
    memo = {}

    def hashed(value, root=False):
        key = (id(value), root)
        if isinstance(value, (dict, list, tuple)) and key in memo:
            return memo[key]
        if isinstance(value, dict) and value.get("tag") == "node":
            signature = ["node-ref", ref(value["slot"])]
        elif isinstance(value, dict):
            fields = [[name, hashed(child)] for name, child in sorted(value.items())
                      if not (root and name == "stateful_role")
                      and not (name == "finalized" and value.get("tag") == "link")
                      and not (name == "items" and value.get("tag") in ("set", "frozenset"))]
            signature = ["dict", fields]
            if value.get("tag") in ("set", "frozenset"):
                signature.append(sorted(hashed(child) for child in value["items"]))
        elif isinstance(value, (list, tuple)):
            signature = ["list", [hashed(child) for child in value]]
        else:
            scalar = (type(value).__name__, value)
            if scalar not in scalars:
                scalars[scalar] = _digest("dryml-node-value-atom/v1", scalar)
                charge()
            return scalars[scalar]
        if root:
            return signature
        result = _digest("dryml-node-value-step/v1", signature)
        charge()
        memo[key] = result
        return result

    return hashed(row, root=True)


def _canonical_values(rows, ref, *, semantic=True):
    """Exact, hash-free canonical value DAG; incidental value sharing is ignored."""
    records, heights, levels, memo, active = [], [], {}, {}, set()

    def collect(value, root=False):
        key = (id(value), root)
        container = isinstance(value, (dict, list, tuple))
        if container and key in active:
            raise ValueError("Cyclic constructor value.")
        if container and key in memo:
            return memo[key]
        if container:
            active.add(key)
        if isinstance(value, dict) and value.get("tag") == "node":
            record, children = ["node-ref", ref(value["slot"])], []
        elif isinstance(value, dict):
            unordered = value.get("tag") in ("set", "frozenset")
            fields = [[name, collect(child)] for name, child in sorted(value.items())
                      if not (semantic and root and name == "stateful_role")
                      and not (semantic and name == "finalized" and value.get("tag") == "link")
                      and not (unordered and name == "items")]
            items = [collect(child) for child in value["items"]] if unordered else []
            record = ["unordered" if unordered else "dict", fields, items]
            children = [child for _, child in fields] + items
        elif isinstance(value, (list, tuple)):
            children = [collect(child) for child in value]
            record = ["list", children]
        else:
            children = []
            record = ["scalar", type(value).__name__, value]
        index = len(records)
        records.append(record)
        heights.append(1 + max((heights[child] for child in children), default=-1))
        levels.setdefault(heights[index], []).append(index)
        if container:
            active.remove(key)
            memo[key] = index
        return index

    roots = [collect(row, root=True) for row in rows]
    labels, canonical = {}, []
    for height in sorted(levels):
        pending = {}
        for index in levels[height]:
            record = records[index]
            kind = record[0]
            if kind in {"dict", "unordered"}:
                normalized = [kind, [[name, labels[child]] for name, child in record[1]],
                              sorted(labels[child] for child in record[2])]
            elif kind == "list":
                normalized = [kind, [labels[child] for child in record[1]]]
            else:
                normalized = record
            pending[index] = _bytes(normalized)
        unique = {encoded: len(canonical) + offset for offset, encoded in enumerate(sorted(set(pending.values())))}
        canonical.extend(json.loads(encoded) for encoded in sorted(unique))
        labels.update((index, unique[encoded]) for index, encoded in pending.items())
    return {"roots": [labels[root] for root in roots], "values": canonical}


def _labelings(cells, index=0, prefix=()):
    if index == len(cells):
        yield prefix
    else:
        for group in permutations(cells[index]):
            yield from _labelings(cells, index + 1, prefix + group)


@dataclass(frozen=True)
class PreparationView:
    """A named, explicit preparation policy over already supplied Node handles.

    ``name`` is part of key identity. ``transform`` optionally maps a root to an
    immutable semantic view before preparation; it must not perform IO. Creation,
    compatibility and object policies are supplied by their owning consumers,
    rather than inferred from a name. Invalid names/callables raise TypeError.

    Args:
        name (str): Nonempty semantic policy name, default ``exact``.
        transform (Callable | None): Optional pure root-to-Node transformation.
    Side Effects:
        Construction only records policy; the transform runs when preparing roots.
    """

    name: str = "exact"
    transform: object = None

    def __post_init__(self):
        if type(self.name) is not str or not self.name or (
                self.transform is not None and not callable(self.transform)):
            raise TypeError("Preparation views need a name and optional callable transform.")


EXACT_VIEW = PreparationView()


class NodePreparation:
    """Prepare one shared unique table and bounded pure facts for Node roots.

    Args:
        roots (Iterable[Node | ConcreteDefinition]): Immutable roots, in any order.
        scheduler (str): ``eager`` folds in dependency order; ``deferred`` folds
            on request. Both execute the same versioned algorithm.
        view (PreparationView): Explicit semantic policy; defaults to exact.
        previous (NodePreparation | None): Reuse unchanged immutable rows/facts
            after an edit. The new owner retains no previous packet or ancestors.
        max_cached_keys (int): Maximum resident key/fold facts; eviction changes
            work only. The unique input table is owned for this operation's life.
        max_cached_bytes (int): Conservative resident fact-byte budget, including
            keys, digest prefilters and cache bookkeeping (default 8 MiB).
    Raises:
        TypeError, ValueError: On unsupported values, cycles or invalid options.
    Side Effects:
        None beyond owned pure caches and work counts. No symbol resolution, IO,
        Object construction, facet allocation, or external authority validation.
        Key/digest requests and cache clear serialize safely within one owner.

    Attributes:
        nodes (tuple): Immutable source view handles, each discovered once.
        roots (tuple): Selected (possibly explicitly transformed) root handles.
        edges (tuple): View-local (parent slot, typed-position JSON, child slot)
            records. Slots are storage coordinates, never portable identities.
        counts (dict[str, int]): Separate discovery, incidence, folds, semantic
            hashes, fallback/search/frontier, ancestor and export work counters.
    """

    def __init__(self, roots, *, scheduler="deferred", view=EXACT_VIEW,
                 previous=None, max_cached_keys=4096, max_cached_bytes=8 * 1024 * 1024):
        from .definition import ConcreteDefinition
        from .node import Node
        from .node_codec import _describe_node

        if scheduler not in {"eager", "deferred"}:
            raise ValueError("scheduler must be eager or deferred.")
        if not isinstance(view, PreparationView):
            raise TypeError("view must be a PreparationView.")
        if previous is not None and not isinstance(previous, NodePreparation):
            raise TypeError("previous must be a NodePreparation or None.")
        if type(max_cached_keys) is not int or max_cached_keys < 1:
            raise ValueError("max_cached_keys must be positive.")
        if type(max_cached_bytes) is not int or max_cached_bytes < 1:
            raise ValueError("max_cached_bytes must be positive.")
        self.view, self.max_cached_keys, self.max_cached_bytes = view, max_cached_keys, max_cached_bytes
        self.counts = dict.fromkeys(("discovery", "edges", "folds", "hashes",
                                    "fallback_rows", "unordered_ties", "frontier_copies",
                                    "ancestor_work", "export_rows", "export_bytes", "block_hashes",
                                    "reused_rows", "incidence_reuse", "view_rows", "partition_rows", "partition_rounds"), 0)
        self.counts.update(edge_rebindings=0, frontier_bits=0)
        self.counts["value_hashes"] = 0
        self._nodes, self._rows, self._slots, self._construction = [], [], {}, {}
        self._identity_slots, self._handles = {}, []
        self._facts = OrderedDict()
        self._lock = RLock()
        self._fact_sizes, self._cache_bytes = {}, 0
        active, postorder = set(), []
        occurrences = set()
        value_state = {"memo": {}, "active": set()}
        old = previous if previous is not None and previous.view == view else None

        def identity_of(node):
            return node._view if isinstance(node, Node) else node._node_id

        def rebase(label):
            return self._identity_slots[identity_of(old.nodes[label])]

        def visit(node):
            if not isinstance(node, (Node, ConcreteDefinition)):
                raise TypeError("Preparation roots and links must be Nodes or CDefs.")
            marker = id(node)
            identity = identity_of(node)
            if identity in active:
                raise ValueError("Node graph contains a cycle.")
            if marker in self._slots:
                return self._slots[marker]
            self._handles.append(node)
            if identity in self._identity_slots:
                slot = self._identity_slots[identity]
                self._slots[marker] = slot
                return slot
            slot = len(self._nodes)
            self._slots[marker] = slot
            self._identity_slots[identity] = slot
            self._nodes.append(node)
            self._rows.append(None)
            active.add(identity)
            occurrence = node._occurrence if isinstance(node, Node) else node._node_id
            discovered = occurrence not in occurrences
            occurrences.add(occurrence)
            try:
                if old is not None and identity in old._identity_slots:
                    oldslot = old._identity_slots[identity]
                    row = _map(old._rows[oldslot], lambda child: {"tag": "node", "slot": visit(old._nodes[child])})
                    self.counts["reused_rows"] += 1
                    self.counts["edge_rebindings"] += len(old._incidences[oldslot])
                    with old._lock:
                        fact = old._facts.get(oldslot)
                        if fact is not None:
                            fact = (*fact[:2], dict(fact[2]), fact[3])
                    if fact is not None:
                        evidence = fact[3]
                        if evidence is not None:
                            evidence = tuple(tuple(rebase(label) for label in group)
                                             for group in evidence)
                        self._remember(slot, (*fact[:3], evidence))
                else:
                    self.counts["discovery"] += discovered
                    row = _describe_node(node, visit, value_state)
                    self.counts["edges"] += len(tuple(_refs(row)))
                self.counts["view_rows"] += 1
                self._rows[slot] = row
            finally:
                active.remove(identity)
            postorder.append(slot)
            return slot

        supplied = tuple(roots)
        self.roots = tuple(view.transform(root) if view.transform is not None else root for root in supplied)
        for root in self.roots:
            visit(root)
        # Construction handles are storage evidence, not exact realization edges.
        # They share this table, but cannot affect a selected semantic closure.
        index = 0
        while index < len(self._nodes):
            node = self._nodes[index]
            if isinstance(node, Node) and node._construction is not None:
                self._construction[index] = visit(node._construction)
            index += 1
        self.nodes = tuple(self._nodes)
        self._handles = tuple(self._handles)
        self._postorder = tuple(postorder)
        self._children, self._closures, self._schemes = {}, {}, {}
        self._incidences = {}
        for slot in postorder:
            children = tuple(dict.fromkeys(_refs(self._rows[slot])))
            self._children[slot] = children
            closure, overlap = fold_closure_masks(children, self._closures)
            self.counts["frontier_copies"] += len(children)
            self.counts["frontier_bits"] += sum(self._closures[child].bit_length() for child in children)
            self._closures[slot] = closure | (1 << slot)
            self._schemes[slot] = "adjacency" if overlap else "compose"
            identity = identity_of(self.nodes[slot])
            if old is not None and identity in old._identity_slots:
                self._incidences[slot] = tuple((rebase(child), position)
                                               for child, position in old._incidences[old._identity_slots[identity]])
                self.counts["incidence_reuse"] += 1
            else:
                self._incidences[slot] = tuple(_incidence(self._rows[slot]))
        self.edges = tuple((parent, position, child) for parent in postorder
                           for child, position in self._incidences[parent])
        if scheduler == "eager":
            for slot in postorder:
                self._fact(slot)

    def _remember(self, slot, fact):
        # Detached mutable digest maps never connect two preparation owners.
        if len(fact) >= 3:
            fact = (*fact[:2], dict(fact[2]), *fact[3:])
        size = 192 + _primitive_size(fact)
        self._cache_bytes -= self._fact_sizes.pop(slot, 0)
        self._facts.pop(slot, None)
        if size > self.max_cached_bytes:
            return
        self._facts[slot] = fact
        self._fact_sizes[slot] = size
        self._cache_bytes += size
        self._facts.move_to_end(slot)
        while len(self._facts) > self.max_cached_keys or self._cache_bytes > self.max_cached_bytes:
            evicted, _ = self._facts.popitem(last=False)
            self._cache_bytes -= self._fact_sizes.pop(evicted)

    def _slot(self, node):
        try:
            return self._slots[id(node)]
        except KeyError:
            raise ValueError("Node is outside this preparation's table.") from None

    def _fact(self, slot):
        if slot in self._facts:
            self._facts.move_to_end(slot)
            return self._facts[slot]
        children = self._children[slot]
        facts = {child: self._fact(child) for child in children}
        row = self._rows[slot]
        scalars = {}
        def charge():
            self.counts["value_hashes"] += 1
        local = _value_signature(row, lambda child: facts[child][0], scalars, charge)
        value = _digest("dryml-node-value/v1", local)
        self.counts["folds"] += 1
        self.counts["hashes"] += 1
        self.counts["ancestor_work"] += bool(children)
        evidence = None
        if self._schemes[slot] == "adjacency":
            topology, evidence = self._flat(slot, retain=True)
            self.counts["fallback_rows"] += self._closures[slot].bit_count()
        else:
            ordered = children
            if _unordered(row):
                # Whole-root labeling retains aliases crossing unordered members
                # and ordered fields without changing the shape-only scheme.
                _, evidence = self._flat(slot, retain=True)
                positions = {child: index for index, child in enumerate(evidence[0])}
                ordered = sorted(children, key=positions.__getitem__)
            labels = {child: index for index, child in enumerate(ordered)}
            topology = _bytes([_value_signature(row, lambda child: labels[child], scalars, charge),
                               [facts[child][1].decode("ascii") for child in ordered]])
        envelope = _bytes([EXACT_KEY_VERSION, self.view.name, value,
                           self._schemes[slot], _digest("dryml-node-topology/v1", topology.hex())])
        self.counts["hashes"] += 1
        fact = value, envelope, {}, evidence
        self._remember(slot, fact)
        return fact

    def exact_key(self, node):
        """Return canonical v1 key bytes for a prepared Node; reject foreign handles.

        Warm resident requests add no discovery, folds or hashes. Keys are digest
        certificates, not collision-proof equality authority; use ``exact_equal``
        to compare exact normalized evidence at collision-sensitive boundaries.

        Args:
            node (Node | ConcreteDefinition): Handle in this preparation.
        Returns:
            bytes: Versioned exact semantic certificate.
        Raises:
            ValueError: If the handle is outside this table.
        Side Effects:
            Updates only owned pure facts and work counts.
        """
        with self._lock:
            return self._fact(self._slot(node))[1]

    def digest(self, node, *, domain="dryml-node-graph/v1"):
        """Return domain-separated lowercase SHA-256 of a root's exact key bytes.

        ``domain`` (str) is explicit caller-owned identity context. This adds one
        hash on the first request per resident fact/domain and none when warm.
        Invalid domains raise ValueError; unknown handles raise ValueError.
        """
        if type(domain) is not str or not domain or "\0" in domain:
            raise ValueError("A nonempty hash domain is required.")
        with self._lock:
            slot = self._slot(node)
            fact = self._fact(slot)
            # Keep digest facts inside the same bounded key owner.
            if domain not in fact[2]:
                if len(fact[2]) >= 8:
                    fact[2].pop(next(iter(fact[2])))
                fact[2][domain] = hashlib.sha256(domain.encode("ascii") + b"\0" + fact[1]).hexdigest()
                self.counts["hashes"] += 1
                self._remember(slot, fact)
            return fact[2][domain]

    def scheme(self, node):
        """Return the shape-only compose/adjacency choice for a prepared Node.

        Args:
            node (Node | ConcreteDefinition): Handle in this preparation.
        Returns:
            str: ``compose`` or ``adjacency``.
        Raises:
            ValueError: If the handle is outside this table.
        Side Effects:
            None; does not fold values or hash keys.
        """
        return self._schemes[self._slot(node)]

    def clear(self):
        """Evict owned value/key/digest facts without changing the immutable table.

        Returns None. Source handles and work counts are preserved; subsequent
        requests recompute evicted facts under the same contract and budgets.
        """
        with self._lock:
            self._facts.clear()
            self._fact_sizes.clear()
            self._cache_bytes = 0

    def exact_equal(self, node, other, other_node):
        """Compare full normalized rooted evidence, even under digest collisions.

        ``other`` is another NodePreparation and ``other_node`` its selected root.
        Returns bool; foreign handles raise ValueError. Does not use digest equality
        as authority or consult external state. Exact unordered symmetry search can
        be expensive; counts expose that cost rather than imposing a refusal limit.
        """
        if self.view.name != other.view.name:
            return False
        with self._lock:
            left = self._flat(self._slot(node))
        with other._lock:
            right = other._flat(other._slot(other_node))
        return left == right

    def _flat(self, root, *, retain=False):
        cached = self._facts.get(root)
        if cached is not None and cached[3] is not None:
            order = cached[3][0]
            labels = {slot: index for index, slot in enumerate(order)}
            result = _bytes(_canonical_values([self._rows[slot] for slot in order], labels.__getitem__))
            return (result, cached[3]) if retain else result
        closure = self._closures[root]
        slots = []
        remaining = closure
        while remaining:
            bit = remaining & -remaining
            slots.append(bit.bit_length() - 1)
            remaining ^= bit
        if not any(_unordered(self._rows[slot]) for slot in slots):
            order, seen = [], set()
            def visit(slot):
                if slot in seen:
                    return
                seen.add(slot)
                order.append(slot)
                for child in self._children[slot]:
                    visit(child)
            visit(root)
            labels = {slot: index for index, slot in enumerate(order)}
            result = _bytes(_canonical_values([self._rows[slot] for slot in order], labels.__getitem__))
            evidence = (tuple(order),)
            return (result, evidence) if retain else result
        # Equitable refinement uses exact bytes, not hashes. Individualization of
        # remaining color cells considers complete rooted incidence, including all
        # unordered/external aliases. Slot numbers never break a semantic tie.
        colors = {slot: 0 for slot in slots}
        while True:
            signatures = {slot: _bytes([slot == root, colors[slot],
                                       _canonical_values([self._rows[slot]], colors.__getitem__)]) for slot in slots}
            self.counts["partition_rounds"] += 1
            self.counts["partition_rows"] += len(slots)
            palette = {signature: index for index, signature in enumerate(sorted(set(signatures.values())))}
            updated = {slot: palette[signatures[slot]] for slot in slots}
            if len(set(colors.values())) == len(set(updated.values())):
                colors = updated
                break
            colors = updated
        cells = [[slot for slot in slots if slot != root and colors[slot] == color] for color in sorted(set(colors.values()))]
        cells = [cell for cell in cells if cell]
        best = None
        best_order = None
        for labeling in _labelings(cells):
            order = [root, *labeling]
            labels = {slot: index for index, slot in enumerate(order)}
            candidate = _bytes(_canonical_values([self._rows[slot] for slot in order], labels.__getitem__))
            if best is None or candidate < best:
                best = candidate
                best_order = tuple(order)
            if any(len(cell) > 1 for cell in cells):
                self.counts["unordered_ties"] += 1
        evidence = (best_order, *(tuple(cell) for cell in cells))
        return (best, evidence) if retain else best
