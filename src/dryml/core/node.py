"""Occurrence-aware immutable graph views, constraint edits and identity allocation.

Definition and the normalized ConcreteNode use the existing constructor record
representation. Facets live on their occurrence views; rooted maps are adapters.
Retained construction views are immutable handles, never mutable facet tables.
Explicit identity allocation fills concrete owned stateful occurrences without
constructing Objects, selecting state, or publishing Store authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar
import numpy as np

from .freeze import FrozenDict, FrozenTuple, FrozenList, FrozenSet, FrozenNDArray
from .utils.graph.path import GraphPath, GraphPathError, normalize_path

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class _Facets:
    object_id: object = None
    state: object = None


def _init_node(node, *, source=None):
    object.__setattr__(node, "_occurrence", object() if source is None else source._occurrence)
    object.__setattr__(node, "_view", object())
    object.__setattr__(node, "_facets", _Facets() if source is None else source._facets)
    object.__setattr__(node, "_construction", None if source is None else source._construction)


def _as_node_view(value, memo=None):
    """Adapt existing immutable CDef records without another recipe representation."""
    from .bound_args import BoundArguments
    from .definition import ConcreteDefinition, ConcreteNode
    from .links import DefLink

    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, Node):
        return value
    if isinstance(value, ConcreteDefinition):
        result = object.__new__(ConcreteNode)
        result._copy_record(value)
        object.__setattr__(result, "_occurrence", value._node_id)
        object.__setattr__(result, "_bound_args", BoundArguments(
            (name, _as_node_view(child, memo)) for name, child in value.parameters.items()))
    elif isinstance(value, DefLink):
        target = _as_node_view(value.target, memo)
        result = value if target is value.target else (DefLink.finalized(value.kind, target) if value.is_finalized else DefLink.assertion(value.kind, target))
    elif isinstance(value, Mapping):
        result = FrozenDict((k, _as_node_view(v, memo)) for k, v in value.items())
    elif isinstance(value, (FrozenList, list)):
        result = FrozenList(_as_node_view(v, memo) for v in value)
    elif isinstance(value, (FrozenTuple, tuple)):
        result = FrozenTuple(_as_node_view(v, memo) for v in value)
    elif isinstance(value, (FrozenSet, set)):
        result = FrozenSet(_as_node_view(v, memo) for v in value)
    elif isinstance(value, frozenset):
        result = frozenset(_as_node_view(v, memo) for v in value)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype arrays are not immutable construction values.")
        result = value if isinstance(value, FrozenNDArray) else FrozenNDArray.from_array(value)
    elif isinstance(value, np.generic) and value.dtype.kind == "V":
        if value.dtype.hasobject:
            raise TypeError("Object-dtype scalars are not immutable construction values.")
        result = np.ndarray((), dtype=value.dtype, buffer=value.tobytes())[()]
    else:
        result = value
    memo[id(value)] = result
    return result


class Node(Generic[T]):
    """Common read-only carrier for soft and normalized constructor graphs.

    Author instances with ``Definition(...)`` and normalize with ``concretize``.
    Identity and local state are independent exact constraints, not constructor
    keywords. Node equality/hash identify an immutable occurrence/view handle;
    use named comparisons for semantic structure. Binding, navigation, extraction
    and replacement never resolve targets, construct Objects, allocate ObjectIds,
    read payloads or publish data. Invalid paths raise GraphPathError; conflicting
    exact constraints raise ValueError and invalid facet types raise TypeError.
    Explicit ``allocate_identities`` fills missing owned stateful identities on a
    concrete graph, preserving existing facets and construction evidence.
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is Node:
            raise TypeError("Author a Node with Definition(...).")
        return super().__new__(cls)

    def __eq__(self, other):
        return isinstance(other, Node) and self._view is other._view

    def __hash__(self):
        return hash(self._view)

    def __ne__(self, other):
        return not self == other

    @property
    def object_id(self):
        """Return this occurrence's exact ObjectId constraint, or None."""
        return self._facets.object_id

    def concretize(self, repo=None):
        """Normalize supplied construction constraints without realizing Objects.

        Args:
            repo (Repo | metadata resolver | None): Existing repository or object
                exposing ``resolve_state_selector(selector)`` when required.
        Returns:
            ConcreteNode: Normalized graph, or this handle if already concrete.
        Raises:
            Exception: Propagates signature, symbolic and value-admission failures.
        Side Effects:
            May resolve targets and run preparation hooks, but never constructs,
            allocates ObjectIds, reads private payloads or publishes data.
        """
        from .definition import ConcreteNode
        return self if isinstance(self, ConcreteNode) else concretize_node(self, repo=repo)

    def build(self, *, repo=None, cache="weak", identities=None, states=None):
        """Concretize and realize this graph with optional exact facet constraints.

        Args:
            repo (Repo | None): Repository owning construction and state sources.
            cache (CachePolicy): Result cache policy, default ``weak``.
            identities, states (Mapping | None): Rooted exact constraints composed
                before concretization. Conflicting assignments raise ValueError.
        Returns:
            Object: Newly constructed or reused runtime result.
        Raises:
            TypeError: If the recipe cannot be concretized/materialized.
            RepoLoadError: If selected state is unavailable or realization fails.
        Side Effects:
            Allocates only missing identities, initializes unspecified nodes and
            restores selected nodes once. All selections preflight before hooks.
        """
        from .repo import manage_repo
        with manage_repo(repo=repo) as selected:
            return selected.build(self.bind(identities=identities, states=states).concretize(repo=selected), cache=cache)

    @property
    def state(self):
        """Return this occurrence's exact LocalState constraint, or None."""
        return self._facets.state

    @property
    def construction_view(self):
        """Return retained construction evidence, unaffected by realization edits."""
        return self if self._construction is None else self._construction

    def at(self, path="$", *, traverse_refs=False, traverse_quoted=False):
        """Return the Node endpoint of a typed or textual metadata path.

        Args:
            path (GraphPathLike): Root-relative path; defaults to this Node.
            traverse_refs (bool): Permit descent through Ref boundaries.
            traverse_quoted (bool): Independently permit quotation descent.
        Returns:
            The selected Node view; a boundary endpoint needs no descent opt-in.
        Raises:
            GraphPathError: For missing paths, blocked descent, or non-Node ends.
            TypeError: If traversal controls are not bools.
        Side Effects:
            None. Roles, ownership, defaults and external state remain unchanged.
        """
        return _at(self, path, traverse_refs=traverse_refs, traverse_quoted=traverse_quoted)

    def bind(self, *, identities=None, states=None):
        """Compose rooted exact constraints, returning a new immutable graph.

        Args:
            identities (Mapping | IdentitySpec | None): Paths to exact ObjectIds.
            states (Mapping | StateSpec | None): Paths to exact LocalStates.
        Returns:
            A Node with equal assignments merged and complementary facets composed.
            An unchanged assignment returns this handle. All aliases are updated.
        Raises:
            TypeError: For nonmapping inputs, alternatives, or nonexact facets.
            ValueError: For conflicting assignments, including through aliases.
            GraphPathError: If a path does not select an admitted Node endpoint.
        Side Effects:
            None; assignment paths are resolved once before any immutable edit.
        """
        return _bind(self, identities, states, replace=False, realization=False)

    def replace_bindings(self, *, identities=None, states=None):
        """Deliberately replace/remove exact facets in a new graph.

        Args:
            identities (Mapping | None): Node paths to ObjectId or None (remove).
            states (Mapping | None): Node paths to LocalState or None (remove).
        Returns:
            An alias-coherent Node view; the original remains unchanged.
        Raises:
            TypeError: For invalid maps/facets.
            ValueError: If this edit itself assigns different values to aliases.
            GraphPathError: For invalid Node paths.
        Side Effects:
            None. Retained construction evidence on realization views is preserved.
        """
        return _bind(self, identities, states, replace=True, realization=False)

    def with_realization(self, *, identities=None, states=None):
        """Return a realization overlay retaining the original construction view.

        Args:
            identities (Mapping | None): Paths to allocated ObjectIds or None.
            states (Mapping | None): Paths to captured LocalStates or None.
        Returns:
            An immutable Node whose construction pins remain separately inspectable.
            Aliases outside quotations receive the overlay together; quoted values
            retain their exact facets and sharing, even for the same occurrence.
        Raises:
            TypeError, ValueError, GraphPathError: As for ``replace_bindings``.
        Side Effects:
            None. This records supplied metadata only; it does not capture state.
        """
        return _bind(self, identities, states, replace=True, realization=True)

    def with_identity(self, identity):
        """Compose one exact root ObjectId; return a Node or raise on conflict/type."""
        return self.bind(identities={GraphPath(): identity})

    def allocate_identities(self, *, check_unique=False, repo=None):
        """Allocate missing identities in a concrete graph before materialization.

        Assign one fresh ObjectId per distinct owned stateful occurrence, using
        the current ``object_namespace``. Preserve existing IDs, state selections,
        sharing and retained construction evidence. Reference-only targets and
        quoted recipes receive no IDs; non-quoted aliases to owned occurrences
        receive the same assignment. Stateless nodes receive no identity.

        Args:
            check_unique (bool): Default False keeps allocation metadata-only.
                True retries candidates already known to the Repo, present in this
                graph, or assigned earlier in this call. Existing IDs are preserved.
            repo (Repo | None): Authority for checked allocation. If omitted,
                use an explicitly active default Repo; never create one implicitly.
                Ignored when ``check_unique`` is False.
        Returns:
            ConcreteNode: An immutable realization overlay, or this same handle
                when no IDs are missing. Calling again on the result is a no-op;
                separate calls on an unbound source allocate independent IDs.
        Raises:
            TypeError: If the owned graph is not concrete, check_unique is not bool, or the
                selected authority is not a Repo. Call ``concretize()`` first.
            ValueError: If independent owned nodes share an ObjectId or a stateless
                node carries identity/state facets, or checked allocation has no
                Repo. Validation precedes allocation.
            RuntimeError: If 32 candidates for one missing ID collide or the Repo
                is closed/inherited. Store lookup failures propagate unchanged.
        Side Effects:
            Allocates UUIDs; checked mode also reads connected Store declaration
            metadata and Repo caches. Never resolves classes, constructs Objects,
            captures/removes state, publishes, or reserves execution ownership.
            Checks observe current evidence, not atomic cross-worker uniqueness.
        """
        from .definition import ConcreteNode
        from .reference_values import ObjectId

        if not isinstance(self, ConcreteNode):
            raise TypeError("Identity allocation requires a concrete Node; call concretize() first.")
        if type(check_unique) is not bool:
            raise TypeError("check_unique must be a bool.")
        if check_unique:
            from .repo import Repo, get_default_repo
            repo = get_default_repo() if repo is None else repo
            if repo is None:
                raise ValueError("check_unique=True requires an explicit or active Repo.")
            if not isinstance(repo, Repo):
                raise TypeError("Checked identity allocation requires a Repo.")
        missing, identities = [], set()
        for _, node in _owned_nodes(self):
            if not isinstance(node, ConcreteNode):
                raise TypeError("Identity allocation requires a concrete owned Node graph.")
            if not node._stateful_role:
                if node.object_id is not None or node.state is not None:
                    raise ValueError("Stateless Nodes cannot select an ObjectId or private state.")
            elif node.object_id is None:
                missing.append(node)
            else:
                if node.object_id in identities:
                    raise ValueError("Independent Nodes cannot share an ObjectId.")
                identities.add(node.object_id)
        if not missing:
            return self
        if check_unique:
            excluded = {node.object_id for node in _all_nodes(self, quoted=True) if node.object_id is not None}
            assigned = repo._allocate_unique_object_ids(len(missing), excluded=excluded)
        else:
            assigned = (ObjectId() for _ in missing)
        facets = {node._occurrence: _Facets(identity, node.state) for node, identity in zip(missing, assigned)}
        return _rewrite(self, facets=facets, realization=True, enter_quoted=False)

    def with_state(self, state):
        """Compose one exact root LocalState; return a Node or raise on conflict/type."""
        return self.bind(states={GraphPath(): state})

    def without_identity(self):
        """Return a new view without the root identity, preserving all aliases."""
        return self.replace_bindings(identities={GraphPath(): None})

    def without_state(self):
        """Return a new view without the root state, preserving retained construction."""
        return self.replace_bindings(states={GraphPath(): None})

    def identity_spec(self):
        """Return an IdentitySpec adapter over this root's materializing closure."""
        from .state_values import IdentitySpec
        return IdentitySpec(self)

    def state_spec(self):
        """Return a StateSpec adapter over owned selections, without identities."""
        from .state_values import StateSpec
        return StateSpec(self)

    def replace(self, path, replacement):
        """Replace a selected occurrence at every alias in a new immutable graph.

        Args:
            path (GraphPathLike): Root-relative Node endpoint to replace.
            replacement (Node): New subtree, retained as supplied.
        Returns:
            A graph with all aliases replaced; independent equal nodes are untouched.
        Raises:
            TypeError: If replacement is not a Node.
            GraphPathError: If path is not an admitted Node endpoint.
            ValueError: If the new composition contains contradictory alias facets.
        Side Effects:
            None. Splitting aliases requires a separate topology edit.
        """
        if not isinstance(replacement, Node):
            raise TypeError("Node replacement requires a Node.")
        selected = self.at(path)
        return _coalesce(_rewrite(self, {selected._occurrence: replacement}, reset_construction=True))

    def lens(self, path):
        """Return a structural copy-on-write editing lens for any value endpoint.

        Args:
            path (GraphPathLike): Root-relative value path, including scalars.
        Returns:
            A DefinitionLens whose ``set`` freezes its replacement.
        Raises:
            GraphPathError: On ``set`` if the path cannot be resolved.
        Side Effects:
            None. Unlike occurrence replacement, this edits one structural slot.
            Rebuilt Node ancestors preserve exact facets and invalidate retained
            construction evidence; state compatibility is not inferred.
        """
        from .definition import DefinitionLens
        return DefinitionLens(self, path)

    def copy_graph(self):
        """Copy a graph with fresh private occurrences and all facets retained.

        Returns:
            Node: Independently handled graph with the same alias partition,
            exact facets, roles, quotations and retained construction evidence.
        Side Effects:
            None. Does not allocate ObjectIds, encode, resolve targets or do IO.
        """
        return _rewrite(self, fresh=True)

    def recipe_equal(self, other):
        """Compare recipe values, ignoring facets and occurrence sharing.

        Args:
            other (object): Candidate graph/value.
        Returns:
            bool: Whether target and supplied constructor structures correspond.
        Side Effects:
            None; no class resolution or IO. Quoted configuration stays exact.
        """
        return _semantic_equal(self, other, topology=False, facets=False)

    def graph_equal(self, other):
        """Compare exact graph structure, roles, facets and alias partition.

        Args:
            other (object): Candidate graph/value.
        Returns:
            bool: Exact semantic equality, independent of private handle tokens.
        Raises:
            TypeError, ValueError: For unsupported semantic values or cycles.
        Side Effects:
            None; unordered incidence is matched without repr-based identity.
        """
        from .definition import ConcreteDefinition
        from .node_prepare import NodePreparation
        if not isinstance(other, (Node, ConcreteDefinition)):
            return False
        return NodePreparation((self,)).exact_equal(self, NodePreparation((other,)), other)

    def exact_key(self, *, prepared=None):
        """Return v1 canonical graph-exact certificate bytes, including facets.

        Args:
            prepared (NodePreparation | None): Optional operation-owned reusable
                preparation containing this immutable handle.
        Returns:
            bytes: ``dryml-node-exact/v1`` envelope, independent of scheduling,
            process tokens, enclosing roots and physical block placement.
        Raises:
            TypeError, ValueError: For unsupported values or a foreign preparation.
        Side Effects:
            Pure owned preparation only. At collision-sensitive authority boundaries
            compare exact evidence with ``NodePreparation.exact_equal`` as well.
        """
        from .node_prepare import NodePreparation
        return (NodePreparation((self,)) if prepared is None else prepared).exact_key(self)

    def exact_digest(self, *, prepared=None):
        """Return the domain-separated Node graph v1 digest of ``exact_key``.

        Accepts an optional owning NodePreparation, with the same errors and pure
        effects as ``exact_key``. Returns lowercase SHA-256 text. The legacy V2
        ``graph_hash`` adapter is retained for consumers awaiting their cutover.

        Args:
            prepared (NodePreparation | None): Optional owning pure preparation.
        Returns:
            str: Lowercase 64-hex Node graph digest.
        Raises:
            TypeError, ValueError: For unsupported values or a foreign preparation.
        Side Effects:
            Updates only owned immutable-preparation caches and work counts.
        """
        from .node_prepare import NodePreparation
        return (NodePreparation((self,)) if prepared is None else prepared).digest(self)

    def object_view(self):
        """Return a view with owned/reference state facets removed, identities kept.

        Returns:
            Node: Same alias partition and roles; quoted configuration stays exact.
        Raises:
            TypeError, ValueError: For unsupported graph values or malformed metadata.
        Side Effects:
            None. This projection neither allocates identities nor asserts availability.
        """
        from .node_views import object_projection
        return object_projection(self)

    def creation_view(self):
        """Return normalized retained creation evidence without own identity/state.

        Returns ConcreteNode; explicit dependency pins, topology, roles and quoted
        data remain exact. Raises TypeError for a nonconcrete root. No IO, target
        resolution, capture or allocation occurs.
        """
        from .node_views import creation_cdef
        return creation_cdef(self)

    def creation_digest(self):
        """Return the creation-view SHA-256 in ``dryml-creation-cdef/v1``.

        Returns str; nonconcrete roots raise TypeError. Updates only owned pure
        preparation facts retained within bounded handle ownership. Resident
        repeated requests reuse discovery/folds; eviction changes work only.
        This digest is an address, not an equality proof.
        """
        from .node_views import CREATION_VIEW, view_digest
        return view_digest(self, CREATION_VIEW)

    def compatibility_view(self):
        """Return concrete configuration with graph identity/state removed recursively.

        Returns ConcreteNode, preserving aliases, link roles and exact quotations.
        Raises TypeError for a nonconcrete root. No imports of targets or IO occur;
        compatibility still requires codec admission at the consuming boundary.
        """
        from .node_views import state_compatibility_view
        return state_compatibility_view(self)

    def compatibility_fingerprint(self):
        """Return a necessary compatibility key, never sufficient admission evidence.

        Returns str in ``dryml-state-compatibility/v1``. Nonconcrete roots raise
        TypeError; only bounded handle-owned pure preparation facts are retained.
        Resident requests reuse discovery/folds; eviction changes work only. Use
        LocalState's ``validate_for`` for creation-evidence, exact-graph and codec checks.
        """
        from .node_views import COMPATIBILITY_VIEW, view_digest
        return view_digest(self, COMPATIBILITY_VIEW)

    def object_projection_digest(self):
        """Return the state-weakened identity key in ``dryml-object-projection/v2``.

        Returns str, retaining required identities and exact quoted data. Performs
        no IO or resolution; unsupported values raise TypeError/ValueError. Digest
        equality alone is not proof; use ``object_projection_equal`` for authority.
        Reuses resident bounded handle-owned pure preparation without caching
        external evidence; eviction changes work only.
        """
        from .node_views import OBJECT_PROJECTION_VIEW, view_digest
        return view_digest(self, OBJECT_PROJECTION_VIEW)

    def object_projection_equal(self, other):
        """Compare exact state-weakened graphs, including identities, roles and aliases.

        Args:
            other (Node | ConcreteDefinition): Original supplied candidate graph.
        Returns:
            bool: Prefilter plus exact normalized graph comparison, quotation exact.
        Raises:
            TypeError, ValueError: For unsupported graph values or cycles.
        Side Effects:
            Reuses bounded handle-owned pure views/keys, comparing exact evidence
            each time. Never restores or consults a Store or caches live authority.
        """
        from .node_views import OBJECT_PROJECTION_VIEW, view_equal
        return view_equal(self, other, OBJECT_PROJECTION_VIEW)

    def adopt_saved_view(self, supplied):
        """Validate original saved projection and adopt its construction pins as metadata.

        Args:
            supplied (ConcreteNode): Complete saved graph with retained construction.
        Returns:
            ConcreteNode: Supplied state/pins over receiver occurrence identities
            and sharing. Later creation inspection uses the adopted construction.
        Raises:
            TypeError: For nonconcrete graphs or unsupported values.
            ValueError: For incomplete saved facets or an original projection mismatch.
        Side Effects:
            None. This is not runtime restore or payload/codec admission. Consumers
            must validate selected LocalStates before any hooks or live mutation.
        """
        from .node_views import adopt_saved_view
        return adopt_saved_view(self, supplied)

    def quote(self):
        """Return this Node as exact inert constructor data in a QuotedDef.

        Returns:
            QuotedDef: A non-owning quotation retaining this immutable handle.
        Raises:
            TypeError: If the underlying construction value is not quotable.
        Side Effects:
            None; quotation does not resolve, normalize or materialize the Node.
        """
        from .quoted import QuotedDef
        return QuotedDef(self)


def _unwrap(value, *, descending, traverse_refs, traverse_quoted):
    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .quoted import QuotedDef

    while isinstance(value, (DefLink, QuotedDef)):
        if isinstance(value, DefLink):
            if descending and value.kind is EdgeKind.REF and not traverse_refs:
                raise GraphPathError("Node path descent crosses a reference boundary; enable traverse_refs.")
            value = value.target
        else:
            if descending and not traverse_quoted:
                raise GraphPathError("Node path descent crosses a quotation boundary; enable traverse_quoted.")
            value = value.value
    return value


def _at(root, path, *, traverse_refs=False, traverse_quoted=False):
    from .utils.graph.value import _get_child

    if type(traverse_refs) is not bool or type(traverse_quoted) is not bool:
        raise TypeError("Node traversal controls must be bools.")
    current = root
    for segment in normalize_path(path):
        current = _unwrap(current, descending=True, traverse_refs=traverse_refs, traverse_quoted=traverse_quoted)
        try:
            current = _get_child(current, segment)
        except Exception as error:
            raise GraphPathError(f"Cannot resolve Node path segment {segment!s}.") from error
    current = _unwrap(current, descending=False, traverse_refs=traverse_refs, traverse_quoted=traverse_quoted)
    if not isinstance(current, Node):
        raise GraphPathError(f"Expected a Node endpoint, found {type(current).__name__}.")
    return current


def _all_nodes(root, *, quoted=False):
    """Visit graph Nodes, treating exact quoted configuration as a separate scope."""
    from .links import DefLink
    from .quoted import QuotedDef
    from .utils.graph.value import iter_value_edges

    seen = set()

    def visit(value):
        if isinstance(value, DefLink):
            yield from visit(value.target)
        elif isinstance(value, QuotedDef):
            if quoted:
                yield from visit(value.value)
        elif id(value) not in seen:
            seen.add(id(value))
            if isinstance(value, Node):
                yield value
            for edge in iter_value_edges(value):
                yield from visit(edge.value)
    return visit(root)


def _owned_nodes(root):
    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .quoted import QuotedDef
    from .utils.graph.value import iter_value_edges
    from .utils.graph.path import graph_path_sort_key

    seen = set()

    def visit(value, path):
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                yield from visit(value.target, path)
            return
        if isinstance(value, QuotedDef):
            return
        if isinstance(value, Node):
            if value._occurrence in seen:
                return
            seen.add(value._occurrence)
            yield path, value
        for edge in sorted(iter_value_edges(value), key=lambda edge: graph_path_sort_key(path.child(edge.segment))):
            yield from visit(edge.value, path.child(edge.segment))
    return visit(root, GraphPath())


def _bind(root, identities, states, *, replace, realization):
    from .reference_values import ObjectId
    from .state_values import LocalState

    changes = {}
    for mapping, slot, expected in ((identities, 0, ObjectId), (states, 1, LocalState)):
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise TypeError("Node bindings must be rooted mappings or facet specifications.")
        for path, value in mapping.items():
            if not isinstance(value, expected) and not (replace and value is None):
                raise TypeError(f"Node bindings require exact {expected.__name__} values.")
            selected = root.at(path)
            key = selected._occurrence
            assignments = changes.setdefault(key, {})
            if slot in assignments and not _facet_equal(assignments[slot], value):
                raise ValueError("Node binding conflict through aliased paths.")
            old = (selected.object_id, selected.state)[slot]
            if not replace and old is not None and not _facet_equal(old, value):
                raise ValueError("Node binding conflict with an existing exact assignment.")
            assignments[slot] = value
    facets = {}
    for node in _all_nodes(root, quoted=not realization):
        assignments = changes.get(node._occurrence)
        if assignments is not None:
            new = _Facets(assignments.get(0, node.object_id), assignments.get(1, node.state))
            if not _facets_equal(new, node._facets):
                facets[node._occurrence] = new
    return _rewrite(root, facets=facets, realization=realization, enter_quoted=not realization)


def _facet_equal(left, right):
    return left == right


def _facets_equal(left, right):
    return left.object_id == right.object_id and _facet_equal(left.state, right.state)


def _clone(node, args, kwargs, facets, *, realization=False, reset_construction=False):
    from .definition import ConcreteDefinition, Definition
    from .bound_args import BoundArguments

    result = object.__new__(type(node))
    if isinstance(node, ConcreteDefinition):
        # Structural records stay in the existing CDef representation. A soft
        # replacement invalidates concreteness rather than invoking normalization.
        from .utils.graph.value import _contains_soft_definition
        if any(_contains_soft_definition(v) for v in kwargs.values()):
            result = Definition._from_prepared_parameters(node.cls, kwargs)
        else:
            result._copy_record(node)
            object.__setattr__(result, "_bound_args", BoundArguments(kwargs))
    else:
        object.__setattr__(result, "_cls", node.cls)
        object.__setattr__(result, "_args", args)
        object.__setattr__(result, "_kwargs", kwargs)
    _init_node(result, source=node)
    object.__setattr__(result, "_facets", facets)
    object.__setattr__(result, "_stable_hash_cache", None)
    if isinstance(result, ConcreteDefinition):
        object.__setattr__(result, "_graph_hash_cache", None)
    if realization:
        object.__setattr__(result, "_construction", node.construction_view)
    elif reset_construction:
        object.__setattr__(result, "_construction", None)
    return result


def _rewrite(root, replacements=None, *, facets=None, realization=False, enter_quoted=True,
             representatives=None, reset_construction=False, fresh=False,
             node_transform=None, value_transform=None, memo=None):
    from .definition import ConcreteDefinition
    from .links import DefLink
    from .quoted import QuotedDef

    replacements, facets = replacements or {}, facets or {}
    if memo is None:
        memo = {}

    def visit(value):
        if representatives is not None and isinstance(value, Node):
            value = representatives[value._occurrence]
            marker = value._occurrence
        else:
            marker = id(value)
        if marker in memo:
            return memo[marker]
        if isinstance(value, Node):
            if value._occurrence in replacements:
                return replacements[value._occurrence]
            if isinstance(value, ConcreteDefinition):
                old_args, old_kwargs = None, value.parameters
            else:
                old_args, old_kwargs = value.args, value.kwargs
            args = None if old_args is None else FrozenTuple(visit(v) for v in old_args)
            kwargs = FrozenDict((k, visit(v)) for k, v in old_kwargs.items())
            new_facets = facets.get(value._occurrence, value._facets)
            unchanged = (old_args is None or all(a is b for a, b in zip(args, old_args))) and all(kwargs[k] is v for k, v in old_kwargs.items())
            result = value if not fresh and unchanged and _facets_equal(new_facets, value._facets) else _clone(
                value, args, kwargs, new_facets, realization=realization,
                reset_construction=reset_construction)
            if fresh:
                token = object()
                object.__setattr__(result, "_occurrence", token)
                if isinstance(result, ConcreteDefinition):
                    object.__setattr__(result, "_node_id", token)
            if node_transform is not None:
                result = node_transform(value, result)
                if not fresh and not reset_construction:
                    result = _reuse_node_view(value, result)
        elif value_transform is not None and (mapped := value_transform(value, visit)) is not NotImplemented:
            result = mapped
        elif isinstance(value, DefLink):
            target = visit(value.target)
            result = value if target is value.target else (DefLink.finalized(value.kind, target) if value.is_finalized else DefLink.assertion(value.kind, target))
        elif isinstance(value, QuotedDef) and enter_quoted:
            target = visit(value.value)
            result = value if target is value.value else QuotedDef(target)
        elif isinstance(value, (FrozenDict, dict)):
            mapped = [(k, visit(v)) for k, v in value.items()]
            result = value if all(v is value[k] for k, v in mapped) else type(value)(mapped)
        elif isinstance(value, (tuple, list, set, frozenset)):
            mapped = [visit(v) for v in value]
            result = value if all(a is b for a, b in zip(mapped, value)) else type(value)(mapped)
        else:
            result = value
        memo[marker] = result
        return result
    return visit(root)


def _reuse_node_view(original, node):
    """Recover original aliases after no-op transforms or exact overlay removal.

    Reusing the retained handle preserves sharing with inert quoted data; equal
    independent occurrences and deliberately different facet views stay distinct.
    """
    from .definition import ConcreteDefinition

    for source in (node._construction, original):
        if (source is None or type(source) is not type(node)
                or source._occurrence is not node._occurrence
                or source.cls != node.cls or not _facets_equal(source._facets, node._facets)
                or source is original and source._construction is not node._construction):
            continue
        if isinstance(node, ConcreteDefinition):
            before, after = source.parameters, node.parameters
        else:
            if ((source.args is None) != (node.args is None)
                    or source.args is not None and (len(source.args) != len(node.args)
                        or any(a is not b for a, b in zip(source.args, node.args)))):
                continue
            before, after = source.kwargs, node.kwargs
        if before.keys() == after.keys() and all(before[k] is v for k, v in after.items()):
            return source
    return node


def _coalesce(root):
    """Compose multiple immutable views of one occurrence on authoring entry."""
    representatives, merged, duplicates = {}, {}, False
    for node in _all_nodes(root, quoted=False):
        key = node._occurrence
        if key not in representatives:
            representatives[key], merged[key] = node, node._facets
            continue
        duplicates = True
        previous = representatives[key]
        if not _semantic_equal(previous, node, topology=True, facets=False):
            raise ValueError("Node construction-view conflict for one occurrence.")
        values = []
        for left, right in zip((merged[key].object_id, merged[key].state), (node.object_id, node.state)):
            if left is not None and right is not None and not _facet_equal(left, right):
                raise ValueError("Node binding conflict between embedded occurrence views.")
            values.append(right if left is None else left)
        merged[key] = _Facets(*values)
    if not duplicates:
        return root
    return _rewrite(root, facets=merged, representatives=representatives, enter_quoted=False)


def _semantic_equal(left, right, *, topology, facets):
    from .definition import ConcreteDefinition, _structural_value_equal
    from .links import DefLink
    from .quoted import QuotedDef

    def sequence(pairs, forward, reverse, index=0):
        if index == len(pairs):
            yield forward, reverse
            return
        for nf, nr in compare(*pairs[index], forward, reverse):
            yield from sequence(pairs, nf, nr, index + 1)

    def compare(a, b, forward, reverse):
        if isinstance(a, (Node, ConcreteDefinition)) or isinstance(b, (Node, ConcreteDefinition)):
            if not isinstance(a, (Node, ConcreteDefinition)) or not isinstance(b, (Node, ConcreteDefinition)):
                return
            ak = a._occurrence if isinstance(a, Node) else a._node_id
            bk = b._occurrence if isinstance(b, Node) else b._node_id
            if topology:
                if ak in forward:
                    if forward[ak] is bk:
                        yield forward, reverse
                    return
                if bk in reverse:
                    return
                forward, reverse = dict(forward), dict(reverse)
                forward[ak], reverse[bk] = bk, ak
            if not _structural_value_equal(a.cls, b.cls):
                return
            if facets and not _facets_equal(a._facets if isinstance(a, Node) else _Facets(), b._facets if isinstance(b, Node) else _Facets()):
                return
            if isinstance(a, ConcreteDefinition) != isinstance(b, ConcreteDefinition):
                return
            if isinstance(a, ConcreteDefinition):
                yield from compare(a.parameters, b.parameters, forward, reverse)
            else:
                yield from sequence([(a.args, b.args), (a.kwargs, b.kwargs)], forward, reverse)
            return
        if isinstance(a, DefLink) or isinstance(b, DefLink):
            if isinstance(a, DefLink) and isinstance(b, DefLink) and a.kind is b.kind:
                yield from compare(a.target, b.target, forward, reverse)
            return
        if isinstance(a, QuotedDef) or isinstance(b, QuotedDef):
            if isinstance(a, QuotedDef) and isinstance(b, QuotedDef) and _semantic_equal(a.value, b.value, topology=True, facets=True):
                yield forward, reverse
            return
        if type(a) is not type(b):
            return
        if isinstance(a, Mapping):
            if a.keys() == b.keys():
                yield from sequence([(a[k], b[k]) for k in a], forward, reverse)
            return
        if isinstance(a, (tuple, list)):
            if len(a) == len(b):
                yield from sequence(list(zip(a, b)), forward, reverse)
            return
        if isinstance(a, (set, frozenset)):
            if len(a) != len(b):
                return
            def match(todo, candidates, fwd, rev):
                if not todo:
                    yield fwd, rev
                    return
                for index, candidate in enumerate(candidates):
                    for nf, nr in compare(todo[0], candidate, fwd, rev):
                        yield from match(todo[1:], candidates[:index] + candidates[index+1:], nf, nr)
            yield from match(list(a), list(b), forward, reverse)
            return
        if _structural_value_equal(a, b):
            yield forward, reverse
    return next(compare(left, right, {}, {}), None) is not None


def require_saved(node):
    """Validate saved completeness and return the same immutable graph handle.

    Args:
        node (ConcreteNode): Normalized graph to validate.
    Returns:
        SavedNode: The original handle, with exact identity/state on every owned
        stateful occurrence. Ephemeral nodes need no own identity/private state.
    Raises:
        TypeError: If the owned graph is not concrete.
        ValueError: If required facets are missing or independent identities collide.
    Side Effects:
        None. Completeness asserts neither availability nor current-use compatibility.
    """
    from .definition import ConcreteNode

    if not isinstance(node, ConcreteNode):
        raise TypeError("Saved completeness requires a ConcreteNode graph.")
    identities = set()
    for _, current in _owned_nodes(node):
        if not isinstance(current, ConcreteNode):
            raise TypeError("Saved completeness requires a ConcreteNode graph.")
        if current._stateful_role:
            if current.object_id is None or current.state is None:
                raise ValueError("Saved completeness requires identity and state for every owned stateful Node.")
            if current.object_id in identities:
                raise ValueError("Independent saved Nodes cannot share an ObjectId.")
            identities.add(current.object_id)
    return node


def concretize_node(node, repo=None):
    """Normalize a Node with the existing construction pipeline and Node outputs.

    Args:
        node (Node): Soft construction graph.
        repo (Repo | metadata resolver | None): Optional repository or object
            exposing ``resolve_state_selector(selector)`` for soft state leaves.
    Returns:
        ConcreteNode: Fully normalized graph with supplied occurrence facets retained.
    Raises:
        Exception: Propagates constructor-binding, canonical-admission and metadata
            resolution failures, including missing aliases and out-of-scope states.
    Side Effects:
        May resolve targets and run preparation hooks; never constructs Objects,
        allocates ObjectIds, reads private payloads, or publishes data.
    """
    from .canonical import _ToCanonicalTransformer
    from .definition import ConcreteDefinition, ConcreteNode, Definition
    from .repo import manage_repo
    from .utils.graph import GraphCtx

    class NodeOutputs(_ToCanonicalTransformer):
        def transform_atomic(self, obj, ctx):
            return _as_node_view(obj, converted)

        def dispatch(self, obj, ctx):
            result = super().dispatch(obj, ctx)
            if isinstance(obj, Definition) and isinstance(result, ConcreteDefinition):
                view = object.__new__(ConcreteNode)
                view._copy_record(result)
                _init_node(view, source=obj)
                object.__setattr__(view, "_node_id", obj._occurrence)
                return view
            return result

    converted = {}
    if repo is not None and callable(getattr(repo, "resolve_state_selector", None)):
        return NodeOutputs().transform(node, GraphCtx(state={"repo": repo}))
    with manage_repo(repo=repo) as selected:
        return NodeOutputs().transform(node, GraphCtx(state={"repo": selected}))


def __getattr__(name):
    if name in {"ConcreteNode", "SavedNode"}:
        from . import definition
        return getattr(definition, name)
    raise AttributeError(name)
