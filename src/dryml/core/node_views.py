"""Named creation, compatibility and operation views over the shared Node carrier.

All transformations are metadata-only. They preserve typed roles and aliases,
never enter quotations, and use NodePreparation's collision-independent verifier.
Payload availability and runtime application belong to the consuming operation.
"""

from .node_prepare import NodePreparation, PreparationView


def _concrete(root):
    from .definition import ConcreteDefinition
    from .node import _as_node_view

    if not isinstance(root, ConcreteDefinition):
        raise TypeError("This view requires a concrete normalized construction root.")
    return _as_node_view(root)


def creation_cdef(root):
    """Return retained normalized construction authority without root realization facets.

    Args:
        root (ConcreteNode | ConcreteDefinition): Captured or construction view.
    Returns:
        ConcreteNode: Root identity/state excluded; explicit dependency selections,
        topology, roles and quoted constructor data retained exactly.
    Raises:
        TypeError: If root or retained construction evidence is not concrete.
    Side Effects:
        None. Captured states and allocated IDs never become construction pins.
    """
    from .node import _clone, _Facets

    source = _concrete(_concrete(root).construction_view)
    if source.object_id is None and source.state is None and source._construction is None:
        return source
    return _clone(source, None, source.parameters, _Facets(), reset_construction=True)


def _project(root, *, compatibility):
    from .config import ConfigRef
    from .definition import ConcreteDefinition
    from .factory import FactorySpec
    from .node import _as_node_view, _clone, _Facets, _rewrite
    from .reference_values import ObjectRef, StateRef, _expected_paths
    from .node_codec import _reference_bindings
    from .template import _BinaryExpr, _RepeatExpr

    def node(source, result):
        facets = _Facets(None if compatibility else source.object_id, None)
        if result._facets == facets:
            return result
        concrete = isinstance(result, ConcreteDefinition)
        return _clone(result, None if concrete else result.args,
                      result.parameters if concrete else result.kwargs, facets)

    def value(current, visit):
        if isinstance(current, StateRef):
            return visit(current.object)
        if isinstance(current, ObjectRef):
            definition = visit(_as_node_view(current.definition, adapted))
            if compatibility:
                return definition
            # Rebase target associations, not context-local unordered path ordinals.
            expected, _ = _expected_paths(definition)
            objects = {}
            for target, identity in _reference_bindings(current):
                key = (("object-id", identity) if target is None else
                       ("cdef", visit(_as_node_view(target, adapted))._node_id))
                objects[expected[key]] = identity
            return ObjectRef(definition, objects)
        if isinstance(current, FactorySpec):
            return FactorySpec._from_symbolic_parts(current.target, visit(current.args), visit(current.kwargs))
        if isinstance(current, ConfigRef) and current.has_default:
            return ConfigRef(current.key, visit(current.default))
        if isinstance(current, _BinaryExpr):
            return _BinaryExpr(current.operation, visit(current.left), visit(current.right))
        if isinstance(current, _RepeatExpr):
            return _RepeatExpr(visit(current.group), visit(current.count), current.shared)
        return NotImplemented

    adapted = {}
    return _rewrite(_as_node_view(root, adapted), enter_quoted=False,
                    node_transform=node, value_transform=value)


def state_compatibility_view(root):
    """Strip realization identity/state recursively, preserving exact configuration.

    Args:
        root (ConcreteNode | ConcreteDefinition): Destination or origin evidence.
    Returns:
        ConcreteNode: Class/configuration, topology/sharing and link-role evidence;
        legacy exact graph references are weakened to their construction graphs.
    Raises:
        TypeError, ValueError: For nonconcrete roots or unsupported graph values.
    Side Effects:
        None. Quoted data is not weakened; this does not admit a codec or payload.
    """
    return _project(_concrete(root), compatibility=True)


def object_projection(root):
    """Remove graph state selections, retaining identities and exact quotations.

    Args:
        root (Node | ConcreteDefinition): Soft or normalized supplied graph.
    Returns:
        Node: Alias-preserving view, including recursively weakened legacy StateRefs.
    Raises:
        TypeError, ValueError: For unsupported graph values.
    Side Effects:
        None. No query traversal option changes operation identity's quotation rule.
    """
    from .definition import ConcreteDefinition
    from .node import Node
    if not isinstance(root, (Node, ConcreteDefinition)):
        raise TypeError("Object projection requires a Node or ConcreteDefinition.")
    return _project(root, compatibility=False)


CREATION_VIEW = PreparationView("dryml-creation-cdef/v1", creation_cdef)
COMPATIBILITY_VIEW = PreparationView("dryml-state-compatibility/v1", state_compatibility_view)
OBJECT_PROJECTION_VIEW = PreparationView("dryml-object-projection/v2", object_projection)


def view_digest(root, view):
    """Return a named view's domain-separated digest using owned shared preparation.

    Args:
        root (Node | ConcreteDefinition): Source graph.
        view (PreparationView): Named pure policy; its name is also the hash domain.
    Returns:
        str: Necessary candidate/address key, not an equality proof.
    Raises:
        TypeError, ValueError: On invalid policy, values or graph structure.
    Side Effects:
        Computes only operation-owned pure facts, without IO or resolution.
    """
    prepared = NodePreparation((root,), view=view)
    return prepared.digest(prepared.roots[0], domain=view.name)


def view_equal(left, right, view):
    """Verify a named view by necessary key followed by exact normalized evidence.

    Args:
        left, right (Node | ConcreteDefinition): Original candidate roots.
        view (PreparationView): Pure named transformation shared by both candidates.
    Returns:
        bool: Graph-aware equality, including unordered aliases even on hash ties.
    Raises:
        TypeError, ValueError: On invalid roots, policy or graph structure.
    Side Effects:
        Only owned preparation work; no Store, hook or arbitrary compatibility call.
    """
    a, b = NodePreparation((left,), view=view), NodePreparation((right,), view=view)
    x, y = a.roots[0], b.roots[0]
    return a.exact_key(x) == b.exact_key(y) and a.exact_equal(x, b, y)


def adopt_saved_view(receiver, supplied):
    """Check the original supplied projection before adopting retained construction.

    Args:
        receiver (ConcreteNode): Existing metadata with required live identities.
        supplied (ConcreteNode): Complete saved view to adopt, including its pins.
    Returns:
        ConcreteNode: Adopted metadata with receiver occurrence tokens and alias
        partition; construction evidence comes from the supplied graph.
    Raises:
        TypeError: For nonconcrete or unsupported metadata.
        ValueError: For incomplete saved metadata or original projection mismatch.
    Side Effects:
        None. This performs neither codec/payload admission nor runtime restoration.
    """
    from .node import Node, _clone, _rewrite, require_saved

    receiver, supplied = _concrete(receiver), require_saved(_concrete(supplied))
    a = NodePreparation((receiver,), view=OBJECT_PROJECTION_VIEW)
    b = NodePreparation((supplied,), view=OBJECT_PROJECTION_VIEW)
    x, y = a.roots[0], b.roots[0]
    if a.exact_key(x) != b.exact_key(y) or not a.exact_equal(x, b, y):
        raise ValueError("The original supplied graph does not match the receiver object projection.")
    # The exact canonical labeling already accounts for unordered symmetry and
    # cross-branch sharing. Its matched positions rebase private occurrence tokens.
    _, left = a._flat(a._slot(x), retain=True)
    _, right = b._flat(b._slot(y), retain=True)
    token_of = lambda node: node._occurrence if isinstance(node, Node) else node._node_id
    tokens = {token_of(b.nodes[j]): token_of(a.nodes[i])
              for i, j in zip(left[0], right[0])}

    construction_memo = {}

    def rebase_construction(root):
        return _rewrite(root, enter_quoted=False, node_transform=retoken,
                        memo=construction_memo)

    def retoken(source, result):
        result = _clone(result, None, result.parameters, result._facets)
        token = tokens.get(source._occurrence, source._occurrence)
        object.__setattr__(result, "_occurrence", token)
        object.__setattr__(result, "_node_id", token)
        if source._construction is not None:
            object.__setattr__(result, "_construction", rebase_construction(source._construction))
        return result

    def adopt(source, result):
        result = retoken(source, result)
        object.__setattr__(result, "_construction", rebase_construction(source.construction_view))
        return result

    return _rewrite(supplied, enter_quoted=False, node_transform=adopt)
