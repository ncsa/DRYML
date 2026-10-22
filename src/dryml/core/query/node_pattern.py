"""Node-pattern interpretation over the shared metadata preparation authority.

This semantic adapter provides no Store inventory, source cut or index-coverage
claim. Legacy posting plans retain it as an exact residual and must visibly scan
or refuse until the format-4 source/lowering consumer is available.
"""

from collections.abc import Mapping

from ..definition import ConcreteDefinition
from ..node import Node
from ..node_prepare import NodePreparation
from ..selector import QueryPredicate
from .model import QueryDomainError


def _runtime_pattern_view(root):
    """Retain runtime matcher code outside canonical pure graph facts."""
    from ..node import _rewrite
    from ..params import (Match, AnyValue, AnyMatcher, ChoiceMatcher, ExactMatcher,
                          IntRangeMatcher, MissingMatcher, PresentMatcher, SubclassMatcher)
    from ..factory import FactorySpec
    supported = (AnyMatcher, ChoiceMatcher, ExactMatcher, IntRangeMatcher,
                 MissingMatcher, PresentMatcher, SubclassMatcher)
    def transform(value, visit):
        if isinstance(value, Match) and type(value.matcher) not in supported:
            return AnyValue()
        if isinstance(value, FactorySpec):
            return FactorySpec._from_symbolic_parts(value.target, tuple(visit(item) for item in value.args),
                                                    {name: visit(item) for name, item in value.kwargs.items()})
        return NotImplemented
    return _rewrite(root, value_transform=transform)


def _parameters(node):
    """Read supplied names, retaining structural absence leaves on live classes."""
    try:
        return node.parameters
    except TypeError:
        from inspect import signature
        from ..bound_args import bind_partial_arguments
        if not isinstance(node.cls, type):
            raise
        names = signature(node.cls.__init__).parameters
        known = {key: value for key, value in node.kwargs.items() if key in names}
        unknown = {key: value for key, value in node.kwargs.items() if key not in names}
        result = dict(bind_partial_arguments(node.cls, node.args or (), known).items())
        result.update(unknown)
        return result


class _PreparedPredicate:
    """Operation-owned semantic facts; mutable witness state is per match only."""

    def __init__(self, predicate, *, domain="node", class_match="selector", strict=False):
        if domain not in {"node", "local_state", "occurrence"}:
            raise QueryDomainError("Unsupported Node query domain.")
        self.predicate, self.domain = predicate, domain
        self.class_match, self.strict = class_match, strict
        roots, seen = [], set()
        def collect(current):
            if id(current) in seen:
                return
            seen.add(id(current))
            if current.kind == "node":
                roots.append(current.value)
            elif current.kind == "support":
                roots.extend((current.value._generator.definition, current.value.prefilter.root))
            for operand in current.operands:
                collect(operand)
        collect(predicate)
        from ..selector import _contains_state_selector
        if any(_contains_state_selector(root) for root in roots):
            raise TypeError("Node query predicates require explicitly resolved state aliases.")
        from ..node_prepare import PreparationView
        self.prepared = NodePreparation(roots, view=PreparationView("query-pattern", _runtime_pattern_view))
        self.parameters = {}
        if domain == "local_state":
            def validate(current):
                if current.kind == "absent" and current.value[0] == "identity":
                    raise QueryDomainError("LocalState has no donor identity; request an explicit occurrence relationship.")
                if current.kind in {"node", "support"}:
                    root = current.value if current.kind == "node" else current.value._generator.definition
                    from ..node import _all_nodes
                    if any(node.object_id is not None for node in _all_nodes(root, quoted=False)):
                        raise QueryDomainError("LocalState has no donor identity; request an explicit occurrence relationship.")
                for operand in current.operands:
                    validate(operand)
            validate(predicate)

    def matches(self, target, *, generator_budget=None, prepared=None):
        from ..state_values import LocalState
        if self.domain == "local_state":
            if not isinstance(target, LocalState):
                raise QueryDomainError("Local-state matching requires a LocalState descriptor.")
            from ..node import _as_node_view
            target = _as_node_view(target.creation_cdef).with_state(target)
        elif not isinstance(target, (Node, ConcreteDefinition)):
            return False
        elif not isinstance(target, Node):
            from ..node import _as_node_view
            target = _as_node_view(target)
        if prepared is None:
            NodePreparation((target,), view=self.prepared.view)
        else:
            prepared._slot(target)
        candidate_parameters = {}

        def evaluate(current):
            if current.kind == "node":
                return _node_matches(current.value, target, self.parameters, candidate_parameters,
                                     class_match=self.class_match, strict=self.strict)
            if current.kind == "absent":
                from ..utils.graph.path import GraphPathError
                try:
                    node = target.at(current.value[1], traverse_refs=True, traverse_quoted=True)
                except GraphPathError:
                    return False
                return getattr(node, "object_id" if current.value[0] == "identity" else "state") is None
            if current.kind == "support":
                if generator_budget is not None:
                    generator_budget.consume()
                selector = current.value
                if not _node_matches(selector.prefilter.root, target, self.parameters, candidate_parameters,
                                     class_match="exact", strict=False):
                    return False
                return selector.matches(target)
            # Evaluate every branch: a true operand must not hide an unsupported
            # exact support proof or an invalid later operand.
            values = tuple(evaluate(operand) for operand in current.operands)
            if current.kind == "and":
                return all(values)
            if current.kind == "or":
                return any(values)
            return not values[0]
        return evaluate(self.predicate)


def _node_matches(pattern, target, pattern_parameters=None, target_parameters=None,
                  *, class_match="selector", strict=False):
    """Backtrack whole assignments, retaining a bijection across unordered joins."""
    from ..canonical import container_family
    from ..factory import FactorySpec
    from ..links import DefLink
    from ..params import Match
    from ..quoted import QuotedDef, SelectorSpec
    from .query import _query_match_class, _query_match_leaf

    pattern_parameters = {} if pattern_parameters is None else pattern_parameters
    target_parameters = {} if target_parameters is None else target_parameters

    def parameters(node, cache, peer):
        key = (id(node), peer.cls if isinstance(peer.cls, type) else None)
        if key not in cache:
            try:
                cache[key] = _parameters(node)
            except TypeError:
                # A quoted Node may also be used as an ordinary pattern. Preserve
                # its raw portable call, borrowing only an already supplied live
                # declaration of the exact same class to name positional fields.
                from ..bound_args import bind_partial_arguments
                from ..symbol import maybe_symbol_ref
                if (isinstance(node, ConcreteDefinition) or not isinstance(peer.cls, type)
                        or maybe_symbol_ref(node.cls) != maybe_symbol_ref(peer.cls)):
                    raise QueryDomainError("Positional Node constraints need available supplied parameter names.") from None
                cache[key] = bind_partial_arguments(peer.cls, node.args or (), node.kwargs).as_frozen_dict()
        return cache[key]

    def sequence(pairs, forward, reverse, index=0):
        if index == len(pairs):
            yield forward, reverse
            return
        left, right, exact = pairs[index]
        for nf, nr in compare(left, right, forward, reverse, exact):
            yield from sequence(pairs, nf, nr, index + 1)

    def compare(left, right, forward, reverse, exact=0):
        # Strength 0 means supplied pattern constraints, 1 exact concrete recipe,
        # 2 exact quotation (including missing facets and raw omission spelling).
        # An alias revisited through quotation must upgrade its earlier proof.
        if isinstance(left, Match) and not exact:
            if left.matches(right):
                yield forward, reverse
            return
        while exact != 2 and isinstance(left, DefLink) and left.kind.value == "materialize":
            left = left.target
        while exact != 2 and isinstance(right, DefLink) and right.kind.value == "materialize":
            right = right.target
        if isinstance(left, (Node, ConcreteDefinition)):
            if not isinstance(right, (Node, ConcreteDefinition)):
                return
            if isinstance(left, ConcreteDefinition) and not isinstance(right, ConcreteDefinition):
                return
            if exact == 2 and isinstance(left, ConcreteDefinition) != isinstance(right, ConcreteDefinition):
                return
            lk = left._view if isinstance(left, Node) else left._node_id
            rk = right._view if isinstance(right, Node) else right._node_id
            if lk in forward:
                previous, strength = forward[lk]
                if previous is not rk:
                    return
                if strength >= exact:
                    yield forward, reverse
                    return
            if rk in reverse and reverse[rk] is not lk:
                return
            anchor = exact or isinstance(left, ConcreteDefinition)
            if left.cls is not None or anchor:
                if not _query_match_class(left.cls, right.cls, strict=strict or anchor,
                                          class_match="exact" if anchor else class_match):
                    return
            for facet in ("object_id", "state"):
                a, b = getattr(left, facet, None), getattr(right, facet, None)
                if (a is not None or exact == 2) and a != b:
                    return
            if exact == 2 and not isinstance(left, ConcreteDefinition):
                nf, nr = dict(forward), dict(reverse)
                nf[lk], nr[rk] = (rk, exact), lk
                yield from sequence([(left.args, right.args, 2), (left.kwargs, right.kwargs, 2)], nf, nr)
                return
            lp, rp = parameters(left, pattern_parameters, right), parameters(right, target_parameters, left)
            if anchor and set(lp) != set(rp):
                return
            pairs = []
            for name, value in lp.items():
                if name not in rp:
                    if not exact and isinstance(value, Match) and value.matches(None, present=False):
                        continue
                    return
                pairs.append((value, rp[name], 2 if exact == 2 else int(anchor)))
            nf, nr = dict(forward), dict(reverse)
            nf[lk], nr[rk] = (rk, exact), lk
            yield from sequence(pairs, nf, nr)
            return
        if isinstance(left, DefLink):
            if isinstance(right, DefLink) and left.kind is right.kind:
                yield from compare(left.target, right.target, forward, reverse, exact)
            return
        if isinstance(left, QuotedDef):
            if isinstance(right, QuotedDef):
                yield from compare(left.value, right.value, forward, reverse, 2)
            return
        if isinstance(left, SelectorSpec):
            if (isinstance(right, SelectorSpec) and left.selector.strict == right.selector.strict
                    and left.selector.cls_policy == right.selector.cls_policy):
                yield from compare(left.selector.root, right.selector.root, forward, reverse, 2)
            return
        if isinstance(right, (DefLink, QuotedDef)):
            return
        if isinstance(left, Match):
            from ..definition import _structural_value_equal
            accepted = _structural_value_equal(left, right) if exact else left.matches(right)
            if accepted:
                yield forward, reverse
            return
        if isinstance(left, FactorySpec):
            if not isinstance(right, FactorySpec):
                return
            yield from sequence([(left.target, right.target, True), (left.args, right.args, exact),
                                 (left.kwargs, right.kwargs, exact)], forward, reverse)
            return
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or exact and set(left) != set(right):
                return
            pairs = []
            for key, value in left.items():
                if key not in right:
                    if not exact and isinstance(value, Match) and value.matches(None, present=False):
                        continue
                    return
                pairs.append((value, right[key], exact))
            yield from sequence(pairs, forward, reverse)
            return
        family = container_family(left)
        if family is not None:
            if container_family(right) != family or len(left) != len(right):
                return
            if family in {"set", "frozenset"}:
                def assign(todo, remaining, fwd, rev):
                    if not todo:
                        yield fwd, rev
                        return
                    for index, value in enumerate(remaining):
                        for nf, nr in compare(todo[0], value, fwd, rev, exact):
                            yield from assign(todo[1:], remaining[:index] + remaining[index+1:], nf, nr)
                yield from assign(list(left), list(right), forward, reverse)
            else:
                yield from sequence([(a, b, exact) for a, b in zip(left, right)], forward, reverse)
            return
        if _query_match_leaf(left, right, strict=strict or exact, class_match=class_match):
            yield forward, reverse

    return next(compare(pattern, target, {}, {}), None) is not None


_PREDICATE_SCHEMA = "dryml-node-query-predicate"


def _predicate_to_data(predicate):
    """Encode closed records inside one shared Node table, never opaque blobs."""
    from ..domains import UniformFromSet, UniformIntRange
    from ..template_codec import _encode_node_value, TemplateCodecError
    from ..node import _rewrite, _clone
    from ..freeze import FrozenDict
    from ..quoted import QuotedDef
    from ..links import DefLink
    from ..template import _template_children
    memo, opaque, seen = {}, set(), set()
    def mark(value, quoted=False):
        key = (id(value), quoted)
        if key in seen:
            return
        seen.add(key)
        if isinstance(value, QuotedDef):
            mark(value.value, True)
        elif isinstance(value, DefLink):
            mark(value.target, quoted)
        else:
            if quoted and isinstance(value, Node):
                opaque.add(value._view)
            for item in _template_children(value):
                mark(item, quoted)
    def inspect(current):
        if current.kind == "node":
            mark(current.value)
        elif current.kind == "support":
            mark(current.value._generator.definition)
            for domain in current.value._generator.distributions.values():
                if type(domain) is UniformFromSet:
                    for item in domain.values:
                        mark(item)
        for operand in current.operands:
            inspect(operand)
    inspect(predicate)
    def named_pattern(root):
        def normalize(old, node):
            if (not isinstance(node, ConcreteDefinition) and isinstance(node.cls, type)
                    and node._view not in opaque and old._view not in opaque):
                return _clone(node, None, FrozenDict(_parameters(node)), node._facets)
            return node
        return _rewrite(root, node_transform=normalize, enter_quoted=False, memo=memo)
    records = {}
    def record(current):
        if id(current) not in records:
            records[id(current)] = record_value(current)
        return records[id(current)]
    def record_value(current):
        if current.kind == "node":
            return {"kind": "node", "pattern": named_pattern(current.value)}
        if current.kind == "absent":
            return {"kind": "absent", "facet": current.value[0], "path": current.value[1].to_data()}
        if current.kind == "support":
            selector, domains = current.value, {}
            generator = selector._generator
            for name, domain in generator.distributions.items():
                if type(domain) is UniformIntRange:
                    domains[name] = {"kind": "int-range", "lo": domain.lo, "hi": domain.hi}
                elif type(domain) is UniformFromSet:
                    domains[name] = {"kind": "set", "values": tuple(named_pattern(item) for item in domain.values)}
                else:
                    raise TemplateCodecError("query predicate has a nonportable runtime provider")
            return {"kind": "support", "definition": named_pattern(generator.definition), "domains": domains,
                    "traverse_refs": generator._traverse_refs, "max_assignments": selector._max_assignments}
        return {"kind": current.kind, "operands": tuple(record(item) for item in current.operands)}
    try:
        data = _encode_node_value("query-predicate", record(predicate))
    except SyntaxError as error:
        raise TemplateCodecError("query predicate contains a nonportable callable") from error
    return {**data, "schema": _PREDICATE_SCHEMA}


def _predicate_from_data(data):
    from ..domains import UniformFromSet, UniformIntRange
    from ..generator import Generator
    from ..template_codec import _decode_node_value, NODE_TEMPLATE_SCHEMA, TemplateCodecError
    from ..utils.graph.path import GraphPath
    if not isinstance(data, Mapping) or data.get("schema") != _PREDICATE_SCHEMA:
        raise TemplateCodecError("Invalid Node query predicate family")
    payload = _decode_node_value({**data, "schema": NODE_TEMPLATE_SCHEMA}, "query-predicate")
    def restore(row):
        if not isinstance(row, Mapping):
            raise TemplateCodecError("Invalid query predicate record")
        kind = row.get("kind")
        if kind == "node" and set(row) == {"kind", "pattern"}:
            return QueryPredicate(row["pattern"])
        if kind == "absent" and set(row) == {"kind", "facet", "path"}:
            return QueryPredicate.absent(row["facet"], path=GraphPath.from_data(row["path"]))
        if kind == "support" and set(row) == {"kind", "definition", "domains", "traverse_refs", "max_assignments"}:
            domains = {}
            for name, domain in row["domains"].items():
                if domain["kind"] == "int-range" and set(domain) == {"kind", "lo", "hi"}:
                    domains[name] = UniformIntRange(domain["lo"], domain["hi"])
                elif domain["kind"] == "set" and set(domain) == {"kind", "values"}:
                    domains[name] = UniformFromSet(domain["values"])
                else:
                    raise TemplateCodecError("Invalid query predicate domain")
            return QueryPredicate.support(Generator(row["definition"], domains, traverse_refs=row["traverse_refs"]),
                                          max_assignments=row["max_assignments"])
        if kind in {"and", "or", "not"} and set(row) == {"kind", "operands"}:
            operands = tuple(restore(item) for item in row["operands"])
            if len(operands) == (1 if kind == "not" else 2):
                return QueryPredicate._make(kind, operands=operands)
        raise TemplateCodecError("Invalid query predicate form")
    try:
        result = restore(payload)
        if result.to_data() != dict(data):
            raise TemplateCodecError("Noncanonical query predicate")
        return result
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise TemplateCodecError("Invalid query predicate payload") from error
