"""Captured template generation and exact finite-support verification.

This module keeps generation separate from the ordinary loose ``Selector``
surface.  A generator owns a complete static/domain capture, while its support
selector proves candidate membership against that same joint assignment space.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from types import MappingProxyType
import random
from typing import Any

from .definition import ConcreteDefinition, Definition
from .domains import Distribution, UniformFromSet
from .errors import (
    TemplateError,
    TemplateLimitError,
    UnresolvedTemplateError,
    UnsupportedTemplateVerificationError,
)
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .template import (
    Expr,
    Par,
    Template,
    _normalize_bindings,
    _snapshot_parameters,
)


_MAX_GRID_RESULTS = 4_096
_MAX_ASSIGNMENTS = 65_536


def _same_value(left: object, right: object) -> bool:
    """Compare generated scalar values without numeric coercion."""

    from .definition import _structural_value_equal

    return _structural_value_equal(left, right)


def _validate_limit(value: object, *, label: str, maximum: int) -> int:
    """Validate one positive caller-lowered operation bound."""

    if type(value) is not int or value <= 0:
        raise TemplateError(f"{label} must be a positive exact int")
    if value > maximum:
        raise TemplateLimitError(f"{label} exceeds the template hard limit")
    return value


def _provider_cardinality(provider: Distribution, root: str) -> int:
    """Read and validate a finite provider cardinality before indexing it."""

    try:
        cardinality = provider.cardinality()
    except Exception as error:
        raise TemplateError("distribution cardinality failed", root=root) from error
    if type(cardinality) is not int or cardinality <= 0:
        raise TemplateError("distribution must declare a positive finite cardinality", root=root)
    return cardinality


def _support_cardinality(provider: Distribution, root: str) -> int:
    """Require finite indexed support for an exact selector proof."""

    try:
        cardinality = provider.cardinality()
    except Exception as error:
        raise TemplateError("distribution cardinality failed", root=root) from error
    if cardinality is None:
        raise UnsupportedTemplateVerificationError(
            "exact template support needs finite indexed domain support", root=root
        )
    if type(cardinality) is not int or cardinality <= 0:
        raise TemplateError("distribution must declare a positive finite cardinality", root=root)
    return cardinality


def _provider_value(provider: Distribution, root: str, index: int) -> object:
    """Read one validated finite support value without exposing provider errors."""

    try:
        return provider.value_at(index)
    except TemplateError:
        raise
    except Exception as error:
        raise TemplateError("distribution indexed support lookup failed", root=root) from error


class TemplateGenerator:
    """Capture complete static and domain bindings for reusable Definitions.

    Args:
        template: A soft-Definition-rooted template with ordinary supplied call
            spelling.
        sub_dict: Fully qualified static or ``Distribution`` root bindings.
        namespace: Namespace applied only to keyword bindings.
        traverse_refs: Whether Ref-carried template recipes participate in the
            capture and completion boundary.
        **bindings: Unqualified bindings under ``namespace``.

    Raises:
        TemplateError: If the root or binding coverage is invalid, static capture
            cannot be represented, or a known choice introduces an uncovered root.

    Construction does not sample providers, resolve targets, or materialize
    Objects. Live static Object values are detached through ``Template.sub``.
    """

    def __init__(
        self,
        template: Template,
        /,
        *,
        sub_dict: Mapping[str, object] | None = None,
        namespace: object = (),
        traverse_refs: bool = False,
        **bindings: object,
    ) -> None:
        if not isinstance(template, Template):
            raise TemplateError("TemplateGenerator requires a Template")
        if type(traverse_refs) is not bool:
            raise TemplateError("traverse_refs must be a bool")
        if not isinstance(template.root, Definition):
            raise TemplateError("TemplateGenerator requires a soft Definition root")
        if template.root.cls is None or template.root.args is None:
            raise TemplateError("TemplateGenerator requires a classed non-skip-args Definition root")

        occurrences = _snapshot_parameters(template.root, traverse_refs=traverse_refs)
        roots = {parameter.name for parameter in occurrences}
        supplied = _normalize_bindings(sub_dict, namespace, bindings)
        missing = sorted(roots - set(supplied))
        unknown = sorted(set(supplied) - roots)
        if missing:
            raise TemplateError(f"missing template generator bindings: {missing!r}")
        if unknown:
            raise TemplateError(f"unknown template generator bindings: {unknown!r}")

        statics = {name: value for name, value in supplied.items() if not isinstance(value, Distribution)}
        domains = {name: value for name, value in supplied.items() if isinstance(value, Distribution)}
        prepared = template.sub(sub_dict=statics, traverse_refs=traverse_refs)
        remaining = tuple(sorted({
            parameter.name
            for parameter in _snapshot_parameters(prepared.root, traverse_refs=traverse_refs)
        }))
        remaining_roots = set(remaining)
        if remaining_roots != set(domains):
            raise TemplateError("static capture does not leave exactly the declared distribution roots")

        self._template = prepared
        self._domains = MappingProxyType({name: domains[name] for name in remaining})
        self._traverse_refs = traverse_refs
        self._validate_known_choices()

    @property
    def template(self) -> Template:
        """Return the immutable template after one static capture pass."""

        return self._template

    @property
    def domains(self) -> Mapping[str, Distribution]:
        """Return the read-only remaining qualified-root domain association."""

        return self._domains

    def sample(self, rng: random.Random | None = None) -> Definition:
        """Draw one value per lexical remaining root and return a Definition.

        Args:
            rng: Optional caller-owned ``random.Random`` advanced for this draw.

        Returns:
            A fully bound soft Definition.

        Raises:
            TemplateError: If a provider fails or supplies an invalid value.
            UnresolvedTemplateError: If a provider introduces active roots.

        A supplied RNG is not rolled back when a provider or completion fails.
        """

        if rng is None:
            rng = random.Random()
        if not isinstance(rng, random.Random):
            raise TemplateError("sample rng must be random.Random or None")
        assignments = {}
        for name, provider in self._domains.items():
            try:
                assignments[name] = provider.sample(rng)
            except TemplateError:
                raise
            except Exception as error:
                raise TemplateError("distribution sampling failed", root=name) from error
        return self._definition_for(assignments)

    def grid(self, *, max_results: int = _MAX_GRID_RESULTS) -> tuple[Definition, ...]:
        """Enumerate the finite captured product in lexical root order.

        Args:
            max_results: Positive result cap no greater than 4,096.

        Returns:
            Every fully bound Definition in mixed-radix root order.

        Raises:
            TemplateLimitError: If the finite product exceeds ``max_results``.
            TemplateError: If a provider violates its indexed-support contract.

        The return is all-or-error: no partial tuple is exposed on failure.
        """

        maximum = _validate_limit(max_results, label="max_results", maximum=_MAX_GRID_RESULTS)
        cardinalities = [
            _provider_cardinality(provider, name)
            for name, provider in self._domains.items()
        ]
        total = 1
        for cardinality in cardinalities:
            total *= cardinality
            if total > maximum:
                raise TemplateLimitError("template grid result limit exceeded")
        names = tuple(self._domains)
        values = [
            tuple(_provider_value(self._domains[name], name, index) for index in range(cardinality))
            for name, cardinality in zip(names, cardinalities)
        ]
        results = [self._definition_for(dict(zip(names, choice))) for choice in product(*values)]
        return tuple(results)

    def support_selector(self, *, max_assignments: int = _MAX_ASSIGNMENTS) -> "TemplateSelector":
        """Return an exact support verifier for this captured specification.

        Args:
            max_assignments: Positive per-candidate finite proof cap.

        Returns:
            A TemplateSelector that reuses this immutable capture.

        Raises:
            TemplateError: If ``max_assignments`` is malformed.
            TemplateLimitError: If it exceeds the fixed hard limit.
        """

        return TemplateSelector(self, max_assignments=max_assignments)

    def _validate_known_choices(self) -> None:
        """Reject built-in choice values that would make a sample incomplete."""

        roots = set(self._domains)
        for name, provider in self._domains.items():
            if not isinstance(provider, UniformFromSet):
                continue
            expected = roots - {name}
            for value in provider.values:
                result = self._template.sub(sub_dict={name: value}, traverse_refs=self._traverse_refs)
                active = {
                    parameter.name
                    for parameter in _snapshot_parameters(result.root, traverse_refs=self._traverse_refs)
                }
                if active != expected:
                    raise TemplateError("known distribution value introduces uncovered active roots", root=name)

    def _definition_for(self, assignments: Mapping[str, object]) -> Definition:
        """Apply one complete assignment and enforce the generator boundary."""

        bound = self._template.sub(sub_dict=assignments, traverse_refs=self._traverse_refs)
        active = _snapshot_parameters(bound.root, traverse_refs=self._traverse_refs)
        if active:
            raise UnresolvedTemplateError("generator assignment left active template roots")
        return bound.to_definition()


class TemplateSelector:
    """Prove exact captured template support for a Definition-like candidate.

    Args:
        generator: Captured generator whose template and domains are retained.
        max_assignments: Positive finite-enumeration cap for one candidate proof.

    Exact verification may infer directly visible roots without enumeration. It
    enumerates only the remaining finite roots, and never replaces an exhausted
    proof with a bounds approximation.
    """

    def __init__(self, generator: TemplateGenerator, /, *, max_assignments: int = _MAX_ASSIGNMENTS) -> None:
        if not isinstance(generator, TemplateGenerator):
            raise TemplateError("TemplateSelector requires a TemplateGenerator")
        self._generator = generator
        self._max_assignments = _validate_limit(
            max_assignments, label="max_assignments", maximum=_MAX_ASSIGNMENTS
        )
        self._prefilter = _loose_selector(generator.template)

    @property
    def prefilter(self):
        """Return the conservative ordinary Selector used for structural filtering."""

        return self._prefilter

    def matches(self, target: Definition | ConcreteDefinition | object, /) -> bool:
        """Return whether ``target`` has one exact captured support assignment.

        Args:
            target: A soft Definition, exact CDef, or Object carrying one.

        Returns:
            ``True`` only when an assignment satisfies all generated structure.

        Raises:
            UnsupportedTemplateVerificationError: If a surviving unknown root
                cannot be exactly verified.
            TemplateLimitError: If finite exact proof work exceeds its budget.
            TemplateError: If a provider or a generated assignment is invalid.
        """

        from .object import Object

        if isinstance(target, Object):
            target = target.definition
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        inferred = self._infer_visible_roots(target)
        for name, value in inferred.items():
            if value is _INCONSISTENT:
                return False
            membership = self._membership(name, value)
            if membership is False:
                return False

        unknown = [name for name in self._generator.domains if name not in inferred]
        cardinalities = []
        total = 1
        for name in unknown:
            provider = self._generator.domains[name]
            cardinality = _support_cardinality(provider, name)
            cardinalities.append(cardinality)
            total *= cardinality
            if total > self._max_assignments:
                raise TemplateLimitError("template support verification assignment limit exceeded")

        values = [
            tuple(_provider_value(self._generator.domains[name], name, index) for index in range(cardinality))
            for name, cardinality in zip(unknown, cardinalities)
        ]
        matched = False
        for choice in product(*values):
            assignments = dict(inferred)
            assignments.update(zip(unknown, choice))
            # Complete every finite branch even after a positive witness. A bad
            # generated branch is a malformed support contract, not a filter.
            generated = self._generator._definition_for(assignments)
            if (
                generated.match(target, strict=True, cls_policy="exact")
                and _topology_matches(generated, target)
            ):
                matched = True
        return matched

    def _membership(self, name: str, value: object) -> bool:
        """Prove visible-root membership without unnecessarily enumerating ranges."""

        provider = self._generator.domains[name]
        bounds = _provider_bounds(provider, name)
        if bounds is not None and _outside_bounds(value, bounds):
            return False
        try:
            contains = provider.contains(value)
        except Exception as error:
            raise TemplateError("distribution membership check failed", root=name) from error
        if type(contains) is bool:
            return contains
        if contains is not None:
            raise TemplateError("distribution membership must return bool or None", root=name)
        cardinality = _support_cardinality(provider, name)
        if cardinality > self._max_assignments:
            raise TemplateLimitError("template support membership scan limit exceeded")
        return any(_same_value(value, _provider_value(provider, name, index)) for index in range(cardinality))

    def _infer_visible_roots(self, target: Definition | ConcreteDefinition) -> dict[str, object]:
        """Collect consistent direct empty-path parameter occurrences from target."""

        found: dict[str, object] = {}

        def record(parameter: Par, value: object) -> None:
            if parameter.path or parameter.name not in self._generator.domains:
                return
            prior = found.get(parameter.name, _MISSING)
            if prior is _MISSING:
                found[parameter.name] = value
            elif not _same_value(prior, value):
                found[parameter.name] = _INCONSISTENT

        def pairs(source: object, candidate: object) -> None:
            if isinstance(source, Par):
                record(source, candidate)
                return
            if isinstance(source, Expr):
                return
            if isinstance(source, Definition):
                try:
                    source_values = source.parameters
                    candidate_values = candidate.parameters if isinstance(candidate, (Definition, ConcreteDefinition)) else {}
                except (TypeError, AttributeError):
                    source_values = {}
                    candidate_values = {}
                for key, value in source_values.items():
                    if key in candidate_values:
                        pairs(value, candidate_values[key])
                return
            from .factory import FactorySpec
            from .links import DefLink

            if isinstance(source, DefLink):
                if self._generator._traverse_refs and isinstance(candidate, DefLink):
                    pairs(source.target, candidate.target)
                return
            if isinstance(source, FactorySpec):
                if not isinstance(candidate, FactorySpec):
                    return
                for left, right in zip(source.args, candidate.args):
                    pairs(left, right)
                for key, value in source.kwargs.items():
                    if key in candidate.kwargs:
                        pairs(value, candidate.kwargs[key])
                return
            if isinstance(source, Mapping) and isinstance(candidate, Mapping):
                for key, value in source.items():
                    if key in candidate:
                        pairs(value, candidate[key])
                return
            if isinstance(source, (list, tuple, FrozenList, FrozenTuple)) and isinstance(candidate, (list, tuple, FrozenList, FrozenTuple)):
                for left, right in zip(source, candidate):
                    pairs(left, right)

        pairs(self._generator.template.root, target)
        if any(value is _INCONSISTENT for value in found.values()):
            return {name: _INCONSISTENT for name, value in found.items() if value is _INCONSISTENT}
        return found


_MISSING = object()
_INCONSISTENT = object()
_UNKNOWN = object()


def _topology_matches(generated: Definition, target: Definition | ConcreteDefinition) -> bool:
    """Require a two-way correspondence between generated and candidate nodes."""

    from .cdef_identity import cdef_node_key
    from .factory import FactorySpec
    from .links import DefLink

    generated_to_target: dict[tuple[str, int], tuple[str, int]] = {}
    target_to_generated: dict[tuple[str, int], tuple[str, int]] = {}

    def node_key(value: object) -> tuple[str, int] | None:
        if isinstance(value, ConcreteDefinition):
            return "cdef", id(cdef_node_key(value))
        if isinstance(value, Definition):
            return "definition", id(value)
        return None

    def definition_values(value: Definition | ConcreteDefinition) -> Mapping[str, object]:
        if isinstance(value, ConcreteDefinition):
            return value.parameters
        try:
            return value.parameters
        except TypeError as error:
            raise UnsupportedTemplateVerificationError(
                "exact topology verification needs available supplied parameter names"
            ) from error

    def visit(left: object, right: object) -> bool:
        left_key = node_key(left)
        right_key = node_key(right)
        if left_key is not None or right_key is not None:
            if left_key is None or right_key is None:
                return False
            if left_key in generated_to_target:
                return generated_to_target[left_key] == right_key
            if right_key in target_to_generated:
                return False
            generated_to_target[left_key] = right_key
            target_to_generated[right_key] = left_key

            left_values = definition_values(left)
            right_values = definition_values(right)
            return all(
                name in right_values and visit(value, right_values[name])
                for name, value in left_values.items()
            )
        if isinstance(left, DefLink):
            return True
        if isinstance(left, FactorySpec) and isinstance(right, FactorySpec):
            return (
                len(left.args) == len(right.args)
                and all(visit(a, b) for a, b in zip(left.args, right.args))
                and all(name in right.kwargs and visit(value, right.kwargs[name]) for name, value in left.kwargs.items())
            )
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return all(name in right and visit(value, right[name]) for name, value in left.items())
        if isinstance(left, (list, tuple, FrozenList, FrozenTuple)) and isinstance(
            right, (list, tuple, FrozenList, FrozenTuple)
        ):
            return len(left) == len(right) and all(visit(a, b) for a, b in zip(left, right))
        return True

    return visit(generated, target)


def _provider_bounds(provider: Distribution, root: str) -> tuple[int | float, int | float] | None:
    """Validate a provider's optional conservative numeric envelope."""

    try:
        bounds = provider.bounds()
    except Exception as error:
        raise TemplateError("distribution bounds check failed", root=root) from error
    if bounds is None:
        return None
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TemplateError("distribution bounds must be a pair", root=root)
    lo, hi = bounds
    if type(lo) not in {int, float} or type(hi) not in {int, float} or lo != lo or hi != hi or lo > hi:
        raise TemplateError("distribution bounds are invalid", root=root)
    return lo, hi


def _outside_bounds(value: object, bounds: tuple[int | float, int | float]) -> bool:
    """Return whether an exact numeric candidate is provably outside bounds."""

    return type(value) in {int, float} and (value < bounds[0] or value > bounds[1])


def _loose_selector(template: Template):
    """Project a soft template into an ordinary structural Selector."""

    from .factory import FactorySpec
    from .links import DefLink
    from .params import AnyValue
    from .selector import Selector

    if not isinstance(template.root, Definition):
        raise TemplateError("Template.as_selector requires a soft Definition root")

    def project(value: object, *, sequence: bool = False) -> object:
        if isinstance(value, (Par, Expr)):
            return _UNKNOWN
        if isinstance(value, DefLink):
            return value
        if isinstance(value, FactorySpec):
            args = tuple(AnyValue() if (item := project(arg, sequence=True)) is _UNKNOWN else item for arg in value.args)
            kwargs = FrozenDict(
                (name, AnyValue() if (item := project(arg)) is _UNKNOWN else item)
                for name, arg in value.kwargs.items()
            )
            return FactorySpec._from_template_parts(value.target, args, kwargs)
        if isinstance(value, Definition):
            args = None if value.args is None else FrozenTuple(
                AnyValue() if (item := project(arg, sequence=True)) is _UNKNOWN else item
                for arg in value.args
            )
            kwargs = FrozenDict(
                (name, item)
                for name, arg in value.kwargs.items()
                if (item := project(arg)) is not _UNKNOWN
            )
            return Definition._from_template_parts(value.cls, args, kwargs)
        if isinstance(value, Mapping):
            return FrozenDict(
                (name, item)
                for name, arg in value.items()
                if (item := project(arg)) is not _UNKNOWN
            )
        if isinstance(value, (list, FrozenList)):
            return FrozenList(AnyValue() if (item := project(arg, sequence=True)) is _UNKNOWN else item for arg in value)
        if isinstance(value, (tuple, FrozenTuple)):
            return FrozenTuple(AnyValue() if (item := project(arg, sequence=True)) is _UNKNOWN else item for arg in value)
        if isinstance(value, (set, frozenset, FrozenSet)):
            items = [project(item) for item in value]
            return _UNKNOWN if any(item is _UNKNOWN for item in items) else FrozenSet(items)
        return value

    projected = project(template.root)
    if projected is _UNKNOWN or not isinstance(projected, Definition):
        raise TemplateError("Template.as_selector requires a projectable soft Definition root")
    return Selector(projected, cls_policy="exact")


__all__ = ["TemplateGenerator", "TemplateSelector"]
