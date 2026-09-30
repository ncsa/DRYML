"""Definition generation and exact finite-support verification.

``Generator`` owns runtime Distribution policy while ``Definition`` owns
symbolic construction structure.  The paired selector proves exact support
without materializing recipe targets or serializing arbitrary providers.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import random
from types import MappingProxyType

from .definition import ConcreteDefinition, Definition, _structural_value_equal
from .domains import Distribution, UniformFromSet
from .errors import (
    ParameterizationError,
    ParameterizationLimitError,
    UnresolvedDefinitionError,
    UnsupportedGeneratorVerificationError,
)
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .template import Expr, Par, _snapshot_parameters


_MAX_GRID_RESULTS = 4_096
_MAX_ASSIGNMENTS = 65_536


def _validate_limit(value: object, *, label: str, maximum: int) -> int:
    """Validate one positive caller-lowered operation bound."""

    if type(value) is not int or value <= 0:
        raise ParameterizationError(f"{label} must be a positive exact int")
    if value > maximum:
        raise ParameterizationLimitError(f"{label} exceeds the generator hard limit")
    return value


def _provider_cardinality(provider: Distribution, root: str) -> int:
    """Read and validate a finite provider cardinality before indexing it."""

    try:
        cardinality = provider.cardinality()
    except Exception as error:
        raise ParameterizationError("distribution cardinality failed", root=root) from error
    if type(cardinality) is not int or cardinality <= 0:
        raise ParameterizationError("distribution must declare a positive finite cardinality", root=root)
    return cardinality


def _support_cardinality(provider: Distribution, root: str) -> int:
    """Require finite indexed support for an exact selector proof."""

    try:
        cardinality = provider.cardinality()
    except Exception as error:
        raise ParameterizationError("distribution cardinality failed", root=root) from error
    if cardinality is None:
        raise UnsupportedGeneratorVerificationError(
            "exact generator support needs finite indexed distribution support", root=root
        )
    if type(cardinality) is not int or cardinality <= 0:
        raise ParameterizationError("distribution must declare a positive finite cardinality", root=root)
    return cardinality


def _provider_value(provider: Distribution, root: str, index: int) -> object:
    """Read one validated finite support value without exposing provider errors."""

    try:
        return provider.value_at(index)
    except ParameterizationError:
        raise
    except Exception as error:
        raise ParameterizationError("distribution indexed support lookup failed", root=root) from error


class Generator:
    """Capture exact Distribution coverage for an immutable symbolic Definition.

    Args:
        definition: Definition whose active roots are generated.
        distributions: Mapping from every active root to one Distribution.
        traverse_refs: Whether Ref-carried Definition quotation data participates
            in coverage and substitution.

    Raises:
        ParameterizationError: If the Definition, mapping, root coverage, or a
            provider is invalid. Static values are rejected; bind them first with
            :meth:`Definition.sub`.

    Construction captures providers but never samples them, resolves targets,
    invokes factories, materializes Objects, or persists provider state.
    """

    def __init__(
            self,
            definition: Definition,
            distributions: Mapping[str, Distribution],
            /,
            *,
            traverse_refs: bool = False) -> None:
        if not isinstance(definition, Definition):
            raise ParameterizationError("Generator requires a Definition")
        if not isinstance(distributions, Mapping):
            raise ParameterizationError("Generator distributions must be a mapping")
        if type(traverse_refs) is not bool:
            raise ParameterizationError("traverse_refs must be a bool")
        roots = {parameter.name for parameter in _snapshot_parameters(definition, traverse_refs=traverse_refs)}
        supplied = dict(distributions)
        if any(type(name) is not str for name in supplied):
            raise ParameterizationError("Generator distribution roots must be strings")
        missing = sorted(roots - set(supplied))
        unknown = sorted(set(supplied) - roots)
        if missing:
            raise ParameterizationError(f"missing Generator distributions: {missing!r}")
        if unknown:
            raise ParameterizationError(f"unknown Generator distributions: {unknown!r}")
        static = [name for name, provider in supplied.items() if not isinstance(provider, Distribution)]
        if static:
            raise ParameterizationError(
                "Generator distributions must be Distribution values; bind static values with Definition.sub(...) first"
            )
        self._definition = definition
        self._distributions = MappingProxyType({name: supplied[name] for name in sorted(roots)})
        self._traverse_refs = traverse_refs
        self._validate_known_choices()

    @property
    def definition(self) -> Definition:
        """Return the immutable Definition captured without static substitution."""

        return self._definition

    @property
    def distributions(self) -> Mapping[str, Distribution]:
        """Return the read-only sorted root-to-Distribution association."""

        return self._distributions

    def sample(self, rng: random.Random | None = None) -> Definition:
        """Sample every provider in sorted fully-qualified root order.

        Args:
            rng: Optional caller-owned random generator advanced for this draw.

        Returns:
            A symbolically resolved Definition containing no Distribution.

        Raises:
            ParameterizationError: If the RNG, provider, or sampled value is
                invalid.
            UnresolvedDefinitionError: If a sampled assignment remains symbolic.

        A supplied RNG is not rolled back after provider or completion failure.
        """

        if rng is None:
            rng = random.Random()
        if not isinstance(rng, random.Random):
            raise ParameterizationError("sample rng must be random.Random or None")
        assignments = {}
        for name, provider in self._distributions.items():
            try:
                assignments[name] = provider.sample(rng)
            except ParameterizationError:
                raise
            except Exception as error:
                raise ParameterizationError("distribution sampling failed", root=name) from error
        return self._definition_for(assignments)

    def grid(self, *, max_results: int = _MAX_GRID_RESULTS) -> tuple[Definition, ...]:
        """Enumerate finite provider support in sorted mixed-radix root order.

        Args:
            max_results: Positive cap no greater than 4,096.

        Returns:
            Every symbolically resolved generated Definition.

        Raises:
            ParameterizationLimitError: If finite support exceeds the cap.
            ParameterizationError: If indexed provider support is invalid.

        Results are all-or-error: no partial tuple is returned on failure.
        """

        maximum = _validate_limit(max_results, label="max_results", maximum=_MAX_GRID_RESULTS)
        cardinalities = [_provider_cardinality(provider, name) for name, provider in self._distributions.items()]
        total = 1
        for cardinality in cardinalities:
            total *= cardinality
            if total > maximum:
                raise ParameterizationLimitError("generator grid result limit exceeded")
        names = tuple(self._distributions)
        values = [
            tuple(_provider_value(self._distributions[name], name, index) for index in range(cardinality))
            for name, cardinality in zip(names, cardinalities)
        ]
        results = [self._definition_for(dict(zip(names, choice))) for choice in product(*values)]
        return tuple(results)

    def support_selector(self, *, max_assignments: int = _MAX_ASSIGNMENTS) -> "GeneratorSelector":
        """Return an exact finite-support selector for this captured generator.

        Args:
            max_assignments: Positive finite proof cap for one candidate.

        Returns:
            A GeneratorSelector reusing this immutable capture.

        Raises:
            ParameterizationError: If the limit is malformed.
            ParameterizationLimitError: If it exceeds the fixed hard limit.
        """

        return GeneratorSelector(self, max_assignments=max_assignments)

    def _validate_known_choices(self) -> None:
        """Reject built-in choices which introduce uncovered active roots."""

        roots = set(self._distributions)
        for name, provider in self._distributions.items():
            if not isinstance(provider, UniformFromSet):
                continue
            expected = roots - {name}
            for value in provider.values:
                result = self._definition.sub(sub_dict={name: value}, traverse_refs=self._traverse_refs)
                active = {
                    parameter.name
                    for parameter in _snapshot_parameters(result, traverse_refs=self._traverse_refs)
                }
                if active != expected:
                    raise ParameterizationError("known distribution value introduces uncovered active roots", root=name)

    def _definition_for(self, assignments: Mapping[str, object]) -> Definition:
        """Apply one complete provider assignment and enforce resolution."""

        bound = self._definition.sub(sub_dict=assignments, traverse_refs=self._traverse_refs)
        active = _snapshot_parameters(bound, traverse_refs=self._traverse_refs)
        if active or not bound.is_resolved:
            raise UnresolvedDefinitionError("generator assignment left active expressions")
        return bound


class GeneratorSelector:
    """Prove exact captured Generator support for a Definition-like candidate.

    Args:
        generator: Captured Generator whose Definition and providers are retained.
        max_assignments: Positive finite enumeration cap for one proof.

    Visible roots are inferred directly. Remaining roots are enumerated only when
    finite support proves exact verification possible; bounds never substitute
    for proof. A selector is portable only when every provider is a supported
    immutable built-in Distribution; a Generator itself is runtime-only.

    Raises:
        ParameterizationError: If the Generator or proof limit is invalid.
        ParameterizationLimitError: If the requested proof limit exceeds the
            fixed hard limit.

    Side Effects:
        Construction creates an inert prefilter only. It does not sample
        providers, resolve recipe targets, materialize Objects, or persist data.
    """

    def __init__(self, generator: Generator, /, *, max_assignments: int = _MAX_ASSIGNMENTS) -> None:
        if not isinstance(generator, Generator):
            raise ParameterizationError("GeneratorSelector requires a Generator")
        self._generator = generator
        self._max_assignments = _validate_limit(max_assignments, label="max_assignments", maximum=_MAX_ASSIGNMENTS)
        self._prefilter = _loose_selector(
            _quote_ref_definitions(generator.definition),
            traverse_refs=generator._traverse_refs,
        )

    @property
    def prefilter(self):
        """Return the conservative ordinary Selector used before exact proof.

        Returns:
            A Selector that may reject candidates cheaply but cannot establish
            Generator support by itself.

        Side Effects:
            None. Access does not inspect a candidate or invoke a provider.
        """

        return self._prefilter

    def to_data(self) -> dict[str, object]:
        """Encode immutable built-in domains as ``dryml-template`` v1 data.

        Raises:
            ParameterizationError: If a provider is not portable built-in data.

        Arbitrary providers and Generator itself are deliberately nonportable.
        """

        from .template_codec import selector_to_data

        return selector_to_data(self)

    @classmethod
    def from_data(cls, data: Mapping[str, object], /) -> "GeneratorSelector":
        """Decode a portable exact selector directly into Definition state.

        Args:
            data: Closed ``dryml-template`` v1 selector data.

        Returns:
            A GeneratorSelector without target import, construction, or sampling.

        Raises:
            ParameterizationError: If the payload or captured domains are invalid.
        """

        from .template_codec import selector_from_data

        return selector_from_data(data)

    def matches(self, target: Definition | ConcreteDefinition | object, /) -> bool:
        """Return whether a candidate has one exact captured assignment.

        Args:
            target: Definition, ConcreteDefinition, or Object candidate. Other
                values return ``False`` without provider interaction.

        Returns:
            ``True`` only when the candidate has one exact Distribution
            assignment and matching symbolic graph topology.

        Raises:
            UnsupportedGeneratorVerificationError: If exact proof is unavailable.
            ParameterizationLimitError: If finite proof work exceeds its budget.
            ParameterizationError: If a provider or generated assignment is invalid.

        Side Effects:
            May invoke provider membership, cardinality, and indexed-value
            methods. It never samples, resolves recipe targets, materializes
            Objects, or persists data.
        """

        return self._matches(target, _AssignmentBudget(self._max_assignments))

    def _matches(self, target: Definition | ConcreteDefinition | object, budget: "_AssignmentBudget") -> bool:
        """Verify one candidate while charging a caller-owned finite budget."""

        from .object import Object

        if isinstance(target, Object):
            target = target.definition
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        inferred = self._infer_visible_roots(target)
        for name, value in inferred.items():
            if value is _INCONSISTENT or self._membership(name, value, budget) is False:
                return False
        unknown = [name for name in self._generator.distributions if name not in inferred]
        cardinalities = []
        total = 1
        for name in unknown:
            cardinality = _support_cardinality(self._generator.distributions[name], name)
            cardinalities.append(cardinality)
            total *= cardinality
            if total > self._max_assignments:
                raise ParameterizationLimitError("generator support verification assignment limit exceeded")
        budget.charge(total)
        values = [
            tuple(_provider_value(self._generator.distributions[name], name, index) for index in range(cardinality))
            for name, cardinality in zip(unknown, cardinalities)
        ]
        matched = False
        for choice in product(*values):
            assignments = dict(inferred)
            assignments.update(zip(unknown, choice))
            generated = _quote_ref_definitions(self._generator._definition_for(assignments))
            if generated.match(target, strict=True, cls_policy="exact") and _topology_matches(generated, target):
                matched = True
        return matched

    def _membership(self, name: str, value: object, budget: "_AssignmentBudget") -> bool:
        """Prove visible root membership without unnecessary enumeration."""

        provider = self._generator.distributions[name]
        bounds = _provider_bounds(provider, name)
        if bounds is not None and _outside_bounds(value, bounds):
            return False
        try:
            contains = provider.contains(value)
        except Exception as error:
            raise ParameterizationError("distribution membership check failed", root=name) from error
        if type(contains) is bool:
            return contains
        if contains is not None:
            raise ParameterizationError("distribution membership must return bool or None", root=name)
        cardinality = _support_cardinality(provider, name)
        if cardinality > self._max_assignments:
            raise ParameterizationLimitError("generator support membership scan limit exceeded")
        budget.charge(cardinality)
        return any(_structural_value_equal(value, _provider_value(provider, name, index)) for index in range(cardinality))

    def _infer_visible_roots(self, target: Definition | ConcreteDefinition) -> dict[str, object]:
        """Collect consistent direct root occurrences from a candidate."""

        found: dict[str, object] = {}

        def record(parameter: Par, value: object) -> None:
            if parameter.path or parameter.name not in self._generator.distributions:
                return
            prior = found.get(parameter.name, _MISSING)
            found[parameter.name] = value if prior is _MISSING else (
                prior if _structural_value_equal(prior, value) else _INCONSISTENT
            )

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
                    source_values = candidate_values = {}
                for key, value in source_values.items():
                    if key in candidate_values:
                        pairs(value, candidate_values[key])
                return
            from .cdef_graph import EdgeKind
            from .factory import FactorySpec
            from .links import DefLink

            if isinstance(source, DefLink):
                if isinstance(candidate, DefLink) and (source.kind is EdgeKind.MATERIALIZE or self._generator._traverse_refs):
                    pairs(source.target, candidate.target)
                return
            if isinstance(source, FactorySpec):
                if isinstance(candidate, FactorySpec):
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

        pairs(self._generator.definition, target)
        if any(value is _INCONSISTENT for value in found.values()):
            return {name: _INCONSISTENT for name, value in found.items() if value is _INCONSISTENT}
        return found


_MISSING = object()
_INCONSISTENT = object()
_UNKNOWN = object()


class _AssignmentBudget:
    """Bound cumulative finite-support work for one verification terminal."""

    def __init__(self, limit: int) -> None:
        self.remaining = limit

    def charge(self, count: int) -> None:
        """Consume assignments or fail before partial verification."""

        if count > self.remaining:
            raise ParameterizationLimitError("generator support verification assignment limit exceeded")
        self.remaining -= count


def _quote_ref_definitions(value: Definition) -> Definition:
    """Lower Ref-carried Definitions to their persisted quotation representation.

    Exact support compares generated Definitions with CDefs already admitted by a
    Template role.  That role persists a carried Definition as ``QuotedDef``;
    this selector-only projection preserves that delivery representation without
    changing Generator samples or inspecting constructor signatures.
    """

    from .cdef_graph import EdgeKind
    from .factory import FactorySpec
    from .links import DefLink
    from .quoted import QuotedDef

    memo: dict[int, object] = {}

    def project(current: object) -> object:
        if isinstance(current, (Expr, QuotedDef)):
            return current
        marker = id(current)
        if marker in memo:
            return memo[marker]
        if isinstance(current, DefLink):
            if current.kind is EdgeKind.REF and isinstance(current.target, Definition):
                target = QuotedDef(current.target)
            elif current.kind is EdgeKind.MATERIALIZE:
                target = project(current.target)
            else:
                return current
            result = (
                current if target is current.target else
                DefLink.finalized(current.kind, target) if current.is_finalized else DefLink.assertion(current.kind, target)
            )
        elif isinstance(current, Definition):
            args = None if current.args is None else FrozenTuple(project(item) for item in current.args)
            kwargs = FrozenDict((name, project(item)) for name, item in current.kwargs.items())
            result = current if args is current.args and kwargs == current.kwargs else Definition._from_symbolic_parts(current.cls, args, kwargs)
        elif isinstance(current, FactorySpec):
            args = FrozenTuple(project(item) for item in current.args)
            kwargs = FrozenDict((name, project(item)) for name, item in current.kwargs.items())
            result = current if args == current.args and kwargs == current.kwargs else FactorySpec._from_symbolic_parts(current.target, args, kwargs)
        elif isinstance(current, Mapping):
            result = FrozenDict((name, project(item)) for name, item in current.items())
        elif isinstance(current, (list, FrozenList)):
            result = FrozenList(project(item) for item in current)
        elif isinstance(current, (tuple, FrozenTuple)):
            result = FrozenTuple(project(item) for item in current)
        elif isinstance(current, (set, frozenset, FrozenSet)):
            result = FrozenSet(project(item) for item in current)
        else:
            return current
        memo[marker] = result
        return result

    result = project(value)
    if not isinstance(result, Definition):
        raise AssertionError("Generator selector projection must retain its Definition root")
    return result


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
        try:
            return value.parameters
        except TypeError as error:
            raise UnsupportedGeneratorVerificationError(
                "exact topology verification needs available supplied parameter names"
            ) from error

    def visit(left: object, right: object) -> bool:
        left_key, right_key = node_key(left), node_key(right)
        if left_key is not None or right_key is not None:
            if left_key is None or right_key is None:
                return False
            if left_key in generated_to_target:
                return generated_to_target[left_key] == right_key
            if right_key in target_to_generated:
                return False
            generated_to_target[left_key] = right_key
            target_to_generated[right_key] = left_key
            return all(name in definition_values(right) and visit(value, definition_values(right)[name]) for name, value in definition_values(left).items())
        if isinstance(left, DefLink):
            return True
        if isinstance(left, FactorySpec) and isinstance(right, FactorySpec):
            return len(left.args) == len(right.args) and all(visit(a, b) for a, b in zip(left.args, right.args)) and all(name in right.kwargs and visit(value, right.kwargs[name]) for name, value in left.kwargs.items())
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return all(name in right and visit(value, right[name]) for name, value in left.items())
        if isinstance(left, (list, tuple, FrozenList, FrozenTuple)) and isinstance(right, (list, tuple, FrozenList, FrozenTuple)):
            return len(left) == len(right) and all(visit(a, b) for a, b in zip(left, right))
        return True

    return visit(generated, target)


def _provider_bounds(provider: Distribution, root: str) -> tuple[int | float, int | float] | None:
    """Validate a provider's optional conservative numeric envelope."""

    try:
        bounds = provider.bounds()
    except Exception as error:
        raise ParameterizationError("distribution bounds check failed", root=root) from error
    if bounds is None:
        return None
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ParameterizationError("distribution bounds must be a pair", root=root)
    lo, hi = bounds
    if type(lo) not in {int, float} or type(hi) not in {int, float} or lo != lo or hi != hi or lo > hi:
        raise ParameterizationError("distribution bounds are invalid", root=root)
    return lo, hi


def _outside_bounds(value: object, bounds: tuple[int | float, int | float]) -> bool:
    """Return whether an exact numeric candidate is outside known bounds."""

    return type(value) in {int, float} and (value < bounds[0] or value > bounds[1])


def _loose_selector(value: Definition, *, traverse_refs: bool = False):
    """Project one symbolic Definition into a conservative ordinary Selector."""

    from .cdef_graph import EdgeKind
    from .factory import FactorySpec
    from .links import DefLink
    from .params import AnyValue
    from .quoted import QuotedDef
    from .selector import Selector

    if not isinstance(value, Definition):
        raise ParameterizationError("loose_selector requires a Definition root")

    def project(current: object) -> object:
        if isinstance(current, Expr):
            return _UNKNOWN
        if isinstance(current, DefLink):
            if current.kind is EdgeKind.MATERIALIZE:
                target = project(current.target)
                if target is _UNKNOWN:
                    return _UNKNOWN
                return DefLink.finalized(current.kind, target) if current.is_finalized else DefLink.assertion(current.kind, target)
            if traverse_refs and current.kind is EdgeKind.REF:
                target = current.target.value if isinstance(current.target, QuotedDef) else current.target
                if isinstance(target, Definition) and not target.is_resolved:
                    return _UNKNOWN
            return current
        if isinstance(current, FactorySpec):
            return FactorySpec._from_symbolic_parts(current.target, tuple(AnyValue() if (item := project(arg)) is _UNKNOWN else item for arg in current.args), FrozenDict((name, AnyValue() if (item := project(arg)) is _UNKNOWN else item) for name, arg in current.kwargs.items()))
        if isinstance(current, Definition):
            args = None if current.args is None else FrozenTuple(AnyValue() if (item := project(arg)) is _UNKNOWN else item for arg in current.args)
            kwargs = FrozenDict((name, item) for name, arg in current.kwargs.items() if (item := project(arg)) is not _UNKNOWN)
            return Definition._from_symbolic_parts(current.cls, args, kwargs)
        if isinstance(current, Mapping):
            return FrozenDict((name, item) for name, arg in current.items() if (item := project(arg)) is not _UNKNOWN)
        if isinstance(current, (list, FrozenList)):
            return FrozenList(AnyValue() if (item := project(arg)) is _UNKNOWN else item for arg in current)
        if isinstance(current, (tuple, FrozenTuple)):
            return FrozenTuple(AnyValue() if (item := project(arg)) is _UNKNOWN else item for arg in current)
        if isinstance(current, (set, frozenset, FrozenSet)):
            items = [project(item) for item in current]
            return _UNKNOWN if any(item is _UNKNOWN for item in items) else FrozenSet(items)
        return current

    projected = project(value)
    if projected is _UNKNOWN or not isinstance(projected, Definition):
        raise ParameterizationError("loose_selector requires a projectable Definition root")
    return Selector(projected, cls_policy="exact")


__all__ = ["Generator", "GeneratorSelector"]
