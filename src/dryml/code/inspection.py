"""Bounded passive projections for generic static dependency analysis.

The projection intentionally carries symbolic target records rather than
source, paths, signatures, or executable objects.  It is a correctness
boundary for ordinary callers, not a safe-deserialization boundary for hostile
objects.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .callable_info import _function_slot, _module_namespace, _type_slot
from .errors import InvalidTargetError
from .source import _source_within_bounds

if TYPE_CHECKING:
    from .source import SourceInfo
    from .targets import CodeTarget, CodeTargetInput, TargetInfo

_VERSION = 1
_MAX_TARGETS = 4_096
_MAX_CALLS = 16_384
_MAX_SOURCE_BYTES = 1_048_576
_MAX_SOURCE_TOTAL = 8 * _MAX_SOURCE_BYTES
_MAX_AST_NODES = 100_000
_MAX_AST_DEPTH = 128
_MAX_MRO = 256
_MAX_TEXT = 65_536
_MISSING = object()


@dataclass(frozen=True, slots=True)
class InspectionCall:
    """One resolved or unresolved direct call edge in a detached projection.

    Args:
        target_id: Projection-local callee identifier, or ``None`` when the
            passive binding grammar could not prove a target.

    Raises:
        ValueError: If the identifier is not a non-empty exact string or
            ``None``.
    """

    target_id: str | None

    def __post_init__(self) -> None:
        """Validate the closed projection edge carrier."""

        if self.target_id is not None and (type(self.target_id) is not str
                                           or not self.target_id):
            raise ValueError("inspection call is invalid")


@dataclass(frozen=True, slots=True)
class InspectionRecord:
    """Source-free static facts for one target in an inspection projection.

    Args:
        target_id: Deterministic projection-local record identifier.
        info: Sanitized immutable semantic target metadata with no source
            fields.
        calls: Ordered direct call facts.
        incomplete: Whether capture encountered unavailable or bounded
            evidence.

    Raises:
        ValueError: If record fields are malformed or retain local provenance.
    """

    target_id: str
    info: "TargetInfo"
    calls: tuple[InspectionCall, ...]
    incomplete: bool = False

    def __post_init__(self) -> None:
        """Reject non-detached records before they can enter a snapshot."""

        if (type(self.target_id) is not str or not self.target_id
                or len(self.target_id) > _MAX_TEXT
                or not _valid_info(self.info)):
            raise ValueError("inspection record is invalid")
        if (self.info.filename is not None or self.info.start_line is not None
                or self.info.import_path is not None):
            raise ValueError("inspection record retains local provenance")
        if type(self.calls) is not tuple or len(self.calls) > _MAX_CALLS:
            raise ValueError("inspection record calls are invalid")
        if (any(type(call) is not InspectionCall for call in self.calls)
                or type(self.incomplete) is not bool):
            raise ValueError("inspection record is invalid")


@dataclass(frozen=True, slots=True)
class InspectionOwnerFacts:
    """
    Describe an owner-proven relationship for one generic capture.

        Args:
            root: The exact capture root to which the relationships apply.
            related: Existing live/static targets reached through the
            owner-proven
                relationship. They become direct root edges in the same bounded
                capture rather than separately merged snapshots.
            opaque: Related targets whose bodies are owner implementation
            detail and
                therefore retain declarations but contribute no inferred call
                edges.

        Raises:
            TypeError: If fields are not exact immutable tuples.
            ValueError: If no relationship is supplied or an opaque target is
            not a
                related target.

        Side Effects:
            None. This is a generic borrowed relationship fact; it does not
            inspect,
            invoke, or interpret any domain owner.
    """

    root: object
    related: tuple[object, ...]
    opaque: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity-only owner facts without target access."""

        if type(self.related) is not tuple or type(self.opaque) is not tuple:
            raise TypeError("inspection owner facts require exact tuples")
        if not self.related:
            raise ValueError("inspection owner facts require a related target")
        if any(
            not any(item is related for related in self.related)
            for item in self.opaque
        ):
            raise ValueError("inspection opaque target is not owner-related")


def _valid_info(value: object) -> bool:
    """Return whether detached metadata has only bounded exact primitives."""

    from .targets import TargetInfo

    if type(value) is not TargetInfo:
        return False
    texts = (
        value.kind,
        value.name,
        value.module,
        value.qualname,
        value.owner_module,
        value.owner_qualname,
        value.descriptor_kind,
        value.filename,
        value.import_path,
    )
    if any(item is not None and (
            type(item) is not str or len(item) > _MAX_TEXT) for item in texts):
        return False
    return (value.kind in (
        "function",
        "bound_method",
        "callable_instance",
        "descriptor",
        "class",
        "import",
        "source",
    ) and (value.descriptor_kind is None or value.descriptor_kind
           in ("function", "staticmethod", "classmethod"))
            and (value.start_line is None or
                 (type(value.start_line) is int and value.start_line > 0)))


def _wire(value: str | None) -> bytes:
    """Encode an optional text field without delimiter ambiguity."""

    if value is None:
        return b"n"
    raw = value.encode("utf-8")
    return b"s" + len(raw).to_bytes(8, "big") + raw


def _snapshot_identity(snapshot: "InspectionSnapshot") -> str:
    """Hash one validated snapshot using the stable wire-field sequence."""

    digest = hashlib.sha256()
    digest.update(b"dryml.code.inspection.v1")
    digest.update(snapshot.version.to_bytes(8, "big"))
    digest.update(_wire(snapshot.root_id))
    for record in snapshot.records:
        digest.update(_wire(record.target_id))
        for value in (
                record.info.kind,
                record.info.name,
                record.info.module,
                record.info.qualname,
                record.info.owner_module,
                record.info.owner_qualname,
                record.info.descriptor_kind,
        ):
            digest.update(_wire(value))
        digest.update(b"1" if record.incomplete else b"0")
        digest.update(len(record.calls).to_bytes(8, "big"))
        for call in record.calls:
            digest.update(_wire(call.target_id))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class InspectionSnapshot:
    """Validated, bounded, source-free static analysis projection.

    Args:
        version: Exact supported version-local projection version.
        root_id: Existing root record identifier.
        records: Unique ordered records, bounded by the capture contract.

    Raises:
        ValueError: If version, identifiers, references, primitive records, or
            fixed bounds are invalid.
    """

    version: int
    root_id: str
    records: tuple[InspectionRecord, ...]
    _identity: str = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate primitive fields and bounds before derived allocations."""

        if type(self.version) is not int or self.version != _VERSION:
            raise ValueError("inspection snapshot version is invalid")
        if (type(self.root_id) is not str or not self.root_id
                or type(self.records) is not tuple):
            raise ValueError("inspection snapshot is invalid")
        if not self.records or len(self.records) > _MAX_TARGETS:
            raise ValueError("inspection snapshot records are invalid")
        identifiers: set[str] = set()
        text_size = len(self.root_id)
        total_calls = 0
        for record in self.records:
            if (type(record) is not InspectionRecord
                    or record.target_id in identifiers):
                raise ValueError("inspection snapshot identifiers are invalid")
            identifiers.add(record.target_id)
            text_size += len(record.target_id)
            text_size += sum(
                len(value) for value in (
                    record.info.kind,
                    record.info.name,
                    record.info.module,
                    record.info.qualname,
                    record.info.owner_module,
                    record.info.owner_qualname,
                    record.info.descriptor_kind,
                ) if value is not None)
            total_calls += len(record.calls)
            if total_calls > _MAX_CALLS or text_size > _MAX_SOURCE_BYTES:
                raise ValueError("inspection snapshot calls are invalid")
        if self.root_id not in identifiers:
            raise ValueError("inspection snapshot identifiers are invalid")
        for record in self.records:
            for call in record.calls:
                if (call.target_id is not None
                        and call.target_id not in identifiers):
                    raise ValueError("inspection snapshot calls are invalid")
        object.__setattr__(self, "_identity", _snapshot_identity(self))

    @property
    def identity(self) -> str:
        """Return an unambiguous SHA-256 identity for this detached projection.
        """

        return self._identity

    def record(self, target_id: str) -> InspectionRecord:
        """Return one selected record without dynamic target reconstruction.

        Args:
            target_id: Existing projection-local identifier.

        Raises:
            InvalidTargetError: If the record does not exist.
        """

        for record in self.records:
            if record.target_id == target_id:
                return record
        raise InvalidTargetError("inspection target is unavailable")


@dataclass(frozen=True, slots=True)
class InspectionTarget:
    """Immutable source-free target selected from an inspection snapshot.

    Args:
        snapshot: Validated detached snapshot.
        target_id: Existing selected record identifier.

    Raises:
            ValueError: If the target does not select an existing exact
                snapshot record.
    """

    snapshot: InspectionSnapshot
    target_id: str

    def __post_init__(self) -> None:
        """Validate selection without consulting any target protocol."""

        if (type(self.snapshot) is not InspectionSnapshot
                or type(self.target_id) is not str):
            raise ValueError("inspection target is invalid")
        self.snapshot.record(self.target_id)

    @property
    def info(self) -> "TargetInfo":
        """Return the immutable metadata for the selected detached record."""

        return self.snapshot.record(self.target_id).info


@dataclass(frozen=True, slots=True)
class _BindingGuard:
    """One local identity observation for a global or closure binding."""

    function: types.FunctionType
    scope: str
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class _MemberGuard:
    """One raw module-member observation retained against local drift."""

    module: types.ModuleType
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class _ClassGuard:
    """One class-layout observation kept only by the coordinator."""

    cls: type
    bases: tuple[type, ...]
    members: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _CallableInstanceGuard:
    """One raw callable-instance binding observation kept locally."""

    instance: object
    owner: type
    owner_mro: tuple[type, ...]
    descriptor: object


@dataclass(frozen=True, slots=True)
class _LiveAssociation:
    """Private coordinator-local target identity and binding guard."""

    target_id: str
    target: "CodeTarget"
    code_id: int | None
    bindings: tuple[_BindingGuard, ...]
    members: tuple[_MemberGuard, ...]
    classes: tuple[_ClassGuard, ...]
    callable_instance: _CallableInstanceGuard | None


@dataclass(frozen=True, slots=True)
class InspectionCapture:
    """Detached root projection plus coordinator-local borrowed associations.

    Args:
        target: Detached root target passed to the ordinary scheduler.

    Side Effects:
        :meth:`validate` reads native function, mapping, and class dictionary
        slots only. It never invokes a body, constructor, descriptor, or hook.
    """

    target: InspectionTarget
    _associations: tuple[_LiveAssociation, ...]
    _target_index: object = field(init=False, compare=False, repr=False)
    _carrier_ids: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Build immutable local identity indexes from bounded associations."""

        targets: dict[str, "CodeTarget"] = {}
        carriers: set[int] = set()
        for association in self._associations:
            if association.target_id in targets:
                raise ValueError("inspection capture associations are invalid")
            targets[association.target_id] = association.target
            carriers.update(
                id(value)
                for value in (
                    association.target.original,
                    association.target.callable,
                    association.target.descriptor,
                )
                if value is not None
            )
        expected_ids = tuple(
            record.target_id for record in self.target.snapshot.records
        )
        if (
            len(targets) != len(expected_ids)
            or any(target_id not in targets for target_id in expected_ids)
        ):
            raise ValueError("inspection capture associations are invalid")
        object.__setattr__(self, "_target_index",
                           types.MappingProxyType(targets))
        object.__setattr__(self, "_carrier_ids", frozenset(carriers))

    def validate(self) -> None:
        """Reject observed code, relevant binding, or class-layout drift.

        Raises:
            InvalidTargetError: If a captured proof-relevant identity changed.
        """

        for association in self._associations:
            callable_ = association.target.callable
            instance_guard = association.callable_instance
            if instance_guard is not None:
                mro = _mro(instance_guard.owner)
                if (
                    type(instance_guard.instance) is not instance_guard.owner
                    or mro is None
                    or len(mro) != len(instance_guard.owner_mro)
                    or any(
                        actual is not expected
                        for actual, expected in zip(
                            mro, instance_guard.owner_mro, strict=True
                        )
                    )
                    or _raw_member(instance_guard.owner, "__call__")
                    is not instance_guard.descriptor
                ):
                    raise InvalidTargetError("inspection target changed")
            if (association.code_id is not None
                    and type(callable_) is types.FunctionType):
                if (id(_function_slot(callable_, "__code__"))
                        != association.code_id):
                    raise InvalidTargetError("inspection target changed")
            for binding in association.bindings:
                value, found = _binding_value(binding.function, binding.scope,
                                              binding.name)
                if not found or value is not binding.value:
                    raise InvalidTargetError("inspection target changed")
            for member in association.members:
                if (_module_namespace(member.module).get(
                        member.name, _MISSING) is not member.value):
                    raise InvalidTargetError("inspection target changed")
            for class_guard in association.classes:
                mro = _mro(class_guard.cls)
                if (mro is None or len(mro) != len(class_guard.bases)
                        or any(actual is not expected for actual, expected in
                               zip(mro, class_guard.bases, strict=True))):
                    raise InvalidTargetError("inspection target changed")
                for name, expected in class_guard.members:
                    value = _raw_member(class_guard.cls, name)
                    if value is None:
                        value = _MISSING
                    if value is not expected:
                        raise InvalidTargetError("inspection target changed")

    def local_target(self, target_id: str) -> "CodeTarget | None":
        """
        Return one coordinator-local borrowed target by projection identifier.

                Args:
                    target_id: Projection-local target identifier from this
                    capture's
                        detached snapshot.

                Returns:
                    The exact borrowed :class:`CodeTarget` while this capture
                    remains
                    live, or ``None`` when ``target_id`` is absent or not an
                    exact
                    string. The returned handle is coordinator-local and must
                    not be
                    persisted or returned as detached analysis data.

                Raises:
                    None. An unavailable identifier returns ``None`` rather
                    than
                    reconstructing or normalizing a target.

                Side Effects:
                    None. Lookup reads the capture-local immutable index and
                    never
                    invokes target code, descriptors, reflection hooks, or
                    imports.
        """

        if type(target_id) is not str:
            return None
        return self._target_index.get(target_id)  # type: ignore[union-attr]

    def _has_local_carrier(self, carrier: object) -> bool:
        """
        Return whether an exact retained carrier identity belongs to capture.
        """

        return id(carrier) in self._carrier_ids


def _mro(cls: type) -> tuple[type, ...] | None:
    """Return a bounded raw MRO without metaclass lookup."""

    value = _type_slot(cls, "__mro__")
    if (type(value) is not tuple or len(value) > _MAX_MRO or any(
            type(base) is not type and not isinstance(base, type)
            for base in value)):
        return None
    return value


def _raw_member(cls: type, name: str) -> object | None:
    """Look up one class member through raw dictionaries and the bounded MRO.
    """

    mro = _mro(cls)
    if mro is None:
        return None
    for base in mro:
        namespace = _type_slot(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return None


def _binding_value(function: types.FunctionType, scope: str,
                   name: str) -> tuple[object | None, bool]:
    """Read a named native binding without copying mappings."""

    if scope == "global":
        namespace = types.FunctionType.__dict__["__globals__"].__get__(
            function, types.FunctionType)
        return namespace.get(name), name in namespace
    code = _function_slot(function, "__code__")
    cells = _function_slot(function, "__closure__")
    if type(code) is not types.CodeType or type(cells) is not tuple:
        return None, False
    for free_name, cell in zip(code.co_freevars, cells, strict=True):
        if free_name == name:
            try:
                return cell.cell_contents, True
            except ValueError:
                return None, False
    return None, False


def _projection_info(info: "TargetInfo") -> "TargetInfo":
    """Drop source and caller-path metadata before snapshot construction."""

    from .targets import TargetInfo

    return TargetInfo(
        info.kind,
        info.name,
        info.module,
        info.qualname,
        info.owner_module,
        info.owner_qualname,
        info.descriptor_kind,
        None,
        None,
        None,
    )


def _bounded_tree(source: str) -> ast.Module | None:
    """Parse bounded source before constructing AST visitors."""

    try:
        tree = ast.parse(source)
    except (MemoryError, RecursionError, SyntaxError, ValueError):
        return None
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
            return None
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return tree


def _function_source(function: types.FunctionType,
                     remaining: int) -> tuple["SourceInfo | None", int, bool]:
    """Read a function source with byte and AST ceilings first."""

    from .source import SourceInfo

    code = _function_slot(function, "__code__")
    if type(code) is not types.CodeType or type(code.co_filename) is not str:
        return None, 0, True
    if remaining < 1:
        return None, 0, True
    try:
        with open(code.co_filename, "rb") as source_file:
            raw = source_file.read(min(_MAX_SOURCE_BYTES, remaining) + 1)
    except OSError:
        return None, 0, True
    if len(raw) > min(_MAX_SOURCE_BYTES, remaining):
        return None, 0, True
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return None, len(raw), True
    tree = _bounded_tree(text)
    if tree is None:
        return None, len(raw), True
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == code.co_name):
            decorated = min(
                (item.lineno for item in node.decorator_list),
                default=node.lineno,
            )
            if (node.lineno == code.co_firstlineno
                    or decorated == code.co_firstlineno):
                candidates.append(node)
    if (len(candidates) != 1
            or type(getattr(candidates[0], "end_lineno", None)) is not int):
        return None, len(raw), True
    node = candidates[0]
    start = min((item.lineno for item in node.decorator_list),
                default=node.lineno)
    lines = text.splitlines(keepends=True)
    return (
        SourceInfo(
            textwrap.dedent("".join(lines[start - 1:node.end_lineno])),
            code.co_filename,
            start,
        ),
        len(raw),
        False,
    )


def _has_excluded_eager_work(expressions: tuple[ast.expr | None, ...]) -> bool:
    """
    Return whether excluded definition metadata can execute user behavior.
    """

    eager_nodes = (
        ast.Attribute,
        ast.Await,
        ast.BinOp,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.DictComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.ListComp,
        ast.NamedExpr,
        ast.SetComp,
        ast.Subscript,
        ast.UnaryOp,
        ast.Yield,
        ast.YieldFrom,
    )
    return any(
        isinstance(child, eager_nodes)
        for expression in expressions
        if expression is not None
        for child in ast.walk(expression)
    )


class _ScopeFacts(ast.NodeVisitor):
    """Collect bindings while refusing dormant nested execution scopes."""

    def __init__(self, body: ast.FunctionDef | ast.AsyncFunctionDef,
                 call_limit: int) -> None:
        """Initialize names, declarations, and unambiguous aliases."""

        self.locals: set[str] = {
            argument.arg
            for argument in (
                *body.args.posonlyargs,
                *body.args.args,
                *body.args.kwonlyargs,
            )
        }
        if body.args.vararg is not None:
            self.locals.add(body.args.vararg.arg)
        if body.args.kwarg is not None:
            self.locals.add(body.args.kwarg.arg)
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        self.writes: dict[str, int] = {}
        self.aliases: dict[str, tuple[ast.expr, int] | None] = {}
        self.calls: list[tuple[ast.Call, frozenset[str]]] = []
        self.call_limit = call_limit
        self.overflow = False
        self.incomplete = False
        self._conditional = 0
        self._comprehension_locals: list[set[str]] = []
        for statement in body.body:
            self.visit(statement)
        self.locals -= self.globals | self.nonlocals

    def _write(self, name: str) -> None:
        """Register one scope-wide local write without resolving its value."""

        self.locals.add(name)
        self.writes[name] = self.writes.get(name, 0) + 1

    def visit_Name(self, node: ast.Name) -> None:
        """Collect assignment and deletion bindings in function scope."""

        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._write(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        """Avoid annotation names as invocation-body binding evidence."""

    def visit_Global(self, node: ast.Global) -> None:
        """Record explicit module bindings before local resolution."""

        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record enclosing-closure bindings before local resolution."""

        self.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Treat exception targets as scope-wide local bindings."""

        if node.name is not None:
            self._write(node.name)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        """Treat imports as local bindings, not imported runtime values."""

        self._write(node.asname or node.name.split(".", 1)[0])

    def visit_Assign(self, node: ast.Assign) -> None:
        """Record a simple direct alias candidate while walking RHS calls."""

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.aliases[node.targets[0].id] = (None if self._conditional else
                                                (node.value, node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Record one simple annotated alias candidate without annotations."""

        if isinstance(node.target, ast.Name) and node.value is not None:
            self.aliases[node.target.id] = (None if self._conditional else
                                            (node.value, node.lineno))
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Collect direct calls in source order, including argument calls."""

        if len(self.calls) >= self.call_limit:
            self.overflow = True
            return
        self.calls.append((node, self._comprehension_bindings()))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Bind nested functions while flagging excluded eager metadata."""

        self._write(node.name)
        if (
            node.decorator_list
            or _has_excluded_eager_work(
                (
                    *node.args.defaults,
                    *node.args.kw_defaults,
                    *(argument.annotation for argument in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )),
                    (None if node.args.vararg is None
                     else node.args.vararg.annotation),
                    (None if node.args.kwarg is None
                     else node.args.kwarg.annotation),
                    node.returns,
                )
            )
            or getattr(node, "type_params", ())
        ):
            self.incomplete = True

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Bind nested classes while excluding their eager definition regions.
        """

        self._write(node.name)
        # A class statement executes its header and body, both outside this
        # direct-call grammar, even when no body call is collected.
        self.incomplete = True

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Keep nested lambda bodies outside the selected invocation body."""

    def _conditional_visit(self, node: ast.AST) -> None:
        """Visit a control-flow region while rejecting conditional aliases."""

        self._conditional += 1
        try:
            self.generic_visit(node)
        finally:
            self._conditional -= 1

    visit_If = _conditional_visit
    visit_For = _conditional_visit
    visit_AsyncFor = _conditional_visit
    visit_While = _conditional_visit
    visit_With = _conditional_visit
    visit_AsyncWith = _conditional_visit
    visit_Try = _conditional_visit
    visit_TryStar = _conditional_visit

    def visit_Match(self, node: ast.Match) -> None:
        """Bind pattern captures while treating cases as conditional."""

        self.visit(node.subject)
        for case in node.cases:
            for pattern in ast.walk(case.pattern):
                name = getattr(pattern, "name", None)
                if type(name) is str:
                    self._write(name)
                rest = getattr(pattern, "rest", None)
                if type(rest) is str:
                    self._write(rest)
            if case.guard is not None:
                self._conditional_visit(case.guard)
            for statement in case.body:
                self._conditional_visit(statement)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit calls in a comprehension without leaking targets to locals."""

        self._visit_comprehension(node)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp

    def _visit_comprehension(self, node: ast.AST) -> None:
        """Visit eager comprehension work under its isolated binding scope."""

        generators = getattr(node, "generators", ())
        if not generators:
            self.incomplete = True
            return
        # Python evaluates the outer iterable in the enclosing scope before the
        # comprehension's implicit function receives its first target binding.
        self.visit(generators[0].iter)
        bindings: set[str] = set()
        self._comprehension_locals.append(bindings)
        try:
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                self._bind_comprehension_target(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)  # type: ignore[attr-defined]
        finally:
            self._comprehension_locals.pop()

    def _bind_comprehension_target(self, target: ast.expr) -> None:
        """
        Record only implicit-local target names without visiting outer scope.
        """

        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(
                    child.ctx, (ast.Store, ast.Del)):
                self._comprehension_locals[-1].add(child.id)

    def _comprehension_bindings(self) -> frozenset[str]:
        """Return active implicit-scope names for one recorded call fact."""

        return frozenset(name for bindings in self._comprehension_locals
                         for name in bindings)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """
        Keep the eager first iterable while excluding deferred generator work.
        """

        if node.generators:
            self.visit(node.generators[0].iter)
        self.incomplete = True
        self.overflow = True


def _function_body(
    source: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return one selected invocation body from isolated function source."""

    tree = _bounded_tree(source)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _descriptor_function(value: object | None) -> types.FunctionType | None:
    """Return a raw Python function from one admitted unbound descriptor."""

    if type(value) is types.FunctionType:
        return value
    if (type(value) in (staticmethod, classmethod)
            and type(value.__func__) is types.FunctionType):
        return value.__func__
    return None


def _receiver_member(owner: type,
                     name: str) -> tuple[object | None, _ClassGuard | None]:
    """Prove a non-shadowable receiver member without reading instances."""

    mro = _mro(owner)
    if mro is None:
        return None, None
    for base in mro:
        namespace = _type_slot(base, "__dict__")
        slots = namespace.get("__slots__", ())
        if ((slots is not None and
             (type(slots) is not tuple or len(slots) != 0))
                or "__dict__" in namespace or
            ("__getattribute__" in namespace
             and namespace["__getattribute__"] is not object.__getattribute__)
                or "__getattr__" in namespace):
            return None, None
    value = _raw_member(owner, name)
    if _descriptor_function(value) is None:
        return None, None
    from .targets import DescriptorTarget, _normalize_target

    return _normalize_target(DescriptorTarget(owner, name),
                             include_source=False), _class_guard(
                                 owner,
                                 (name, "__getattribute__", "__getattr__"))


def _class_guard(cls: type, names: tuple[str, ...]) -> _ClassGuard | None:
    """Capture bounded MRO and raw member identities used by a class proof."""

    mro = _mro(cls)
    if mro is None:
        return None
    members = tuple((
        name,
        (_raw_member(cls, name) if _raw_member(cls, name
                                               ) is not None else _MISSING),
    ) for name in names)
    return _ClassGuard(cls, mro, members)


def _callable_instance_guard(
    target: "CodeTarget",
) -> _CallableInstanceGuard | None:
    """Capture native callable-instance dispatch facts without binding it."""

    if target.info.kind != "callable_instance":
        return None
    instance = target.original
    owner = target.owner
    descriptor = target.descriptor
    if (
        instance is None
        or owner is None
        or descriptor is None
        or type(instance) is not owner
    ):
        return None
    mro = _mro(owner)
    if mro is None or _raw_member(owner, "__call__") is not descriptor:
        return None
    return _CallableInstanceGuard(instance, owner, mro, descriptor)


def _identity_unique(
    values: list[object] | tuple[object, ...],
) -> tuple[object, ...]:
    """Deduplicate guard carriers without hash or equality operations."""

    result: list[object] = []
    identifiers: set[int] = set()
    for value in values:
        identifier = id(value)
        if identifier not in identifiers:
            identifiers.add(identifier)
            result.append(value)
    return tuple(result)


def _resolve_expression(
    expression: ast.expr,
    facts: _ScopeFacts,
    function: types.FunctionType,
    owner: type | None,
    receiver_name: str | None,
    line: int,
    comprehension_locals: frozenset[str] = frozenset(),
) -> tuple[object | None, list[_BindingGuard], list[_MemberGuard],
           list[_ClassGuard]]:
    """Resolve one supported direct binding expression or return unresolved."""

    guards: list[_BindingGuard] = []
    members: list[_MemberGuard] = []
    classes: list[_ClassGuard] = []

    def name_value(name: str) -> object | None:
        """Read a name through its proven scope, never a guessed global."""

        if name in comprehension_locals:
            return None
        if name in facts.locals:
            alias = facts.aliases.get(name)
            if (facts.writes.get(name) != 1 or alias is None
                    or alias[1] >= line):
                return None
            value, nested_guards, nested_members, nested_classes = (
                _resolve_expression(alias[0], facts, function, owner,
                                    receiver_name, alias[1],
                                    comprehension_locals))
            guards.extend(nested_guards)
            members.extend(nested_members)
            classes.extend(nested_classes)
            return value
        code = _function_slot(function, "__code__")
        closure_names = (code.co_freevars if type(code) is types.CodeType else
                         ())
        scope = ("closure" if name in facts.nonlocals or name in closure_names
                 else "global")
        if facts.writes.get(name, 0):
            return None
        value, found = _binding_value(function, scope, name)
        if found:
            guards.append(_BindingGuard(function, scope, name, value))
            return value
        if scope == "global":
            import builtins

            value = builtins.__dict__.get(name, _MISSING)
            return None if value is _MISSING else value
        return None

    if isinstance(expression, ast.Name):
        return name_value(expression.id), guards, members, classes
    if not isinstance(expression, ast.Attribute):
        return None, guards, members, classes
    if (owner is not None and isinstance(expression.value, ast.Name)
            and expression.value.id == receiver_name
            and facts.writes.get(receiver_name, 0) == 0):
        value, guard = _receiver_member(owner, expression.attr)
        if guard is not None:
            classes.append(guard)
        return value, guards, members, classes
    base, nested_guards, nested_members, nested_classes = _resolve_expression(
        expression.value, facts, function, owner, receiver_name, line,
        comprehension_locals)
    guards.extend(nested_guards)
    members.extend(nested_members)
    classes.extend(nested_classes)
    if isinstance(base, types.ModuleType):
        value = _module_namespace(base).get(expression.attr, _MISSING)
        members.append(_MemberGuard(base, expression.attr, value))
        return (
            None if value is _MISSING else value,
            guards,
            members,
            classes,
        )
    if isinstance(base, type):
        if base is object and expression.attr == "__new__":
            return object.__new__, guards, members, classes
        if base is object and expression.attr == "__init__":
            return object.__init__, guards, members, classes
        class_guard = _class_guard(base, (expression.attr, ))
        if class_guard is not None:
            classes.append(class_guard)
        from .targets import DescriptorTarget, _normalize_target

        try:
            value = _normalize_target(DescriptorTarget(base, expression.attr),
                                      include_source=False)
        except InvalidTargetError:
            value = None
        return value, guards, members, classes
    return None, guards, members, classes


def _calls(
    source: str,
    function: types.FunctionType | None,
    owner: type | None,
    remaining_calls: int,
) -> tuple[
        list[object | None],
        bool,
        tuple[_BindingGuard, ...],
        tuple[_MemberGuard, ...],
        tuple[_ClassGuard, ...],
]:
    """Collect conservative direct call values from one invocation body."""

    if function is None:
        return [], True, (), (), ()
    body = _function_body(source)
    if body is None:
        return [], True, (), (), ()
    facts = _ScopeFacts(body, remaining_calls)
    positional = (*body.args.posonlyargs, *body.args.args)
    receiver_name = (positional[0].arg
                     if owner is not None and positional else None)
    values: list[object | None] = []
    guards: list[_BindingGuard] = []
    members: list[_MemberGuard] = []
    classes: list[_ClassGuard] = []
    incomplete = False
    for call, comprehension_locals in facts.calls:
        if (isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "object"
                and call.func.attr in ("__new__", "__init__")):
            # These two built-in construction operations have explicit terminal
            # summaries; they are not ordinary extension-body dependencies.
            value, call_guards, call_members, call_classes = (
                _resolve_expression(
                    call.func,
                    facts,
                    function,
                    owner,
                    receiver_name,
                    call.lineno,
                    comprehension_locals,
                ))
            if value is object.__new__ or value is object.__init__:
                continue
        value, call_guards, call_members, call_classes = _resolve_expression(
            call.func,
            facts,
            function,
            owner,
            receiver_name,
            call.lineno,
            comprehension_locals,
        )
        values.append(value)
        guards.extend(call_guards)
        members.extend(call_members)
        classes.extend(call_classes)
        incomplete |= value is None
    return (
        values,
        incomplete or facts.incomplete or facts.overflow,
        tuple(guards),
        tuple(members),
        tuple(classes),
    )


def _source_only_calls(
    source: str,
    remaining_calls: int,
) -> tuple[list[object | None], bool]:
    """Project unsupported source-only calls as bounded unresolved evidence."""

    body = _function_body(source)
    if body is None:
        return [], True
    facts = _ScopeFacts(body, remaining_calls)
    count = len(facts.calls)
    return [None] * count, count > 0 or facts.incomplete or facts.overflow


def _constructor_values(
    cls: type,
) -> tuple[list[object], bool, tuple[_ClassGuard, ...]]:
    """Return conservative construction targets and coverage for one class."""

    mro = _mro(cls)
    if mro is None:
        return [], True, ()
    members = ("__new__", "__init__", "__call__")
    guard = _ClassGuard(
        cls,
        mro,
        tuple((
            name,
            (_raw_member(cls, name)
             if _raw_member(cls, name) is not None else _MISSING),
        ) for name in members),
    )
    metaclass = type(cls)
    if metaclass is not type:
        call = _descriptor_function(_raw_member(metaclass, "__call__"))
        return ([call] if call is not None else []), True, (guard, )
    values: list[object] = []
    incomplete = False
    for name, builtin in (
        ("__new__", object.__new__),
        ("__init__", object.__init__),
    ):
        member = _raw_member(cls, name)
        function = _descriptor_function(member)
        if function is not None:
            values.append(function)
        elif member is not builtin:
            incomplete = True
    return values, incomplete, (guard, )


def _target_key(target: "CodeTarget") -> int:
    """Return native identity without caller equality, hashing, or repr."""

    if target.original is not None:
        return id(target.original)
    if target.descriptor is not None:
        return id(target.owner) ^ id(target.descriptor)
    return id(target)


def _oversize_source_target(target: object) -> "CodeTarget | None":
    """Project an oversized explicit source root without parsing it."""

    from .targets import CodeTarget, SourceTarget, TargetInfo

    if (type(target) is not SourceTarget
            or _source_within_bounds(target.source)):
        return None
    return CodeTarget(
        TargetInfo(
            "source",
            target.name,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _source_root(target: object) -> "CodeTarget | None":
    """Admit a source root without parsing before capture limits apply."""

    from .source import SourceInfo
    from .targets import CodeTarget, SourceTarget, TargetInfo

    if type(target) is not SourceTarget:
        return None
    return CodeTarget(
        TargetInfo(
            "source",
            target.name,
            None,
            None,
            None,
            None,
            None,
            None,
            target.start_line,
            None,
        ),
        None,
        None,
        None,
        None,
        SourceInfo(target.source, target.filename, target.start_line),
        None,
    )


def capture_inspection(target: CodeTargetInput) -> InspectionCapture:
    """
    Passively capture a bounded source-free candidate closure.

        Args:
            target: A supported live/static target. Existing
                :class:`InspectionTarget` values are rejected because a
                detached
                target has no borrowed handle.

        Returns:
            A detached root snapshot and coordinator-local identity
            associations.

        Raises:
            InvalidTargetError: If target admission fails or a detached target
            is
                recaptured. Safe source, AST, binding, MRO, fact, and target
                ceilings
                produce incomplete coverage rather than a complete result.

        Side Effects:
            Reads bounded source files and native storage. It never executes
            target
            code, constructors, descriptors, imports, iteration, or hooks.
    """

    return _capture_inspection(target)


def _capture_inspection(
    target: CodeTargetInput,
    *,
    owner_facts: InspectionOwnerFacts | None = None,
) -> InspectionCapture:
    """
    Capture one bounded closure with an optional owner-proven bridge.

        Args:
            target: A supported live/static target.
            owner_facts: Exact internal owner-proven related targets. Generic
            capture
                validates them under its ordinary bounds without inferring
                decorator
                relationships.

        Returns:
            A detached root snapshot and coordinator-local identity
            associations.

        Raises:
            InvalidTargetError: If target admission or relationship facts fail.
            Safe
                capture ceilings produce incomplete coverage rather than
                completeness.

        Side Effects:
            Reads bounded source files and native storage without executing
            code.
    """

    if type(target) is InspectionTarget:
        raise InvalidTargetError("inspection targets cannot be recaptured")
    from .targets import CodeTarget, _normalize_target

    root = _oversize_source_target(target)
    if root is None:
        root = _source_root(target)
    if root is None:
        root = _normalize_target(target, include_source=False)
    if type(root) is InspectionTarget:
        raise InvalidTargetError("inspection targets cannot be recaptured")
    if owner_facts is not None:
        if type(owner_facts) is not InspectionOwnerFacts:
            raise TypeError("inspection owner facts are invalid")
        if owner_facts.root is not target:
            raise InvalidTargetError(
                "inspection owner facts have another root")
    identifiers: dict[int, str] = {_target_key(root): "t0000"}
    targets: list[CodeTarget] = [root]
    owner_related: list[CodeTarget] = []
    opaque_keys: set[int] = set()
    if owner_facts is not None:
        for value in owner_facts.related:
            try:
                candidate = _normalize_target(value, include_source=False)
            except InvalidTargetError as exc:
                raise InvalidTargetError(
                    "inspection owner relationship target is invalid"
                ) from exc
            if type(candidate) is InspectionTarget:
                raise InvalidTargetError(
                    "inspection owner relationship target is invalid"
                )
            key = _target_key(candidate)
            if key not in identifiers:
                if len(targets) >= _MAX_TARGETS:
                    raise InvalidTargetError(
                        "inspection owner relationships exceed target limit"
                    )
                identifiers[key] = f"t{len(targets):04d}"
                targets.append(candidate)
            owner_related.append(candidate)
        opaque_keys = {
            _target_key(_normalize_target(value, include_source=False))
            for value in owner_facts.opaque
        }
    raw_calls: dict[
        str,
        tuple[
            list[object | None],
            bool,
            tuple[_BindingGuard, ...],
            tuple[_MemberGuard, ...],
            tuple[_ClassGuard, ...],
        ],
    ] = {}
    remaining_source = _MAX_SOURCE_TOTAL
    remaining_calls = _MAX_CALLS
    index = 0
    while index < len(targets):
        current = targets[index]
        current_id = identifiers[_target_key(current)]
        values: list[object | None] = []
        incomplete = False
        guards: tuple[_BindingGuard, ...] = ()
        member_guards: tuple[_MemberGuard, ...] = ()
        class_guards: tuple[_ClassGuard, ...] = ()
        function = (current.callable
                    if type(current.callable) is types.FunctionType else None)
        if _target_key(current) in opaque_keys:
            incomplete = False
        elif function is not None:
            source, consumed, unavailable = _function_source(
                function, remaining_source)
            remaining_source -= consumed
            if unavailable or source is None:
                incomplete = True
            else:
                values, incomplete, guards, member_guards, class_guards = (
                    _calls(source.source, function, current.owner,
                           remaining_calls))
        elif current.source is not None:
            if not _source_within_bounds(current.source.source):
                raw_calls[current_id] = (values, True, guards, member_guards,
                                         class_guards)
                index += 1
                continue
            raw = current.source.source.encode("utf-8")
            if (len(raw) > _MAX_SOURCE_BYTES or len(raw) > remaining_source
                    or _bounded_tree(current.source.source) is None):
                incomplete = True
            else:
                remaining_source -= len(raw)
                values, incomplete = _source_only_calls(
                    current.source.source, remaining_calls)
        elif isinstance(current.original, type):
            values, incomplete, class_guards = _constructor_values(
                current.original)
        else:
            incomplete = True
        if current_id == "t0000" and owner_related:
            values.extend(owner_related)
        if len(values) > remaining_calls:
            raise InvalidTargetError("inspection capture call limit exceeded")
        remaining_calls -= len(values)
        raw_calls[current_id] = (
            values,
            incomplete,
            guards,
            member_guards,
            class_guards,
        )
        for value in values:
            if value is None or not (type(value) is CodeTarget
                                     or type(value) is types.FunctionType
                                     or type(value) is types.MethodType
                                     or isinstance(value, type)):
                continue
            if type(value) is CodeTarget:
                candidate = value
            else:
                try:
                    candidate = _normalize_target(value, include_source=False)
                except InvalidTargetError:
                    incomplete = True
                    continue
            if type(candidate) is InspectionTarget:
                incomplete = True
                continue
            key = _target_key(candidate)
            if key not in identifiers:
                if len(targets) >= _MAX_TARGETS:
                    incomplete = True
                    continue
                identifiers[key] = f"t{len(targets):04d}"
                targets.append(candidate)
        if incomplete != raw_calls[current_id][1]:
            raw_calls[current_id] = (
                values,
                incomplete,
                guards,
                member_guards,
                class_guards,
            )
        index += 1
    records: list[InspectionRecord] = []
    associations: list[_LiveAssociation] = []
    value_ids: dict[int, str] = {}
    for current in targets:
        values, _, _, _, _ = raw_calls[identifiers[_target_key(current)]]
        for value in values:
            if value is None:
                continue
            if type(value) is CodeTarget:
                value_ids[id(value)] = identifiers[_target_key(value)]
            else:
                try:
                    candidate = _normalize_target(value, include_source=False)
                except InvalidTargetError:
                    continue
                if type(candidate) is not InspectionTarget:
                    value_ids[id(value)] = identifiers[_target_key(candidate)]
    for current in targets:
        current_id = identifiers[_target_key(current)]
        values, incomplete, guards, member_guards, class_guards = raw_calls[
            current_id]
        calls = tuple(
            InspectionCall(
                value_ids.get(id(value)) if value is not None else None)
            for value in values)
        records.append(
            InspectionRecord(current_id, _projection_info(current.info), calls,
                             incomplete))
        callable_ = current.callable
        callable_instance = _callable_instance_guard(current)
        if (current.info.kind == "callable_instance"
                and callable_instance is None):
            raise InvalidTargetError("unsupported callable")
        associations.append(
            _LiveAssociation(
                current_id,
                current,
                (id(_function_slot(callable_, "__code__"))
                 if type(callable_) is types.FunctionType else None),
                _identity_unique(guards),
                _identity_unique(member_guards),
                _identity_unique(class_guards),
                callable_instance,
            ))
    snapshot = InspectionSnapshot(_VERSION, "t0000", tuple(records))
    return InspectionCapture(InspectionTarget(snapshot, snapshot.root_id),
                             tuple(associations))


__all__ = [
    "InspectionCapture",
    "InspectionTarget",
    "capture_inspection",
]
