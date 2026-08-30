"""Typed, serializable paths through canonical definition graph values.

V2 CDefs address constructor fields with ``Parameter`` segments, while legacy
V1 CDefs retain invocation-oriented ``Arg`` and ``Kwarg`` segments. These
types describe paths only; resolving a path is owned by graph-value utilities.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable


GRAPH_PATH_SCHEMA_VERSION = 3
_SUPPORTED_GRAPH_PATH_SCHEMA_VERSIONS = frozenset((1, 2, GRAPH_PATH_SCHEMA_VERSION))


class GraphPathError(Exception):
    """Raised when a graph path is malformed, unsupported, or cannot resolve."""


QueryPathError = GraphPathError


def _validate_segment_field(value: Any, field: str, expected_type: type, kind: str) -> None:
    if not isinstance(value, expected_type):
        raise QueryPathError(
            f"Graph path {kind} field {field!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )


def _validate_segment_index(value: Any, field: str, kind: str) -> None:
    _validate_segment_field(value, field, int, kind)
    if type(value) is not int or value < 0:
        raise QueryPathError(f"Graph path {kind} field {field!r} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class Kwarg:
    """Legacy V1 keyword-call path segment.

    Args:
        name: Persisted keyword argument name.

    Raises:
        QueryPathError: If ``name`` is not a string.
    """
    name: str

    def __post_init__(self) -> None:
        _validate_segment_field(self.name, "name", str, "kwarg")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Parameter:
    """A persisted semantic constructor parameter path segment.

    V2 concrete definitions use this segment instead of invocation-specific
    ``Arg`` or ``Kwarg`` segments, so a path remains stable across equivalent
    positional and keyword call spellings.

    Attributes:
        name: The persisted semantic constructor parameter name.

    Raises:
        QueryPathError: If ``name`` is not a string.
    """

    name: str

    def __post_init__(self) -> None:
        _validate_segment_field(self.name, "name", str, "parameter")

    def __str__(self) -> str:
        return f"@param({json.dumps(self.name)})"


@dataclass(frozen=True, slots=True)
class Arg:
    """Legacy V1 positional-call path segment.

    Args:
        index: Zero-based position in the persisted raw argument tuple.

    Raises:
        QueryPathError: If ``index`` is not a non-negative integer.
    """
    index: int

    def __post_init__(self) -> None:
        _validate_segment_index(self.index, "index", "arg")

    def __str__(self) -> str:
        return f"args[{self.index}]"


@dataclass(frozen=True, slots=True)
class Index:
    """Sequence-index path segment.

    Args:
        index: Zero-based index in the current sequence value.

    Raises:
        QueryPathError: If ``index`` is not a non-negative integer.
    """
    index: int

    def __post_init__(self) -> None:
        _validate_segment_index(self.index, "index", "index")

    def __str__(self) -> str:
        return f"[{self.index}]"


@dataclass(frozen=True, slots=True)
class Key:
    """Mapping-key path segment distinct from a sequence index.

    Args:
        key: Exact mapping key to resolve.
    """
    key: Any

    def __str__(self) -> str:
        if isinstance(self.key, str):
            return f"[{self.key!r}]"
        return f"[@key({self.key!r})]"


@dataclass(frozen=True, slots=True)
class SetMember:
    """Stable set-member path segment.

    Args:
        fingerprint: Stable fingerprint of the selected member.
        ordinal: Zero-based occurrence among colliding fingerprints.

    Raises:
        QueryPathError: If ``fingerprint`` is not a string or ``ordinal`` is
            not a non-negative integer.
    """
    fingerprint: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        _validate_segment_field(self.fingerprint, "fingerprint", str, "set_member")
        _validate_segment_index(self.ordinal, "ordinal", "set_member")

    def __str__(self) -> str:
        return f'@set("{self.fingerprint}", {self.ordinal})'


PathSegment = Parameter | Kwarg | Arg | Index | Key | SetMember
GraphPathLike = str | Iterable[PathSegment | str | int] | "GraphPath"


@dataclass(frozen=True, slots=True, eq=False)
class GraphPath:
    """Immutable ordered sequence of typed canonical-value path segments.

    Args:
        segments: Typed path segments from the root to a child value. The empty
            tuple represents the root.
    """
    segments: tuple[PathSegment, ...] = ()

    def __iter__(self):
        """Iterate typed path segments from root to leaf.

        Returns:
            An iterator over ``PathSegment`` instances.
        """
        return iter(self.segments)

    def __len__(self) -> int:
        """Return the number of segments in this path.

        Returns:
            Zero for the root path, otherwise the path depth.
        """
        return len(self.segments)

    def __bool__(self) -> bool:
        """Report whether this path addresses a value below the root.

        Returns:
            ``False`` for the root path and ``True`` otherwise.
        """
        return bool(self.segments)

    def __getitem__(self, item):
        """Return a segment or path slice.

        Args:
            item: Integer index or slice into the typed segments.

        Returns:
            A ``PathSegment`` for an index or a new ``GraphPath`` for a slice.

        Raises:
            IndexError: If an integer index is outside this path.
        """
        if isinstance(item, slice):
            return type(self)(self.segments[item])
        return self.segments[item]

    def __hash__(self) -> int:
        return hash(self.segments)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphPath):
            return self.segments == other.segments
        return False

    def child(self, segment: PathSegment | str | int) -> "GraphPath":
        """Return a new path with one normalized child segment appended.

        Args:
            segment: Typed segment, legacy string keyword, or integer index.

        Returns:
            A new ``GraphPath``.

        Raises:
            QueryPathError: If ``segment`` is unsupported.
        """
        return type(self)(self.segments + (_normalize_segment(segment),))

    def join(self, other: GraphPathLike) -> "GraphPath":
        """Return a new path formed by appending another normalized path.

        Args:
            other: Typed, textual, or iterable path representation.

        Returns:
            A combined ``GraphPath``.

        Raises:
            QueryPathError: If ``other`` cannot be normalized.
        """
        norm = normalize_path(other)
        return type(self)(self.segments + norm.segments)

    @property
    def parent(self) -> "GraphPath":
        """Return the path one segment nearer the root.

        Returns:
            A new parent path, or the root path itself when already at root.
        """
        if not self.segments:
            return self
        return type(self)(self.segments[:-1])

    @property
    def name(self) -> str | None:
        """Return the display form of the last segment, if any.

        Returns:
            A segment display string, or ``None`` for the root path.
        """
        if not self.segments:
            return None
        return str(self.segments[-1])

    @property
    def is_root(self) -> bool:
        """Report whether this path denotes the root value.

        Returns:
            ``True`` when this path has no segments.
        """
        return not self.segments

    def startswith(self, other: GraphPathLike) -> bool:
        """Report whether this path begins with another normalized path.

        Args:
            other: Candidate typed, textual, or iterable prefix.

        Returns:
            ``True`` when ``other`` is a prefix of this path.

        Raises:
            QueryPathError: If ``other`` cannot be normalized.
        """
        norm = normalize_path(other)
        return self.segments[:len(norm.segments)] == norm.segments

    def overlaps(self, other: GraphPathLike) -> bool:
        """Report whether either normalized path is a prefix of the other.

        Args:
            other: Candidate typed, textual, or iterable path.

        Returns:
            ``True`` when the paths share one addressed root-to-leaf branch.

        Raises:
            QueryPathError: If ``other`` cannot be normalized.
        """
        norm = normalize_path(other)
        return self.startswith(norm) or norm.startswith(self)

    def relative_to(self, prefix: GraphPathLike) -> "GraphPath":
        """Return this path with a required normalized prefix removed.

        Args:
            prefix: Required leading path.

        Returns:
            The suffix relative to ``prefix``.

        Raises:
            QueryPathError: If this path does not begin with ``prefix``.
        """
        norm = normalize_path(prefix)
        if not self.startswith(norm):
            raise QueryPathError(f"Path {self!s} is not relative to prefix {norm!s}.")
        return type(self)(self.segments[len(norm.segments):])

    def legacy_tuple(self) -> tuple[str | int, ...]:
        """Return the lossy historical tuple display representation.

        Returns:
            A tuple of strings and integers for compatibility callers. Use typed
            paths for identity-sensitive V1/V2 graph processing.
        """
        return tuple(_legacy_segment_value(seg) for seg in self.segments)

    def to_legacy_tuple(self) -> tuple[str | int, ...]:
        """Return ``legacy_tuple()`` under its compatibility API name.

        Returns:
            The historical tuple representation of this path.
        """
        return self.legacy_tuple()

    def legacy_str(self) -> str:
        """Return the slash-delimited historical display representation.

        Returns:
            ``"<root>"`` for root or a compatibility display string.
        """
        if not self.segments:
            return "<root>"
        return "/".join(map(str, self.legacy_tuple()))

    def to_data(self) -> dict[str, Any]:
        """Serialize this path with its current schema version.

        Returns:
            A machine-readable mapping containing ``schema_version`` and typed
            segment data.
        """
        return {
            "schema_version": GRAPH_PATH_SCHEMA_VERSION,
            "segments": [_segment_to_data(seg) for seg in self.segments],
        }

    def to_bytes(self) -> bytes:
        """Encode this path into the canonical tagged ordering byte stream.

        Returns:
            Versioned bytes whose lexicographic order is the graph-authority
            order for paths. The representation distinguishes every segment
            type, including integer mapping keys versus sequence indexes.
        """

        return graph_path_bytes(self)

    @classmethod
    def from_data(cls, data: Any) -> "GraphPath":
        """Deserialize a supported versioned graph path.

        Args:
            data: A schema-versioned mapping or compatible segment iterable.

        Returns:
            The decoded immutable graph path.

        Raises:
            QueryPathError: If the schema version or segment representation is
                missing, malformed, or unsupported.
        """
        segments_data, schema_version = _segments_from_path_data(data)
        return cls(tuple(_segment_from_data(seg_data, schema_version) for seg_data in segments_data))

    def __str__(self) -> str:
        if not self.segments:
            return "$"

        out = "$"
        for seg in self.segments:
            if isinstance(seg, Parameter):
                out += f"[{seg!s}]"
            elif isinstance(seg, Kwarg):
                out += f".{seg.name}"
            elif isinstance(seg, Arg):
                out += f".args[{seg.index}]"
            elif isinstance(seg, Index):
                out += f"[{seg.index}]"
            elif isinstance(seg, Key):
                if isinstance(seg.key, str):
                    out += f"[{seg.key!r}]"
                else:
                    out += f"[@key({seg.key!r})]"
            elif isinstance(seg, SetMember):
                out += f'[@set("{seg.fingerprint}", {seg.ordinal})]'
        return out


DefinitionPath = GraphPath
DefinitionPathLike = GraphPathLike


def normalize_path(path: GraphPathLike = "$") -> GraphPath:
    """Normalize typed, textual, or iterable input to a ``GraphPath``.

    Args:
        path: ``GraphPath``, textual path, or iterable of supported segments.

    Returns:
        The normalized path; an empty path denotes the root.

    Raises:
        QueryPathError: If a segment or textual form is invalid.
    """
    if isinstance(path, GraphPath):
        return path
    if isinstance(path, str):
        return parse_path(path)

    segments: list[PathSegment] = []
    for part in path:
        segments.append(_normalize_segment(part))
    return GraphPath(tuple(segments))


normalize_graph_path = normalize_path


def graph_path_bytes(path: GraphPathLike) -> bytes:
    """Return canonical tagged bytes for a normalized graph path.

    Args:
        path: A typed graph path or supported normalization input.

    Returns:
        A versioned, self-delimiting byte representation suitable for stable
        sorting and graph-record labels.

    Raises:
        QueryPathError: If a path segment or mapping key is unsupported.
    """

    normalized = normalize_path(path)
    out = bytearray(b"DRYML-GRAPH-PATH-3\x00")
    for segment in normalized:
        tag, payload = _segment_bytes(segment)
        out.extend(tag)
        out.extend(len(payload).to_bytes(8, "big"))
        out.extend(payload)
    return bytes(out)


def graph_path_sort_key(path: GraphPathLike) -> bytes:
    """Return the deterministic total-order key for a graph path."""

    return graph_path_bytes(path)


def _segment_bytes(segment: PathSegment) -> tuple[bytes, bytes]:
    if isinstance(segment, Parameter):
        return b"P", segment.name.encode("utf-8")
    if isinstance(segment, Kwarg):
        return b"W", segment.name.encode("utf-8")
    if isinstance(segment, Arg):
        return b"A", _nonnegative_int_bytes(segment.index)
    if isinstance(segment, Index):
        return b"I", _nonnegative_int_bytes(segment.index)
    if isinstance(segment, Key):
        return b"K", canonical_key_bytes(segment.key)
    if isinstance(segment, SetMember):
        return b"S", _frame(segment.fingerprint.encode("utf-8")) + _nonnegative_int_bytes(segment.ordinal)
    raise QueryPathError(f"Unsupported graph path segment {segment!r}.")


def canonical_key_bytes(key: Any) -> bytes:
    """Return canonical tagged bytes for a supported mapping key.

    Args:
        key: A canonical ``str`` or exact ``int`` mapping key.

    Returns:
        A self-delimiting typed key encoding.

    Raises:
        QueryPathError: If ``key`` is not a canonical mapping-key type.
    """

    if type(key) is str:
        return b"s" + _frame(key.encode("utf-8"))
    if type(key) is int:
        sign = b"+" if key >= 0 else b"-"
        return b"i" + sign + _frame(str(abs(key)).encode("ascii"))
    raise QueryPathError(f"Graph path mapping keys must be str or int, got {type(key).__name__}.")


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "big") + payload


def _nonnegative_int_bytes(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise QueryPathError("Canonical graph path indexes must be non-negative integers.")
    return _frame(str(value).encode("ascii"))


def normalize_ctx_path(path: GraphPathLike | None = None) -> GraphPath:
    """Normalize a graph-context path, preserving legacy tuple semantics.

    Args:
        path: Optional path. Tuples of strings and integers use legacy context
            normalization; all other forms use ``normalize_path``.

    Returns:
        A normalized ``GraphPath``.

    Raises:
        QueryPathError: If ``path`` is invalid.
    """
    if path is None:
        return GraphPath()
    if isinstance(path, tuple) and all(isinstance(part, (str, int)) for part in path):
        # Generic graph callers historically stored raw strings/ints.
        return GraphPath(tuple(_normalize_legacy_segment(part) for part in path))
    return normalize_path(path)


def parse_path(text: str) -> GraphPath:
    """Parse textual graph-path syntax into typed segments.

    Args:
        text: Root ``$`` or a path using ``args[i]``, keyword, index, key, set,
            or semantic parameter syntax such as ``$[@param("model")]``.

    Returns:
        The parsed ``GraphPath``.

    Raises:
        QueryPathError: If syntax, quoting, brackets, or segment kinds are
            invalid.
    """
    if text == "" or text == "$":
        return GraphPath()
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        if text.startswith("$["):
            text = text[1:]
        else:
            raise QueryPathError(f"Invalid path syntax {text!r}; expected '$' or '$.'.")

    tokens = _split_path_tokens(text)
    segments: list[PathSegment] = []
    for token in tokens:
        if token == "":
            raise QueryPathError(f"Invalid empty path segment in {text!r}.")
        segments.extend(_parse_token(token))
    return GraphPath(tuple(segments))


def _normalize_segment(part: PathSegment | str | int) -> PathSegment:
    if isinstance(part, (Parameter, Kwarg, Arg, Index, Key, SetMember)):
        return part
    if isinstance(part, str):
        return Kwarg(part)
    if isinstance(part, int):
        return Index(part)
    raise QueryPathError(f"Unsupported path segment {part!r}.")


def _normalize_legacy_segment(part: str | int) -> PathSegment:
    if isinstance(part, int):
        return Index(part)
    return Kwarg(part)


def _legacy_segment_value(seg: PathSegment) -> str | int:
    if isinstance(seg, Parameter):
        return str(seg)
    if isinstance(seg, Kwarg):
        return seg.name
    if isinstance(seg, Arg):
        return f"args[{seg.index}]"
    if isinstance(seg, Index):
        return seg.index
    if isinstance(seg, Key):
        return seg.key if isinstance(seg.key, (str, int)) else repr(seg.key)
    if isinstance(seg, SetMember):
        return str(seg)
    raise TypeError(seg)


def _segment_to_data(seg: PathSegment) -> dict[str, Any]:
    if isinstance(seg, Parameter):
        return {"kind": "parameter", "name": seg.name}
    if isinstance(seg, Kwarg):
        return {"kind": "kwarg", "name": seg.name}
    if isinstance(seg, Arg):
        return {"kind": "arg", "index": seg.index}
    if isinstance(seg, Index):
        return {"kind": "index", "index": seg.index}
    if isinstance(seg, Key):
        return {"kind": "key", "value": seg.key}
    if isinstance(seg, SetMember):
        return {"kind": "set_member", "fingerprint": seg.fingerprint, "ordinal": seg.ordinal}
    raise TypeError(seg)


def _segments_from_path_data(data: Any) -> tuple[Iterable[Mapping[str, Any]], int]:
    if isinstance(data, Mapping):
        version = data.get("schema_version")
        if type(version) is not int or version not in _SUPPORTED_GRAPH_PATH_SCHEMA_VERSIONS:
            raise QueryPathError(f"Unsupported graph path schema version {version!r}.")
        segments = data.get("segments")
        if segments is None:
            raise QueryPathError("Graph path data is missing 'segments'.")
        return segments, version
    return data, GRAPH_PATH_SCHEMA_VERSION


def _segment_from_data(data: Mapping[str, Any], schema_version: int) -> PathSegment:
    if not isinstance(data, Mapping):
        raise QueryPathError(f"Graph path segment data must be a mapping, got {type(data).__name__}.")
    kind = data.get("kind")
    if kind == "parameter":
        if schema_version < 2:
            raise QueryPathError("Semantic parameter segments require graph path schema version 2.")
        return Parameter(_required_segment_field(data, "name", str, kind))
    if kind == "kwarg":
        return Kwarg(_required_segment_field(data, "name", str, kind))
    if kind == "arg":
        return Arg(_required_segment_index(data, "index", kind))
    if kind == "index":
        return Index(_required_segment_index(data, "index", kind))
    if kind == "key":
        if "value" not in data:
            raise QueryPathError("Graph path key segment is missing required field 'value'.")
        return Key(data["value"])
    if kind == "set_member":
        fingerprint = _required_segment_field(data, "fingerprint", str, kind)
        ordinal = data.get("ordinal", 0)
        if type(ordinal) is not int or ordinal < 0:
            raise QueryPathError("Graph path set_member field 'ordinal' must be a non-negative integer.")
        return SetMember(fingerprint, ordinal)
    raise QueryPathError(f"Unknown graph path segment kind {kind!r}.")


def _required_segment_field(data: Mapping[str, Any], field: str, expected_type: type, kind: str) -> Any:
    if field not in data:
        raise QueryPathError(f"Graph path {kind} segment is missing required field {field!r}.")
    value = data[field]
    _validate_segment_field(value, field, expected_type, kind)
    return value


def _required_segment_index(data: Mapping[str, Any], field: str, kind: str) -> int:
    value = _required_segment_field(data, field, int, kind)
    _validate_segment_index(value, field, kind)
    return value


def _split_path_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False
    for ch in text:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if quote is not None:
            cur.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            cur.append(ch)
            continue
        if ch == "[":
            depth += 1
            cur.append(ch)
            continue
        if ch == "]":
            depth -= 1
            if depth < 0:
                raise QueryPathError(f"Unmatched ']' in path {text!r}.")
            cur.append(ch)
            continue
        if ch == "." and depth == 0:
            tokens.append("".join(cur))
            cur = []
            continue
        cur.append(ch)

    if quote is not None:
        raise QueryPathError(f"Unclosed quote in path {text!r}.")
    if depth != 0:
        raise QueryPathError(f"Unclosed '[' in path {text!r}.")
    tokens.append("".join(cur))
    return tokens


def _parse_token(token: str) -> list[PathSegment]:
    base: list[str] = []
    idx = 0
    while idx < len(token) and token[idx] != "[":
        base.append(token[idx])
        idx += 1

    base_text = "".join(base)
    segments: list[PathSegment] = []
    bracket_count = 0

    if base_text and not (base_text == "args" and idx < len(token)):
        segments.append(Kwarg(base_text))

    while idx < len(token):
        if token[idx] != "[":
            raise QueryPathError(f"Invalid path token {token!r}.")
        end = _find_bracket_end(token, idx)
        inside = token[idx + 1:end].strip()
        if inside == "":
            raise QueryPathError(f"Empty bracket path segment in {token!r}.")

        if inside.startswith("@set("):
            value = _parse_set_member(inside)
            segments.append(value)
        elif inside.startswith("@param("):
            segments.append(_parse_parameter(inside))
        elif inside.startswith("@key("):
            value = _parse_key_member(inside)
            segments.append(Key(value))
        else:
            value = _parse_bracket_value(inside)
            if base_text == "args" and bracket_count == 0:
                if not isinstance(value, int):
                    raise QueryPathError(f"args[...] requires an integer index in {token!r}.")
                segments.append(Arg(value))
            elif isinstance(value, int):
                segments.append(Index(value))
            else:
                segments.append(Key(value))

        bracket_count += 1
        idx = end + 1

    if not segments and base_text:
        segments.append(Kwarg(base_text))
    return segments


def _find_bracket_end(token: str, start: int) -> int:
    quote: str | None = None
    escape = False
    paren_depth = 0
    for idx in range(start + 1, len(token)):
        ch = token[idx]
        if escape:
            escape = False
            continue
        if quote is not None:
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")":
            paren_depth -= 1
            continue
        if ch == "]" and paren_depth == 0:
            return idx
    raise QueryPathError(f"Unclosed '[' in path token {token!r}.")


def _parse_bracket_value(text: str) -> int | str:
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return bytes(text[1:-1], "utf-8").decode("unicode_escape")
    try:
        return int(text)
    except ValueError:
        return text


def _parse_set_member(text: str) -> SetMember:
    if not text.endswith(")"):
        raise QueryPathError(f"Invalid set-member segment {text!r}.")
    inner = text[len("@set("):-1].strip()
    parts = _split_call_args(inner)
    if len(parts) not in {1, 2}:
        raise QueryPathError(f"Invalid set-member segment {text!r}.")
    fp = _parse_bracket_value(parts[0].strip())
    if not isinstance(fp, str):
        raise QueryPathError(f"Set-member fingerprint must be a string in {text!r}.")
    ordinal = int(parts[1].strip()) if len(parts) == 2 else 0
    return SetMember(fp, ordinal)


def _parse_parameter(text: str) -> Parameter:
    if not text.endswith(")"):
        raise QueryPathError(f"Invalid semantic parameter segment {text!r}.")
    try:
        name = ast.literal_eval(text[len("@param("):-1].strip())
    except Exception as error:
        raise QueryPathError(f"Invalid semantic parameter segment {text!r}.") from error
    if not isinstance(name, str):
        raise QueryPathError(f"Semantic parameter names must be strings in {text!r}.")
    return Parameter(name)


def _parse_key_member(text: str) -> Any:
    if not text.endswith(")"):
        raise QueryPathError(f"Invalid mapping-key segment {text!r}.")
    inner = text[len("@key("):-1].strip()
    try:
        return ast.literal_eval(inner)
    except Exception as e:
        raise QueryPathError(f"Invalid mapping-key literal in {text!r}.") from e


def _split_call_args(text: str) -> list[str]:
    args: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    escape = False
    for ch in text:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if quote is not None:
            cur.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            cur.append(ch)
            continue
        if ch == ",":
            args.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if quote is not None:
        raise QueryPathError(f"Unclosed quote in call args {text!r}.")
    args.append("".join(cur))
    return args
