from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


GRAPH_PATH_SCHEMA_VERSION = 1


class GraphPathError(Exception):
    pass


QueryPathError = GraphPathError


@dataclass(frozen=True, slots=True)
class Kwarg:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Arg:
    index: int

    def __str__(self) -> str:
        return f"args[{self.index}]"


@dataclass(frozen=True, slots=True)
class Index:
    index: int

    def __str__(self) -> str:
        return f"[{self.index}]"


@dataclass(frozen=True, slots=True)
class Key:
    key: Any

    def __str__(self) -> str:
        return f"[{self.key!r}]"


@dataclass(frozen=True, slots=True)
class SetMember:
    fingerprint: str
    ordinal: int = 0

    def __str__(self) -> str:
        return f'@set("{self.fingerprint}", {self.ordinal})'


PathSegment = Kwarg | Arg | Index | Key | SetMember
GraphPathLike = str | Iterable[PathSegment | str | int] | "GraphPath"


@dataclass(frozen=True, slots=True, eq=False)
class GraphPath:
    segments: tuple[PathSegment, ...] = ()

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def __getitem__(self, item):
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
        return type(self)(self.segments + (_normalize_segment(segment),))

    def join(self, other: GraphPathLike) -> "GraphPath":
        norm = normalize_path(other)
        return type(self)(self.segments + norm.segments)

    @property
    def parent(self) -> "GraphPath":
        if not self.segments:
            return self
        return type(self)(self.segments[:-1])

    @property
    def name(self) -> str | None:
        if not self.segments:
            return None
        return str(self.segments[-1])

    @property
    def is_root(self) -> bool:
        return not self.segments

    def startswith(self, other: GraphPathLike) -> bool:
        norm = normalize_path(other)
        return self.segments[:len(norm.segments)] == norm.segments

    def overlaps(self, other: GraphPathLike) -> bool:
        norm = normalize_path(other)
        return self.startswith(norm) or norm.startswith(self)

    def relative_to(self, prefix: GraphPathLike) -> "GraphPath":
        norm = normalize_path(prefix)
        if not self.startswith(norm):
            raise QueryPathError(f"Path {self!s} is not relative to prefix {norm!s}.")
        return type(self)(self.segments[len(norm.segments):])

    def legacy_tuple(self) -> tuple[str | int, ...]:
        return tuple(_legacy_segment_value(seg) for seg in self.segments)

    def to_legacy_tuple(self) -> tuple[str | int, ...]:
        return self.legacy_tuple()

    def legacy_str(self) -> str:
        if not self.segments:
            return "<root>"
        return "/".join(map(str, self.legacy_tuple()))

    def __str__(self) -> str:
        if not self.segments:
            return "$"

        out = "$"
        for seg in self.segments:
            if isinstance(seg, Kwarg):
                out += f".{seg.name}"
            elif isinstance(seg, Arg):
                out += f".args[{seg.index}]"
            elif isinstance(seg, Index):
                out += f"[{seg.index}]"
            elif isinstance(seg, Key):
                out += f"[{seg.key!r}]"
            elif isinstance(seg, SetMember):
                out += f'[@set("{seg.fingerprint}", {seg.ordinal})]'
        return out


DefinitionPath = GraphPath
DefinitionPathLike = GraphPathLike


def normalize_path(path: GraphPathLike = "$") -> GraphPath:
    if isinstance(path, GraphPath):
        return path
    if isinstance(path, str):
        return parse_path(path)

    segments: list[PathSegment] = []
    for part in path:
        segments.append(_normalize_segment(part))
    return GraphPath(tuple(segments))


normalize_graph_path = normalize_path


def normalize_ctx_path(path: GraphPathLike | None = None) -> GraphPath:
    if path is None:
        return GraphPath()
    if isinstance(path, tuple) and all(isinstance(part, (str, int)) for part in path):
        # Generic graph callers historically stored raw strings/ints.
        return GraphPath(tuple(_normalize_legacy_segment(part) for part in path))
    return normalize_path(path)


def parse_path(text: str) -> GraphPath:
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
    if isinstance(part, (Kwarg, Arg, Index, Key, SetMember)):
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
