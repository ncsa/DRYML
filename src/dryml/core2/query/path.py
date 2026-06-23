from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..definition import ConcreteDefinition, Definition, SKIP_ARGS
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple


class QueryPathError(Exception):
    pass


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


PathSegment = Kwarg | Arg | Index | Key


@dataclass(frozen=True, slots=True)
class DefinitionPath:
    segments: tuple[PathSegment, ...] = ()

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return DefinitionPath(self.segments[item])
        return self.segments[item]

    def child(self, segment: PathSegment) -> "DefinitionPath":
        return DefinitionPath(self.segments + (segment,))

    def startswith(self, other: "DefinitionPath") -> bool:
        return self.segments[:len(other.segments)] == other.segments

    def overlaps(self, other: "DefinitionPath") -> bool:
        return self.startswith(other) or other.startswith(self)

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
        return out


DefinitionPathLike = str | Iterable[PathSegment | str | int] | DefinitionPath


def normalize_path(path: DefinitionPathLike = "$") -> DefinitionPath:
    if isinstance(path, DefinitionPath):
        return path
    if isinstance(path, str):
        return parse_path(path)

    segments: list[PathSegment] = []
    for part in path:
        if isinstance(part, (Kwarg, Arg, Index, Key)):
            segments.append(part)
        elif isinstance(part, str):
            segments.append(Kwarg(part))
        elif isinstance(part, int):
            segments.append(Index(part))
        else:
            raise QueryPathError(f"Unsupported path segment {part!r} in {path!r}.")
    return DefinitionPath(tuple(segments))


def parse_path(text: str) -> DefinitionPath:
    if text == "" or text == "$":
        return DefinitionPath()
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        raise QueryPathError(f"Invalid path syntax {text!r}; expected '$' or '$.'.")

    tokens = _split_path_tokens(text)
    segments: list[PathSegment] = []
    for token in tokens:
        if token == "":
            raise QueryPathError(f"Invalid empty path segment in {text!r}.")
        segments.extend(_parse_token(token))
    return DefinitionPath(tuple(segments))


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
        if ch == "]":
            return idx
    raise QueryPathError(f"Unclosed '[' in path token {token!r}.")


def _parse_bracket_value(text: str) -> int | str:
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return bytes(text[1:-1], "utf-8").decode("unicode_escape")
    try:
        return int(text)
    except ValueError:
        return text


def get_subtree(obj: Any, path: DefinitionPathLike = "$") -> Any:
    norm = normalize_path(path)
    cur = obj
    for idx, seg in enumerate(norm):
        try:
            cur = _get_child(cur, seg)
        except Exception as e:
            failing = DefinitionPath(norm.segments[:idx + 1])
            raise QueryPathError(f"Failed to resolve segment {seg!s} at {failing!s}.") from e
    return cur


def replace_subtree(obj: Any, path: DefinitionPathLike, replacement: Any) -> Any:
    norm = normalize_path(path)
    if len(norm) == 0:
        return replacement
    seg = norm[0]
    rest = DefinitionPath(norm.segments[1:])
    child = get_subtree(obj, DefinitionPath((seg,)))
    new_child = replace_subtree(child, rest, replacement)
    return _replace_child(obj, seg, new_child)


def _get_child(obj: Any, seg: PathSegment) -> Any:
    if isinstance(obj, (Definition, ConcreteDefinition)):
        if isinstance(seg, Kwarg):
            return obj.kwargs[seg.name]
        if isinstance(seg, Arg):
            if obj.args is None:
                raise KeyError(seg.index)
            return obj.args[seg.index]
        raise TypeError(f"{seg!s} is not valid on a definition.")

    if isinstance(obj, (dict, FrozenDict)):
        if isinstance(seg, Key):
            return obj[seg.key]
        if isinstance(seg, Kwarg):
            return obj[seg.name]
        raise TypeError(f"{seg!s} is not valid on a mapping.")

    if isinstance(obj, (list, tuple, FrozenList, FrozenTuple)):
        if isinstance(seg, Index):
            return obj[seg.index]
        raise TypeError(f"{seg!s} is not valid on a sequence.")

    raise TypeError(f"Cannot traverse into {type(obj).__name__}.")


def _replace_child(obj: Any, seg: PathSegment, child: Any) -> Any:
    if isinstance(obj, (Definition, ConcreteDefinition)):
        args = None if obj.args is None else list(obj.args)
        kwargs = dict(obj.kwargs)
        if isinstance(seg, Kwarg):
            if seg.name not in kwargs:
                raise QueryPathError(f"Missing kwarg {seg.name!r} while replacing {seg!s}.")
            kwargs[seg.name] = child
        elif isinstance(seg, Arg):
            if args is None:
                raise QueryPathError(f"Cannot replace arg {seg.index}; definition skips args.")
            args[seg.index] = child
        else:
            raise QueryPathError(f"{seg!s} is not valid on a definition.")

        if args is None:
            return Definition(obj.cls, SKIP_ARGS, **kwargs)
        return Definition(obj.cls, *args, **kwargs)

    if isinstance(obj, list):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a list.")
        out = list(obj)
        out[seg.index] = child
        return out

    if isinstance(obj, tuple) and not isinstance(obj, (FrozenList, FrozenTuple)):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a tuple.")
        out = list(obj)
        out[seg.index] = child
        return tuple(out)

    if isinstance(obj, FrozenList):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a FrozenList.")
        out = list(obj)
        out[seg.index] = child
        return FrozenList(out)

    if isinstance(obj, FrozenTuple):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a FrozenTuple.")
        out = list(obj)
        out[seg.index] = child
        return FrozenTuple(out)

    if isinstance(obj, dict):
        key = _mapping_key_from_segment(seg)
        if key not in obj:
            raise QueryPathError(f"Missing mapping key {key!r} while replacing {seg!s}.")
        out = dict(obj)
        out[key] = child
        return out

    if isinstance(obj, FrozenDict):
        key = _mapping_key_from_segment(seg)
        if key not in obj:
            raise QueryPathError(f"Missing mapping key {key!r} while replacing {seg!s}.")
        out = dict(obj.items())
        out[key] = child
        return FrozenDict(out)

    if isinstance(obj, (set, FrozenSet)):
        raise QueryPathError("Replacing set members by path is not supported because set order is not stable.")

    raise QueryPathError(f"Cannot replace a child on {type(obj).__name__}.")


def _mapping_key_from_segment(seg: PathSegment) -> Any:
    if isinstance(seg, Key):
        return seg.key
    if isinstance(seg, Kwarg):
        return seg.name
    raise QueryPathError(f"{seg!s} is not valid on a mapping.")
