from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CompilerInfo:
    pure: bool = True
    elementwise: bool = False
    shape_preserving: bool = False
    opaque: bool = False
    static_argnames: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()



