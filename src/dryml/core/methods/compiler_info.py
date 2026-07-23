from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompilerInfo:
    """Compiler-facing metadata for DRYML methods.

    Args:
        pure: Whether the method is pure.
        elementwise: Whether the method acts elementwise.
        shape_preserving: Whether output shape follows input shape.
        opaque: Whether compiler tooling should treat the method as opaque.
        static_argnames: Names of static arguments.
        tags: Additional compiler tags.
    """

    pure: bool = True
    elementwise: bool = False
    shape_preserving: bool = False
    opaque: bool = False
    static_argnames: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()


__all__ = ["CompilerInfo"]
