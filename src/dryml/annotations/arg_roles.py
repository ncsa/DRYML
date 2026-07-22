"""Conservative re-export layer for constructor argument-role annotations."""

from dryml.core.arg_roles import ArgRole, MaterializeArg, RefCDef, RefCDefArg, SelectorArg, ValueArg, apply_arg_roles, apply_definition_arg_roles, normalize_role, resolve_arg_roles, role_from_annotation

ARG_ROLE_HELP = (
    "DRYML argument-role annotations affect constructor canonicalization and ConcreteDefinition identity; "
    "environment/world/runtime AnnotationFragments are sidecar planning metadata."
)

__all__ = [
    "ARG_ROLE_HELP",
    "ArgRole",
    "MaterializeArg",
    "RefCDef",
    "RefCDefArg",
    "SelectorArg",
    "ValueArg",
    "apply_arg_roles",
    "apply_definition_arg_roles",
    "normalize_role",
    "resolve_arg_roles",
    "role_from_annotation",
]
