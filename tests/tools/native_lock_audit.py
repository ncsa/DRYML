"""AST checks that reserve native advisory locking for ``dryml.locking``."""

from __future__ import annotations

import ast


def native_advisory_lock_offenders(source: str | bytes, *, filename: str = "<unknown>") -> list[str]:
    """Return native advisory-lock access found in Python source.

    ``fcntl`` is always reserved for the shared locking owner. ``msvcrt`` is
    likewise reserved except for ``get_osfhandle``, which converts a Windows
    file descriptor for unrelated native pipe handling. Parsing source avoids
    treating comments and string literals as imports or lock access.
    """

    tree = ast.parse(source, filename=filename)
    parents = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    msvcrt_bindings = set()
    converted_msvcrt_bindings = set()
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fcntl" or alias.name.startswith("fcntl."):
                    offenders.append("fcntl import")
                elif alias.name == "msvcrt":
                    msvcrt_bindings.add(alias.asname or "msvcrt")
                elif alias.name.startswith("msvcrt."):
                    offenders.append("msvcrt import")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "fcntl" or (node.module and node.module.startswith("fcntl.")):
                offenders.append("fcntl import")
            elif node.module == "msvcrt":
                offenders.extend("msvcrt import" for alias in node.names if alias.name != "get_osfhandle")
            elif node.module and node.module.startswith("msvcrt."):
                offenders.append("msvcrt import")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id not in msvcrt_bindings:
            continue
        parent = parents[node]
        if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "get_osfhandle":
            converted_msvcrt_bindings.add(node.id)
        else:
            offenders.append("msvcrt module escape")
    offenders.extend("msvcrt import" for binding in msvcrt_bindings - converted_msvcrt_bindings)
    return offenders
