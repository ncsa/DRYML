import importlib
import inspect
import types
from typing import Any, Mapping, Iterable

def reload_and_patch(
    module_name: str,
    *,
    target_globals: dict[str, Any] | None = None,
    names: Iterable[str] | Mapping[str, str] | None = None,
    patch_predicate=None,
    verbose: bool = False,
):
    """
    Reload `module_name` and update `from module import X` bindings in `target_globals`.

    - If `names` is None: auto-detect caller globals that reference objects previously
      exported by the module (by identity), and rebind them to the new exports.
    - If `names` is an iterable: treat as export names; patch same local names.
    - If `names` is a mapping: {local_name: export_name} (supports aliases).

    By default patches only functions/classes/modules/callables to avoid clobbering
    common immutables (ints/strings).
    """
    mod = importlib.import_module(module_name)
    old_dict = dict(mod.__dict__)  # snapshot exports before reload

    # Decide where to patch (defaults to caller's globals)
    if target_globals is None:
        frame = inspect.currentframe()
        assert frame is not None
        target_globals = frame.f_back.f_globals  # caller
        del frame

    # Default safety filter
    if patch_predicate is None:
        def patch_predicate(x):
            return (
                isinstance(x, types.ModuleType)
                or inspect.isfunction(x)
                or inspect.isclass(x)
                or inspect.isbuiltin(x)
                or callable(x)
            )

    # Build id(old_obj) -> [export_names...] for objects we consider patchable
    old_id_to_exports: dict[int, list[str]] = {}
    for export_name, old_obj in old_dict.items():
        if patch_predicate(old_obj):
            old_id_to_exports.setdefault(id(old_obj), []).append(export_name)

    # Reload
    importlib.reload(mod)
    new_dict = mod.__dict__

    patched = {}

    # Explicit patch list
    if names is not None:
        if isinstance(names, Mapping):
            items = list(names.items())  # local -> export
        else:
            items = [(n, n) for n in names]
        for local_name, export_name in items:
            if export_name in new_dict:
                target_globals[local_name] = new_dict[export_name]
                patched[local_name] = export_name
        if verbose and patched:
            print(f"[reload_and_patch] patched explicit: {patched}")
        return mod, patched

    # Auto-detect: patch any caller globals that refer to old exported objects
    for local_name, local_obj in list(target_globals.items()):
        export_names = old_id_to_exports.get(id(local_obj))
        if not export_names:
            continue
        # Choose the first export name that still exists after reload
        for export_name in export_names:
            if export_name in new_dict:
                target_globals[local_name] = new_dict[export_name]
                patched[local_name] = export_name
                break

    if verbose:
        print(f"[reload_and_patch] reloaded {module_name}, patched {len(patched)} names")
        if patched:
            print(f"[reload_and_patch] patched: {patched}")

    return mod, patched
