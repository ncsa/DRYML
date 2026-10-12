"""Frozen v1 progress validator from fold.py at ba368afc.

Only domain-neutral spec/carry helpers are borrowed from the current module.
The envelope/version/key gates below are frozen, including the pre-carry v2
rejection. This qualification is not a supported product fallback reader.
"""

from dryml.artifacts.fold import _decode_spec_tree, _restore_carry, _carry_backend_devices

_PROGRESS_FORMAT = "dryml.artifacts.fold-progress"
_PROGRESS_VERSION = 1
_PROGRESS_KEYS = frozenset((
    "state", "count", "carry", "source_binding", "source_spec",
    "observed_spec", "carry_spec", "observation_devices", "program",
    "backend_devices", "operation_id", "attempt_id",
))


def _validate_progress_payload(payload):
    """Validate one complete v1 Fold progress envelope before assignment."""

    fields = {"format", "version", "processed_count", "progress"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Fold progress payload must be a complete v1 envelope.")
    if (
            payload["format"] != _PROGRESS_FORMAT
            or type(payload["version"]) is not int
            or payload["version"] != _PROGRESS_VERSION
    ):
        raise ValueError("Fold progress payload format or version is unsupported.")
    count = payload["processed_count"]
    if type(count) is not int or count < 0:
        raise ValueError("Fold processed count is invalid.")
    progress = payload["progress"]
    if progress is None:
        return count, None
    if not isinstance(progress, dict) or set(progress) != _PROGRESS_KEYS:
        raise ValueError("Fold active progress is malformed.")
    if progress["state"] not in {"active", "exhausted"}:
        raise ValueError("Fold active progress state is unsupported.")
    if type(progress["count"]) is not int or progress["count"] <= 0 or progress["count"] != count:
        raise ValueError("Fold active progress count is inconsistent.")
    if not isinstance(progress["source_binding"], dict) or set(progress["source_binding"]) != {"kind", "digest"}:
        raise ValueError("Fold checkpoint source binding is malformed.")
    if not all(type(progress["source_binding"][name]) is str for name in ("kind", "digest")):
        raise ValueError("Fold checkpoint source binding is invalid.")
    for name in ("source_spec", "observed_spec", "carry_spec"):
        _decode_spec_tree(progress[name])
    if not isinstance(progress["observation_devices"], tuple) or not all(
            item is None or type(item) is str for item in progress["observation_devices"]):
        raise ValueError("Fold checkpoint observation device evidence is malformed.")
    program = progress["program"]
    if not isinstance(program, dict) or set(program) != {"initializer", "accumulator", "finalizer"}:
        raise ValueError("Fold checkpoint program evidence is malformed.")
    if not all(type(program[name]) is str for name in ("initializer", "accumulator")):
        raise ValueError("Fold checkpoint program evidence is invalid.")
    if program["finalizer"] is not None and type(program["finalizer"]) is not str:
        raise ValueError("Fold checkpoint finalizer evidence is invalid.")
    if not all(type(progress[name]) is str and progress[name] for name in ("operation_id", "attempt_id")):
        raise ValueError("Fold checkpoint attempt evidence is invalid.")
    carry_spec = _decode_spec_tree(progress["carry_spec"])
    _restore_carry(progress["carry"], carry_spec, materialize=False)
    if progress["backend_devices"] != _carry_backend_devices(progress["carry"]):
        raise ValueError("Fold checkpoint backend/device evidence is inconsistent.")
    return count, progress
