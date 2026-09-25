"""Result-only Artifact persistence for completed Value payloads."""

from __future__ import annotations

import os
from typing import Any, Generic, TypeVar

from dryml.core.utils.general import pickle_load, pickle_save

from .base import Artifact


T = TypeVar("T")
_VALUE_FILENAME = "value.pkl"
_VALUE_FORMAT = "dryml.artifacts.value"
_VALUE_VERSION = 1
_VALUE_KEYS = frozenset(("format", "version", "present", "result"))


class ArtifactNotReadyError(RuntimeError):
    """Raised when a Value result is requested before it is usable."""


class Value(Artifact, Generic[T]):
    """Abstract Artifact that persists one complete, optionally ``None`` result.

    Concrete subclasses define managed computation and readiness. Value stores
    only its validated result envelope in ``value.pkl``; it does not persist
    input graphs, live iterators, managed controls, or ordinary subclass fields.
    Subclasses may contribute separate Serializable hook files and may constrain
    present results with :meth:`_validate_value_result`.
    """

    def value(self) -> T:
        """Return the completed result without computing or loading inputs.

        Returns:
            The stored result, including a legitimately computed ``None``.

        Raises:
            ArtifactNotReadyError: If readiness is false or no complete result
                envelope is installed.
        """

        if not self.ready or not self._value_is_present():
            raise ArtifactNotReadyError("Artifact result is not ready.")
        return self._value_payload["result"]

    def _value_is_present(self) -> bool:
        """Return whether this Value has an installed complete result envelope.

        Returns:
            ``True`` when the protected envelope slot is installed and marks a
            result as present. This helper does not evaluate readiness itself.
        """

        payload = getattr(self, "_value_payload", None)
        return payload is not None and payload["present"]

    def _validate_value_result(self, result: T) -> None:
        """Validate one present result before protected payload installation.

        Args:
            result: Decoded or newly computed present result.

        Raises:
            ValueError: Concrete subclasses may reject a result outside their
                documented domain.

        Side Effects:
            Implementations must be side-effect-free because validation occurs
            before the new envelope replaces the current result.
        """

    def _install_value_payload(self, payload: dict[str, Any]) -> None:
        """Validate and atomically install one complete Value result envelope.

        Args:
            payload: Exact v1 envelope with ``format``, ``version``, ``present``,
                and ``result`` fields. Absent envelopes must use ``None`` as the
                result.

        Raises:
            ValueError: If the envelope is malformed, unsupported, or rejected
                by this subclass's result validator.

        Side Effects:
            Replaces the protected result payload slot only after structural and
            subclass domain validation both succeed. Used by compute paths and
            state restoration; it never computes or resolves inputs.
        """

        if type(payload) is not dict or set(payload) != _VALUE_KEYS:
            raise ValueError("Value payload must be a complete v1 envelope.")
        if payload["format"] != _VALUE_FORMAT:
            raise ValueError("Value payload format is unsupported.")
        if type(payload["version"]) is not int or payload["version"] != _VALUE_VERSION:
            raise ValueError("Value payload version is unsupported.")
        if type(payload["present"]) is not bool:
            raise ValueError("Value payload presence marker is invalid.")
        if not payload["present"] and payload["result"] is not None:
            raise ValueError("An absent Value payload must have a null result.")
        if payload["present"]:
            self._validate_value_result(payload["result"])
        self._value_payload = {
            "format": _VALUE_FORMAT,
            "version": _VALUE_VERSION,
            "present": payload["present"],
            "result": payload["result"],
        }

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write this Value's narrow v1 result envelope to ``value.pkl``.

        Args:
            dest_dir: Framework-provided payload directory.
            codec: Validated opaque state codec identifier.

        Side Effects:
            Writes only the result-presence envelope through DRYML's existing
            trusted dill protocol-5 pickle helper. An uncomputed Value writes an
            explicit absent envelope.
        """

        payload = getattr(self, "_value_payload", {
            "format": _VALUE_FORMAT,
            "version": _VALUE_VERSION,
            "present": False,
            "result": None,
        })
        pickle_save(payload, os.path.join(dest_dir, _VALUE_FILENAME))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Restore and validate this Value's v1 result envelope from ``value.pkl``.

        Args:
            src_dir: Framework-provided payload directory containing ``value.pkl``.
            codec: Validated opaque state codec identifier.

        Raises:
            OSError: If the result payload is missing or unreadable.
            ValueError: If the decoded envelope or result domain is invalid.

        Side Effects:
            Installs a new result only after decoding and complete validation;
            failed restoration leaves the prior protected result untouched.
        """

        payload = pickle_load(os.path.join(src_dir, _VALUE_FILENAME))
        self._install_value_payload(payload)


ArtifactNotReadyError.__module__ = "dryml.artifacts"
Value.__module__ = "dryml.artifacts"
