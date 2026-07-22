from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ConfigError(KeyError):
    pass


class _MissingConfigDefault:
    def __repr__(self) -> str:
        return "CONFIG_MISSING"

    def __reduce__(self):
        return (_get_missing_config_default, ())


def _get_missing_config_default():
    return CONFIG_MISSING


CONFIG_MISSING = _MissingConfigDefault()


@dataclass(frozen=True, slots=True)
class ConfigRef:
    """Reference to a runtime configuration value in the current Repo."""

    key: str
    default: Any = CONFIG_MISSING

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("ConfigRef key must be a string.")
        if self.key == "":
            raise ValueError("ConfigRef key cannot be empty.")

    @property
    def has_default(self) -> bool:
        return self.default is not CONFIG_MISSING

    def resolve(self, repo=None):
        from dryml.core.repo import manage_repo

        with manage_repo(repo=repo) as repo_obj:
            if self.has_default:
                return repo_obj.get_config(self.key, default=self.default)
            return repo_obj.get_config(self.key)

    @staticmethod
    def resolve_value(value, repo=None):
        from dryml.core.repo import manage_repo

        with manage_repo(repo=repo) as repo_obj:
            return repo_obj.resolve_config(value)

    def __stable_leaf_bytes__(self) -> bytes:
        if self.has_default:
            from dryml.core.utils.stable_hash import stable_hash_function

            default_hash = stable_hash_function(self.default)
        else:
            default_hash = "<missing>"
        return f"ConfigRef:{self.key}:{default_hash}".encode("utf-8")
