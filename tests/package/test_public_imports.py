"""Verify public exports and passive imports from the installed wheel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


_EXPECTED_ROOT_EXPORTS = {
    "AnyValue", "Choice", "ConcreteDefinition", "Definition", "Exact",
    "IntRange", "Mat", "Missing", "Par", "Present", "QuotedDef", "Ref",
    "RefCDef", "RefCDefArg", "SKIP_ARGS", "Satisfies", "SearchSpace",
    "Selector", "SelectorArg", "SelectorSpec", "SubclassOf", "UniformFromSet",
    "UniformIntRange", "annotations", "artifacts", "config", "configure",
    "context", "core", "definition_mode", "env", "environments", "execute",
    "freeze", "reset_config", "selector_mode", "session", "space_mode",
    "status", "runtime", "world", "worlds",
}


def test_installed_root_exports_and_version_match_metadata(installed_python: Path) -> None:
    """Inspect exact root exports from the installed artifact."""

    result = _installed_probe(
        installed_python,
        """
import importlib.metadata
import json
import dryml
print(json.dumps({
    "exports": sorted(dryml.__all__),
    "module": dryml.__file__,
    "version": dryml.__version__,
    "metadata_version": importlib.metadata.version("dryml"),
}))
""",
    )
    data = json.loads(result.stdout)
    assert set(data["exports"]) == _EXPECTED_ROOT_EXPORTS
    assert data["version"] == data["metadata_version"] == "0.3.0.dev0"
    assert "site-packages" in data["module"].replace("\\", "/")


def test_installed_declaration_imports_are_passive(installed_python: Path) -> None:
    """Ensure root and declaration imports do not load optional frameworks."""

    result = _installed_probe(
        installed_python,
        """
import importlib.util
import json
import sys
import dryml
import dryml.annotations
import dryml.environments
import dryml.formats
import dryml.runtime
import dryml.session
import dryml.worlds
try:
    retired = importlib.util.find_spec("dryml.core2") is not None
except ModuleNotFoundError:
    retired = False
print(json.dumps({
    "heavy": sorted(name for name in ("tensorflow", "torch", "jax", "jaxlib", "ray") if name in sys.modules),
    "retired": retired,
}))
""",
    )
    assert json.loads(result.stdout) == {"heavy": [], "retired": False}


def _installed_probe(python: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a probe outside the checkout with inherited source paths removed."""

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(python), "-c", code],
        cwd="/tmp/dryml",
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
