"""Verify public exports and passive imports from the installed wheel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

_EXPECTED_ROOT_EXPORTS = {
    "AnyValue",
    "Choice",
    "ConcreteDefinition",
    "Definition",
    "Exact",
    "IntRange",
    "Mat",
    "Missing",
    "Par",
    "Present",
    "QuotedDef",
    "Ref",
    "RefCDef",
    "RefCDefArg",
    "SKIP_ARGS",
    "Satisfies",
    "SearchSpace",
    "Selector",
    "SelectorArg",
    "SelectorSpec",
    "SubclassOf",
    "UniformFromSet",
    "UniformIntRange",
    "annotations",
    "artifacts",
    "config",
    "configure",
    "context",
    "core",
    "definition_mode",
    "environments",
    "execute",
    "freeze",
    "load_object",
    "load_state_ref",
    "methods",
    "Object",
    "ObjectId",
    "ObjectRef",
    "Repo",
    "save_object",
    "Serializable",
    "StateRef",
    "StateSelectorRef",
    "StoreReport",
    "object_namespace",
    "reset_config",
    "selector_mode",
    "session",
    "space_mode",
    "status",
    "runtime",
    "worlds",
}

_EXPECTED_METHOD_EXPORTS = {
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "Method",
    "MethodCallMode",
    "MethodCallNode",
    "MethodCallNodeKind",
    "MethodCallSignature",
    "MethodError",
    "MethodImplementation",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
    "Traits",
    "traits",
}


def test_installed_root_exports_and_version_match_metadata(
    installed_python: Path,
) -> None:
    """Inspect exact root exports from the installed artifact."""

    result = _installed_probe(
        installed_python,
        """
import importlib.metadata
import json
import dryml
import dryml.core
print(json.dumps({
    "exports": sorted(dryml.__all__),
     "root_core_conveniences": all(
         getattr(dryml, name) is getattr(dryml.core, name)
         for name in dryml.core.__all__
         if name in dryml.__all__
     ),
     "root_methods": dryml.methods is __import__("dryml.methods", fromlist=["*"]),
    "module": dryml.__file__,
    "version": dryml.__version__,
    "metadata_version": importlib.metadata.version("dryml"),
}))
""",
    )
    data = json.loads(result.stdout)
    assert set(data["exports"]) == _EXPECTED_ROOT_EXPORTS
    assert data["root_core_conveniences"]
    assert data["root_methods"]
    assert data["version"] == data["metadata_version"] == "0.3.0.dev0"
    assert "site-packages" in data["module"].replace("\\", "/")


def test_installed_declaration_imports_are_passive(
    installed_python: Path,
) -> None:
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
import dryml.jax
import dryml.ray
import dryml.runtime
import dryml.session
import dryml.tf
import dryml.torch
import dryml.worlds
assert set(dryml.annotations.__all__) == {
    "Annotation", "ANNOTATION_ATTR", "attach_annotation", "own_annotations",
    "collect_annotations", "annotations_for_class", "annotations_for_method",
    "AnnotatedMember", "annotations_for_members",
    "AnnotationError", "AnnotationValidationError", "UnsupportedAnnotationTargetError",
}
assert "env" not in dryml.__all__
assert "world" not in dryml.__all__
assert "default" not in dryml.runtime.__all__
for name in ("decorators", "env", "world", "runtime", "merge", "namespaces", "storage"):
    try:
        importlib.import_module(f"dryml.annotations.{name}")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"retired annotation module remains importable: {name}")
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


def test_installed_methods_manifest_and_retired_imports(
    installed_python: Path,
) -> None:
    """Require the wheel to publish only the new Method owner and its API."""

    result = _installed_probe(
        installed_python,
        """
import importlib
import json

import dryml.code
import dryml.methods

assert set(dryml.methods.__all__) == {
    'ImplementationDeclarationError', 'ImplementationSelectionError', 'Method',
    'MethodCallMode', 'MethodCallNode', 'MethodCallNodeKind',
    'MethodCallSignature', 'MethodError', 'MethodImplementation',
    'PreparedCallMismatchError', 'SelectionFailureReason', 'SelectionTraitName',
    'Traits', 'traits',
}
assert not {'Method', 'Traits', 'traits'} & set(dryml.code.__all__)
for statement in (
    'from dryml.code import Method',
    'from dryml.code import Traits',
    'from dryml.code import traits',
):
    try:
        exec(statement, {})
    except ImportError:
        pass
    else:
        raise AssertionError(f'retired import succeeded: {statement}')
for module in ('dryml.code.method', 'dryml.code.traits'):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        assert error.name == module
    else:
        raise AssertionError(f'retired module remains importable: {module}')
print(json.dumps(sorted(dryml.methods.__all__)))
""",
    )
    assert set(json.loads(result.stdout)) == _EXPECTED_METHOD_EXPORTS


def test_installed_sdist_wheel_exercises_current_reference_authority(
    installed_python: Path,
) -> None:
    """Probe graph references, exact persistence, and retired authority in isolation."""

    result = _installed_probe(
        installed_python,
        """
import json
import pickle
import sys
import tempfile
from pathlib import Path

import dryml
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "probe_value.py").write_text(
        "from pathlib import Path\\n"
        "from dryml import Serializable\\n\\n"
        "class Value(Serializable):\\n"
        "    def __init__(self, value):\\n"
        "        self.value = value\\n\\n"
        "    def save_state_to_dir_imp(self, dest_dir, *, codec):\\n"
        "        Path(dest_dir, 'value.txt').write_text(str(self.value), encoding='ascii')\\n\\n"
        "    def restore_state_from_dir_imp(self, src_dir, *, codec):\\n"
        "        self.value = int(Path(src_dir, 'value.txt').read_text(encoding='ascii'))\\n",
        encoding="ascii",
    )
    sys.path.insert(0, str(root))
    from probe_value import Value
    repo = dryml.Repo(DirStore(root / "store"))
    value = Value(7, repo=repo)
    state = value.save(repo=repo)
    definition = pickle.loads(pickle.dumps(value.definition))
    load_repo = dryml.Repo(DirStore(root / "store"))
    loaded = dryml.load_state_ref(state, repo=load_repo, reuse_live="never")
    old = root / "old"
    (old / "objects" / "legacy").mkdir(parents=True)
    (old / "objects" / "legacy" / "definition.pkl").write_bytes(b"retired")
    try:
        DirStore(old)
    except StoreAuthorityError:
        old_authority_rejected = True
    else:
        old_authority_rejected = False
    load_repo.close(flush=False)
    repo.close(flush=False)
    print(json.dumps({
        "graph_round_trip": definition.graph_equal(value.definition),
        "object_paths": len(state.object.objects),
        "state_paths": len(state.states),
        "loaded_value": loaded.value,
        "old_authority_rejected": old_authority_rejected,
    }))
""",
    )
    assert json.loads(result.stdout) == {
        "graph_round_trip": True,
        "object_paths": 1,
        "state_paths": 1,
        "loaded_value": 7,
        "old_authority_rejected": True,
    }


def _installed_probe(
    python: Path, code: str
) -> subprocess.CompletedProcess[str]:
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
