"""Verify public exports and passive imports from the installed wheel."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess

_EXPECTED_CODE_EXPORTS = [
    "AccessCollection",
    "AnalysisErrorCode",
    "AnalysisKernel",
    "AnalysisResult",
    "CallableInfo",
    "CodeAnalysisError",
    "CodeFact",
    "CodeFacts",
    "CodeTarget",
    "CodeTargetInput",
    "Diagnostic",
    "DescriptorTarget",
    "FactRecord",
    "ImportTarget",
    "InvalidKernelError",
    "InvalidTargetError",
    "InvocationOutcome",
    "KernelCall",
    "KernelDependencyError",
    "KernelExecutionError",
    "KernelOutcome",
    "MissingOutputError",
    "ProgramGraph",
    "SourceInfo",
    "SourceTarget",
    "SourceUnavailableError",
    "TargetInfo",
    "TraversalKernel",
    "analyze",
    "analyze_callable",
    "extract_source",
    "get_source_info",
    "probe",
    "trace",
]

_EXPECTED_CODE_MODULE_EXPORTS = {
    "dryml.code.algorithms": [
        "LexicalDependencies", "LexicalDependency", "LexicalDependencyKernel",
        "collect_lexical_dependencies",
    ],
    "dryml.code.analysis": ["AnalysisResult", "InvocationOutcome", "analyze"],
    "dryml.code.ast_tools": [
        "AccessCollection", "AttrAccess", "MethodCall",
        "collect_accesses_from_source", "parse_source",
    ],
    "dryml.code.callable_info": ["CallableInfo", "analyze_callable"],
    "dryml.code.errors": [
        "AnalysisErrorCode", "CodeAnalysisError", "InvalidKernelError",
        "InvalidTargetError", "KernelDependencyError", "KernelExecutionError",
        "MissingOutputError", "SourceUnavailableError",
    ],
    "dryml.code.facts": [
        "CodeFact", "CodeFacts", "Diagnostic", "FactRecord", "FactScalar",
        "FactValue", "SourceLocation",
    ],
    "dryml.code.graph": [
        "ProgramEdge", "ProgramEdgeKind", "ProgramGraph", "ProgramNode",
        "ProgramNodeKind", "build_program_graph",
    ],
    "dryml.code.kernels": [
        "AnalysisKernel", "KernelCall", "KernelContext", "KernelMode",
        "KernelOutcome", "TraversalKernel",
    ],
    "dryml.code.probe": ["probe"],
    "dryml.code.source": ["SourceInfo", "extract_source", "get_source_info"],
    "dryml.code.targets": [
        "CodeTarget", "CodeTargetInput", "DescriptorKind", "DescriptorTarget",
        "ImportTarget", "SourceTarget", "TargetInfo", "TargetKind",
        "normalize_target",
    ],
    "dryml.code.trace": ["trace"],
}

_EXPECTED_CODE_DATACLASS_FIELDS = {
    "dryml.code.callable_info.CallableInfo": [
        "original", "func", "bound_self", "signature", "qualname", "module",
        "is_bound_method", "is_function", "is_callable_instance",
    ],
    "dryml.code.source.SourceInfo": ["source", "filename", "start_line"],
    "dryml.code.ast_tools.AttrAccess": ["root", "chain", "ctx", "lineno", "col_offset"],
    "dryml.code.ast_tools.MethodCall": ["root", "chain", "lineno", "col_offset"],
    "dryml.code.ast_tools.AccessCollection": ["attr_accesses", "method_calls"],
    "dryml.code.targets.SourceTarget": ["source", "name", "filename", "start_line"],
    "dryml.code.targets.ImportTarget": ["path"],
    "dryml.code.targets.DescriptorTarget": ["owner", "name"],
    "dryml.code.targets.TargetInfo": [
        "kind", "name", "module", "qualname", "owner_module", "owner_qualname",
        "descriptor_kind", "filename", "start_line", "import_path",
    ],
    "dryml.code.targets.CodeTarget": [
        "info", "original", "callable", "owner", "descriptor", "source",
        "import_path",
    ],
    "dryml.code.graph.ProgramNode": ["id", "kind", "value", "source"],
    "dryml.code.graph.ProgramEdge": ["source", "target", "kind"],
    "dryml.code.graph.ProgramGraph": ["target", "nodes", "edges", "diagnostics"],
    "dryml.code.facts.SourceLocation": ["filename", "line", "column"],
    "dryml.code.facts.CodeFact": ["kind", "value", "source"],
    "dryml.code.facts.CodeFacts": ["values"],
    "dryml.code.facts.FactRecord": ["fact", "producer", "graph_digest", "origin"],
    "dryml.code.facts.Diagnostic": ["code", "message", "severity", "kernel", "source"],
    "dryml.code.kernels.KernelCall": ["kernel", "input"],
    "dryml.code.kernels.KernelOutcome": [
        "kernel", "graph_digest", "status", "value", "diagnostics", "skipped_for",
    ],
    "dryml.code.analysis.InvocationOutcome": ["status", "diagnostic"],
    "dryml.code.analysis.AnalysisResult": [
        "target", "base_graph", "graph", "outcomes", "facts", "diagnostics", "invocation",
    ],
    "dryml.code.algorithms.LexicalDependency": ["name", "source"],
    "dryml.code.algorithms.LexicalDependencies": ["dependencies"],
}

_EXPECTED_CODE_SIGNATURES = {
    "dryml.code.callable_info.analyze_callable": "(obj: 'Callable[..., Any]') -> 'CallableInfo'",
    "dryml.code.source.get_source_info": "(obj: 'object') -> 'SourceInfo | None'",
    "dryml.code.source.extract_source": "(target: 'CodeTargetInput') -> 'SourceInfo'",
    "dryml.code.ast_tools.parse_source": "(source: 'str | SourceInfo') -> 'ast.Module'",
    "dryml.code.ast_tools.collect_accesses_from_source": "(source: 'str | SourceInfo') -> 'AccessCollection'",
    "dryml.code.targets.normalize_target": "(target: 'CodeTargetInput') -> 'CodeTarget'",
    "dryml.code.graph.build_program_graph": "(target: 'CodeTargetInput') -> 'ProgramGraph'",
    "dryml.code.analysis.analyze": "(target: 'CodeTargetInput', calls: 'Iterable[KernelCall[Any, Any]]') -> 'AnalysisResult'",
    "dryml.code.probe.probe": "(target: 'CodeTargetInput', calls: 'Iterable[KernelCall[Any, Any]]') -> 'AnalysisResult'",
    "dryml.code.trace.trace": "(target: 'CodeTargetInput', calls: 'Iterable[KernelCall[Any, Any]]', *, args: 'tuple[Any, ...]' = (), kwargs: 'Mapping[str, Any] | None' = None, max_events: 'int' = 100000) -> 'AnalysisResult'",
    "dryml.code.algorithms.collect_lexical_dependencies": "(target: 'CodeTargetInput') -> 'LexicalDependencies'",
}

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
    "env",
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
    "requirements",
    "selector_mode",
    "session",
    "space_mode",
    "status",
    "runtime",
    "world",
    "worlds",
}

_EXPECTED_REQUIREMENT_EXPORTS = {
    "AdmissionReport",
    "RequirementBarrierError",
    "RequirementCombinationError",
    "RequirementCombiner",
    "RequirementDeclaration",
    "RequirementError",
    "RequirementIssue",
    "RequirementReport",
    "RequirementResult",
    "RequirementSource",
    "combine_requirements",
    "require_admission",
}

_EXPECTED_ENVIRONMENT_EXPORTS = {
    "COMPATIBILITY_REPORT_SCHEMA_VERSION", "ENVIRONMENT_LOCK_REF_SCHEMA_VERSION",
    "ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION", "ENVIRONMENT_RECORD_SCHEMA_VERSION",
    "ENVIRONMENT_REQUIREMENT_SCHEMA_VERSION", "ENVIRONMENT_SPEC_SCHEMA_VERSION",
    "CompatibilityIssue", "CompatibilityReport", "CondaEnvironmentSpec",
    "ContainerEnvironmentSpec", "CurrentEnvironmentSpec", "DrymlEnvironmentError",
    "DrymlRuntimeRecord", "EnvironmentCompatibilityError",
    "EnvironmentFeatureUnavailable", "EnvironmentInternTable", "EnvironmentLockRef",
    "EnvironmentProbeError", "EnvironmentRecord", "EnvironmentRegistry",
    "EnvironmentRegistryEntry", "EnvironmentRegistryError", "EnvironmentRequirement",
    "EnvironmentRequirementError", "EnvironmentSerializationError", "EnvironmentSpecError",
    "EnvironmentProbeResult", "PackageRecord", "PlatformRecord", "PythonExecutableSpec",
    "PythonRecord", "coerce_policy", "current", "inspect_current",
    "malformed_report", "marker_environment_from_record", "normalize_distribution_name",
    "normalize_requirement_string", "probe", "probe_conda", "probe_current",
    "probe_python", "req", "requirements_for", "requirements_for_method", "reset_current",
    "set_current", "spec_from_data", "unavailable_report", "use",
}

_EXPECTED_WORLD_EXPORTS = {
    "CountConstraint", "LocalResourceInventory", "ProcessAllocation", "ProcessSpec",
    "ResourceRequirement", "ResourceSpec", "ResourceValidationError", "RoleRequirement",
    "RoleSpec", "WorldAllocation", "WorldCompatibilityError", "WorldCompatibilityIssue",
    "WorldCompatibilityReport", "WorldError", "WorldRequirement", "WorldRequirementError",
    "WorldSpec", "WorldSpecValidationError", "WorldSynthesisDiagnostic",
    "WorldSynthesisResult", "assign_local_world", "canonical_byte_size",
    "check_allocation_satisfies_requirement", "check_world_spec_satisfies_requirement",
    "current", "local_inventory", "parse_byte_size", "req", "requirements_for",
    "requirements_for_method", "reset_current", "set_current", "synthesize", "use",
}

_RETIRED_ENVIRONMENT_SURFACE = {
    "ENVIRONMENT_FRAGMENT_SCHEMA_VERSION",
    "FRAGMENT_ATTR",
    "RequirementFragment",
    "add_req",
    "compose_fragments",
    "fragments_for_class",
    "override_req",
    "requirements_for_class",
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

_EXPECTED_ANNOTATION_EXPORTS = {
    "ANNOTATION_ATTR",
    "AnnotatedMember",
    "Annotation",
    "AnnotationError",
    "AnnotationValidationError",
    "UnsupportedAnnotationTargetError",
    "annotations_for_class",
    "annotations_for_members",
    "annotations_for_method",
    "attach_annotation",
    "collect_annotations",
    "own_annotations",
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
     "aliases": {
         "env": dryml.env is dryml.environments,
         "world": dryml.world is dryml.worlds,
         "requirements": dryml.requirements.__name__,
     },
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
    assert data["aliases"] == {"env": True, "world": True, "requirements": "dryml.requirements"}
    assert data["version"] == data["metadata_version"] == "0.3.0.dev2"
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
assert dryml.env is dryml.environments
assert dryml.world is dryml.worlds
assert "requirements" in dryml.__all__
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


def test_installed_requirement_domain_surface_has_no_defaults_or_fragment_shims(
    installed_python: Path,
) -> None:
    """Require the installed wheel to retain only public domain requirement APIs."""

    result = _installed_probe(
        installed_python,
        """
import importlib.util
import inspect
import json

import dryml
from dryml import env, world

retired_modules = (
    'dryml.env', 'dryml.world', 'dryml.environments.fragments',
    'dryml.environments.fragment', 'dryml.runtime.default',
)
print(json.dumps({
    'requirements': sorted(dryml.requirements.__all__),
    'environment': sorted(env.__all__),
    'world': sorted(world.__all__),
    'environment_key_exported': 'ENVIRONMENT_REQUIREMENT_KEY' in env.__all__,
    'world_key_exported': 'WORLD_REQUIREMENT_KEY' in world.__all__,
    'selectors': {
        'env': all(hasattr(env, name) for name in ('current', 'set_current', 'reset_current', 'use')),
        'world': all(hasattr(world, name) for name in ('current', 'set_current', 'reset_current', 'use')),
    },
    'defaults': {
        'env': [name for name in ('default', 'default_for', 'set_default', 'reset_default', 'use_default') if hasattr(env, name)],
        'world': [name for name in ('default', 'default_for', 'set_default', 'reset_default', 'use_default') if hasattr(world, name)],
        'runtime': hasattr(dryml.runtime, 'default'),
    },
    'retired': [name for name in %r if hasattr(env, name)],
    'retired_modules': [
        name for name in retired_modules if importlib.util.find_spec(name) is not None
    ],
    'signatures': {
        'env.req': str(inspect.signature(env.req)),
        'env.requirements_for': str(inspect.signature(env.requirements_for)),
        'env.requirements_for_method': str(inspect.signature(env.requirements_for_method)),
        'world.req': str(inspect.signature(world.req)),
        'world.requirements_for': str(inspect.signature(world.requirements_for)),
        'world.requirements_for_method': str(inspect.signature(world.requirements_for_method)),
    },
}))
""" % (tuple(sorted(_RETIRED_ENVIRONMENT_SURFACE)),),
    )
    data = json.loads(result.stdout)
    assert set(data["requirements"]) == _EXPECTED_REQUIREMENT_EXPORTS
    assert set(data["environment"]) == _EXPECTED_ENVIRONMENT_EXPORTS
    assert set(data["world"]) == _EXPECTED_WORLD_EXPORTS
    assert not data["environment_key_exported"]
    assert not data["world_key_exported"]
    assert data["selectors"] == {"env": True, "world": True}
    assert data["defaults"] == {"env": [], "world": [], "runtime": False}
    assert data["retired"] == []
    assert data["retired_modules"] == []
    assert data["signatures"] == {
        "env.req": "(*, python: 'str | None' = None, requirements: 'Iterable[str]' = (), excludes: 'Iterable[str]' = (), capabilities: 'Iterable[str]' = (), tags: 'Iterable[str]' = (), dryml_protocol: 'str | None' = None, schema_versions: 'Mapping[str, str] | None' = None, source: 'RequirementSource | str | None' = None) -> 'Callable[[T], T]'",
        "env.requirements_for": "(target: 'object') -> 'RequirementResult[EnvironmentRequirement]'",
        "env.requirements_for_method": "(owner: 'type | object', method_name: 'str') -> 'RequirementResult[EnvironmentRequirement]'",
        "world.req": "(*, role: 'str' = 'main', roles: 'Mapping[str, RoleRequirement | Mapping[str, Any]] | None' = None, replicas: 'CountConstraint | int | Mapping[str, int | None] | None' = None, cpus: 'CountConstraint | int | Mapping[str, int | None] | None' = None, memory: 'CountConstraint | int | str | Mapping[str, int | str | None] | None' = None, accelerators: 'Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None' = None, accelerator_memory: 'Mapping[str, CountConstraint | int | str | Mapping[str, int | str | None]] | None' = None, devices: 'Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None' = None, named: 'Mapping[str, CountConstraint | int | Mapping[str, int | None]] | None' = None, topology: 'Mapping[str, Any] | None' = None, source: 'RequirementSource | str | None' = None) -> 'Callable[[T], T]'",
        "world.requirements_for": "(target: 'object') -> 'RequirementResult[WorldRequirement]'",
        "world.requirements_for_method": "(owner: 'type | object', method_name: 'str') -> 'RequirementResult[WorldRequirement]'",
    }


def test_installed_stage_four_exports_resolve_with_exact_root_aliases(
    installed_python: Path,
) -> None:
    """Exercise every installed Stage 4 export and its lazy root-owner identity."""

    result = _installed_probe(
        installed_python,
        """
import json

import dryml
from dryml import annotations, env, environments, requirements, world, worlds

modules = {
    'annotations': annotations,
    'requirements': requirements,
    'environments': environments,
    'worlds': worlds,
}
retired = {
    'annotations': ('decorators', 'env', 'world', 'runtime', 'merge', 'namespaces', 'storage'),
    'environments': %r,
    'worlds': ('default', 'default_for', 'set_default', 'reset_default', 'use_default'),
}
print(json.dumps({
    'exports': {key: list(module.__all__) for key, module in modules.items()},
    'resolved': {
        key: all(getattr(module, name) is module.__dict__[name] for name in module.__all__)
        for key, module in modules.items()
    },
    'unique': {
        key: len(module.__all__) == len(set(module.__all__))
        for key, module in modules.items()
    },
    'aliases': {
        'env': env is environments is dryml.env is dryml.environments,
        'world': world is worlds is dryml.world is dryml.worlds,
        'requirements': requirements is dryml.requirements,
        'root_exports': all(name in dryml.__all__ for name in ('env', 'environments', 'world', 'worlds', 'requirements')),
    },
    'retired': {
        key: [name for name in names if hasattr(modules[key], name)]
        for key, names in retired.items()
    },
}))
""" % (tuple(sorted(_RETIRED_ENVIRONMENT_SURFACE)),),
    )
    data = json.loads(result.stdout)
    assert set(data["exports"]["annotations"]) == _EXPECTED_ANNOTATION_EXPORTS
    assert set(data["exports"]["requirements"]) == _EXPECTED_REQUIREMENT_EXPORTS
    assert set(data["exports"]["environments"]) == _EXPECTED_ENVIRONMENT_EXPORTS
    assert set(data["exports"]["worlds"]) == _EXPECTED_WORLD_EXPORTS
    assert data["resolved"] == {
        "annotations": True,
        "requirements": True,
        "environments": True,
        "worlds": True,
    }
    assert data["unique"] == {
        "annotations": True,
        "requirements": True,
        "environments": True,
        "worlds": True,
    }
    assert data["aliases"] == {
        "env": True,
        "world": True,
        "requirements": True,
        "root_exports": True,
    }
    assert data["retired"] == {"annotations": [], "environments": [], "worlds": []}


def test_installed_root_entry_points_are_effect_free_and_domain_isolated(
    installed_python: Path,
) -> None:
    """Probe each installed root requirement entry point in a fresh interpreter."""

    actions = (
        ("import dryml.requirements", ("dryml.environments", "dryml.worlds")),
        ("assert dryml.env is dryml.environments", ("dryml.worlds",)),
        ("assert dryml.world is dryml.worlds", ("dryml.environments",)),
        (
            "from dryml import env, world\nassert env is dryml.environments\nassert world is dryml.worlds",
            (),
        ),
    )
    for action, inverse in actions:
        result = _installed_probe(
            installed_python,
            """
import json
import os
import platform
import socket
import subprocess
import sys

effects = []
def blocked(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"unexpected {name}")
    return call

os.cpu_count = blocked("cpu_count")
platform.system = blocked("platform.system")
socket.socket = blocked("socket.socket")
subprocess.Popen = blocked("subprocess.Popen")

import dryml
%s

forbidden = %r + %r
print(json.dumps({
    "effects": effects,
    "loaded": sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    ),
}))
""" % (
                action,
                (
                    "dryml.artifacts", "dryml.context", "dryml.core", "dryml.dispatch",
                    "dryml.execute", "dryml.runtime", "dryml.session", "tensorflow",
                    "torch", "jax", "jaxlib", "ray",
                ),
                inverse,
            ),
        )
        assert json.loads(result.stdout) == {"effects": [], "loaded": []}


def test_source_requirement_domain_surface_matches_installed_contract() -> None:
    """Keep source exports aligned with the installed requirement-domain manifest."""

    import dryml

    assert set(dryml.requirements.__all__) == _EXPECTED_REQUIREMENT_EXPORTS
    assert set(dryml.env.__all__) == _EXPECTED_ENVIRONMENT_EXPORTS
    assert set(dryml.world.__all__) == _EXPECTED_WORLD_EXPORTS


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


def test_installed_code_analysis_contract_and_removed_apis(
    installed_python: Path,
) -> None:
    """Require the installed Stage 3 code-analysis package to match its contract."""

    result = _installed_probe(
        installed_python,
        """
import dataclasses
import importlib
import importlib.util
import inspect
import json

import dryml.code

modules = {
    name: list(importlib.import_module(name).__all__)
    for name in %s
}
fields = {}
for dotted in %s:
    module_name, class_name = dotted.rsplit('.', 1)
    fields[dotted] = [field.name for field in dataclasses.fields(
        getattr(importlib.import_module(module_name), class_name)
    )]
signatures = {}
for dotted in %s:
    module_name, function_name = dotted.rsplit('.', 1)
    signatures[dotted] = str(inspect.signature(
        getattr(importlib.import_module(module_name), function_name)
    ))
removed_attributes = (
    'AttrAccess', 'MethodCall', 'collect_accesses_from_source', 'parse_source',
    'ProgramNode', 'ProgramEdge', 'ProgramNodeKind', 'ProgramEdgeKind',
    'build_program_graph', 'TargetKind', 'DescriptorKind', 'normalize_target',
    'KernelContext', 'KernelMode', 'FactScalar', 'FactValue', 'SourceLocation',
    'LexicalDependency', 'LexicalDependencies', 'LexicalDependencyKernel',
    'collect_lexical_dependencies', 'func_source_extract', 'CompilerInfo',
    'Method', 'Traits', 'traits', 'AnnotationFact', 'RequirementFact',
    'MethodContractFact', 'ShapeFact', 'CodeTargetSpec', 'FunctionAnalyzer',
    'register_analyzer', 'get_analyzer', 'available_analyzers', 'CodeProbeRequest',
    'CodeProbeResult', 'normalize_probe_request', 'probe_target',
    'probe_target_in_subprocess', 'request_from_data', 'result_from_data',
    'run_probe_request',
)
removed_modules = (
    'dryml.code.compiler_info', 'dryml.code.method', 'dryml.code.traits',
    'dryml.code.probe_worker', 'dryml.code.transformation',
    'dryml.code.algorithms.direct_annotations',
    'dryml.code.algorithms.method_contracts',
)
print(json.dumps({
    'exports': list(dryml.code.__all__),
    'modules': modules,
    'fields': fields,
    'signatures': signatures,
    'removed_attributes': [name for name in removed_attributes if hasattr(dryml.code, name)],
    'removed_modules': [name for name in removed_modules if importlib.util.find_spec(name) is not None],
}))
""" % (
            repr(tuple(_EXPECTED_CODE_MODULE_EXPORTS)),
            repr(tuple(_EXPECTED_CODE_DATACLASS_FIELDS)),
            repr(tuple(_EXPECTED_CODE_SIGNATURES)),
        ),
    )
    data = json.loads(result.stdout)
    _assert_code_analysis_contract(data)


def test_source_tree_code_analysis_contract_and_removed_apis() -> None:
    """Require the source-tree Stage 3 code-analysis package to match its contract."""

    import dryml.code

    modules = {
        name: list(importlib.import_module(name).__all__)
        for name in _EXPECTED_CODE_MODULE_EXPORTS
    }
    fields = {}
    for dotted in _EXPECTED_CODE_DATACLASS_FIELDS:
        module_name, class_name = dotted.rsplit(".", 1)
        fields[dotted] = [
            field.name
            for field in dataclasses.fields(
                getattr(importlib.import_module(module_name), class_name)
            )
        ]
    signatures = {}
    for dotted in _EXPECTED_CODE_SIGNATURES:
        module_name, function_name = dotted.rsplit(".", 1)
        signatures[dotted] = str(
            inspect.signature(getattr(importlib.import_module(module_name), function_name))
        )
    _assert_code_analysis_contract({
        "exports": list(dryml.code.__all__),
        "modules": modules,
        "fields": fields,
        "signatures": signatures,
        "removed_attributes": [
            name
            for name in (
                "AttrAccess", "MethodCall", "collect_accesses_from_source",
                "parse_source", "ProgramNode", "ProgramEdge", "ProgramNodeKind",
                "ProgramEdgeKind", "build_program_graph", "TargetKind",
                "DescriptorKind", "normalize_target", "KernelContext", "KernelMode",
                "FactScalar", "FactValue", "SourceLocation", "LexicalDependency",
                "LexicalDependencies", "LexicalDependencyKernel",
                "collect_lexical_dependencies", "func_source_extract", "CompilerInfo",
                "Method", "Traits", "traits", "AnnotationFact", "RequirementFact",
                "MethodContractFact", "ShapeFact", "CodeTargetSpec", "FunctionAnalyzer",
                "register_analyzer", "get_analyzer", "available_analyzers",
                "CodeProbeRequest", "CodeProbeResult", "normalize_probe_request",
                "probe_target", "probe_target_in_subprocess", "request_from_data",
                "result_from_data", "run_probe_request",
            )
            if hasattr(dryml.code, name)
        ],
        "removed_modules": [
            name
            for name in (
                "dryml.code.compiler_info", "dryml.code.method", "dryml.code.traits",
                "dryml.code.probe_worker", "dryml.code.transformation",
                "dryml.code.algorithms.direct_annotations",
                "dryml.code.algorithms.method_contracts",
            )
            if importlib.util.find_spec(name) is not None
        ],
    })


def _assert_code_analysis_contract(data: dict[str, object]) -> None:
    """Compare one source-tree or installed contract probe with Stage 3 values."""

    assert data["exports"] == _EXPECTED_CODE_EXPORTS
    assert data["modules"] == _EXPECTED_CODE_MODULE_EXPORTS
    assert data["fields"] == _EXPECTED_CODE_DATACLASS_FIELDS
    assert data["signatures"] == _EXPECTED_CODE_SIGNATURES
    assert data["removed_attributes"] == []
    assert data["removed_modules"] == []


def test_installed_code_and_core_imports_are_passive(
    installed_python: Path,
) -> None:
    """Ensure installed code, core, and symbol imports do not load consumers."""

    result = _installed_probe(
        installed_python,
        """
import json
import sys

import dryml.code
import dryml.core
import dryml.core.symbol

forbidden = (
    'dryml.annotations', 'dryml.artifacts', 'dryml.data', 'dryml.dispatch',
    'dryml.environments', 'dryml.execute', 'dryml.managed', 'dryml.methods',
    'dryml.models', 'dryml.runtime', 'dryml.session', 'dryml.worlds',
    'tensorflow', 'torch', 'jax', 'jaxlib', 'ray',
)
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith(forbidden)
)))
""",
    )
    assert json.loads(result.stdout) == []


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
