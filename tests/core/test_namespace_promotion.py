"""Characterize the public core namespace after the promotion."""

import importlib
import os
from pathlib import Path
import subprocess
import sys


def _run_fresh_process(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with this checkout's source tree first on ``PYTHONPATH``."""

    src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(src), env.get("PYTHONPATH", "")))
    return subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def test_root_exports_core_lazily_with_the_existing_public_symbols():
    """The lazy root router exposes the promoted package and its existing symbols."""

    _run_fresh_process(
        """
import sys
import dryml

assert "core" in dryml.__all__
assert "dryml.core" not in sys.modules
core = dryml.core
assert core is importlib.import_module("dryml.core")
assert dryml.Definition is core.Definition
assert dryml.definition_mode is core.definition_mode
assert dryml.F is dryml.FactorySpec is core.F is core.FactorySpec
assert not {"tensorflow", "torch"} & sys.modules.keys()
assert "dryml.core" + "2" not in sys.modules
""".replace("import sys", "import importlib\nimport sys")
    )


def test_promoted_package_keeps_the_destination_export_manifest():
    """The package export manifest remains the destination manifest after promotion."""

    core = importlib.import_module("dryml.core")

    assert core.__all__ == [
        "load_object", "save_object", "load_state_ref", "LiveReusePolicy", "StoreReport", "SavePublication", "SavedSnapshot", "PublicationPhase", "PublicationStatus", "SaveRouting", "StateGraphReservation", "Object", "Serializable",
        "Compute", "Definition", "Node", "ConcreteNode", "SavedNode", "require_saved", "StateKey", "LocalState", "IdentitySpec", "StateSpec", "categorical_definition", "ConcreteDefinition", "DefLink", "Ref", "Mat", "AutoRef",
        "normalize_args", "normalize_return", "signature_context", "function", "authoring_helper", "SignatureError",
        "ObjectId", "object_namespace", "freeze",
        "QuotedDef", "SelectorSpec", "Selector", "QueryPredicate", "selector", "Par", "Expr", "Shared", "Template", "Generator", "GeneratorSelector", "repeat", "Match", "Distribution", "ParameterizationError", "UnresolvedDefinitionError", "ParameterizationLimitError", "UnsupportedGeneratorVerificationError", "Present",
        "Missing", "AnyValue", "Exact", "Choice", "IntRange", "SubclassOf", "Satisfies",
        "UniformIntRange", "UniformFromSet", "SKIP_ARGS", "Repo", "MetadataConflictError", "RepoDefinition", "RepoDefinitionError", "RepoReconstructionError", "configure",
        "reset_config", "status", "definition_mode", "selector_mode", "dtype",
        "DType", "ConfigRef", "F", "FactorySpec", "ConfigError", "CONFIG_MISSING", "as_tensor_spec",
        "SpecHint", "TensorSpec", "ImportRef", "SourceSpec", "symbol_ref", "resolve_symbol",
        "CDefEdge", "CDefNode", "CDefOccurrence", "ConcreteDefinitionGraph",
        "ConcreteDefinitionGraphCycleError", "ConcreteDefinitionGraphError", "EdgeKind",
        "iter_direct_cdef_edges", "Arg", "DefinitionPath", "EdgePolicy",
        "GraphPathError", "Index", "Key", "Kwarg", "Parameter", "IdentityQuery", "IdentitySet",
        "Occurrence", "OccurrenceQuery", "OccurrenceSet", "ObjectResultSet", "RelationshipKind", "RelationshipPath", "MetadataField", "MetadataPredicate", "field", "QueryCardinalityError", "QueryDomainError", "QueryDiagnostic", "QueryError",
        "QueryExplanation", "QueryIndexError", "QueryPathError", "SetMember", "SourceEvidence", "intersection", "union",
        "CoreExecutionError", "CoreExecutionFuture", "CoreExecutionSnapshot", "CoreOptions",
        "CoreOutcomeEvidence", "CorePublicationEvidence", "CoreRefreshEvidence", "Executor",
        "ExecutorView", "PreparedCoreCall", "SharedDirStoreStrategy", "run", "submit",
        "SaveAnnotations", "NodeLineageMetadata", "NodeSnapshotMetadata", "SnapshotCapture",
        "encode_metadata_mapping", "decode_metadata_mapping", "timestamp_to_seconds",
        "timestamp_from_seconds", "MetadataScalar", "MetadataValue",
        "MetadataMapping", "MetadataTarget", "MetadataDiagnostic", "EnvironmentStatus",
        "RequirementStatus", "EvidenceCoverage",
    ]
