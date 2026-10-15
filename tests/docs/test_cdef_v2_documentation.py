"""Keep public CDef V2 guidance current and relative documentation links valid."""

from pathlib import Path
import ast
import inspect
import re
from urllib.parse import urlsplit

from dryml.managed.control import ControlSnapshot
from dryml.managed.identity import _operation_digest_from_object_ref_digest


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    *(ROOT / "docs" / name for name in (
        "artifacts.md", "objects_and_defs.md", "immutable_definition_graph.md",
        "repos.md", "managed_operations.md", "formats.md", "graph_querying.md",
        "query_index_backend_contracts.md", "sqlite_lowering.md",
        "ref_selector_values.md", "release_notes.md", "table_of_content.md",
        "testing.md", "execute.md", "signatures.md", "session.md",
        "world_runtime.md", "models.md", "metadata.md", "annotations.md",
    )),
)
RETIRED = re.compile(r"\b(ObjectDef|load_alias|RepoSaveOptions|ephemeral_depth|save_self|dry_args|dry_kwargs)\b")


def test_cdef_v2_docs_have_no_retired_api_examples_or_tutorials():
    """Public guides describe only V2 and the tracked tutorial set is empty."""

    assert not tuple((ROOT / "tutorials").glob("*.ipynb"))
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        assert not RETIRED.search(text), document


def test_local_markdown_links_in_cdef_v2_docs_resolve():
    """Every relative Markdown link in the maintained V2 guides resolves."""

    for document in DOCUMENTS:
        for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", document.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            assert (document.parent / target).exists(), f"{document}: {target}"


def test_stage_2_guides_describe_current_factory_projection_and_metadata_boundaries():
    """Current guides retain the Stage 2 migration and no hidden hook promises."""

    signatures = (ROOT / "docs" / "signatures.md").read_text(encoding="utf-8")
    objects = (ROOT / "docs" / "objects_and_defs.md").read_text(encoding="utf-8")
    querying = (ROOT / "docs" / "graph_querying.md").read_text(encoding="utf-8")
    models = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    metadata = (ROOT / "docs" / "metadata.md").read_text(encoding="utf-8")
    annotations = (ROOT / "docs" / "annotations.md").read_text(encoding="utf-8")

    assert "`__prepare_args__`" not in signatures
    assert re.search(r"They neither inspect a\s+target signature", objects)
    assert re.search(r"Factory identity is the\s+supplied call recipe", models)
    assert re.search(r"named drops are validated against the original\s+selected traversal", querying)
    assert "Migration Only: Constructor Mixins" in metadata
    assert re.search(r"separate from \[Persistent Metadata\]\(metadata\.md\)", annotations)


def test_managed_current_format_documentation_matches_control_codec():
    """The format guide states the current snapshot codec schema and version."""

    object_ref_digest = "b" * 64
    snapshot = ControlSnapshot(
        _operation_digest_from_object_ref_digest(object_ref_digest, "run"),
        object_ref_digest,
        "c" * 64,
        "run",
        "1" * 32,
        "2" * 32,
        1,
        "running",
        None,
        None,
        None,
        None,
        {"version": 1, "store_keys": ["d" * 64], "object_keys": ["e" * 64]},
    )
    codec = snapshot.to_data()
    text = (ROOT / "docs" / "formats.md").read_text(encoding="utf-8")

    assert f"`{codec['schema']}` v{codec['version']}" in text
    assert "`dryml-managed` v1" in text
    assert "`dryml-managed-pending` v1" in text


def test_validation_reuse_guides_and_public_docstrings_keep_authority_boundaries():
    """Current guides/API contracts explain bounded preparation and fresh authority."""
    from dryml.core import Repo
    from dryml.core.cdef_codec import decode_cdef_graph
    from dryml.managed.storage import publish_managed_state

    repos = (ROOT / "docs" / "repos.md").read_text(encoding="utf-8")
    managed = (ROOT / "docs" / "managed_operations.md").read_text(encoding="utf-8")
    graph = (ROOT / "docs" / "immutable_definition_graph.md").read_text(encoding="utf-8")
    normalize = lambda text: " ".join(text.split())
    for phrase in ("256 entries, 32 MiB", "2 MiB per entry", "freshly decodes dill",
                   "Every independent cut still reads current sibling bytes", "pending reservations",
                   "MeanInitial", "MeanUpdate", "MeanFinalize", "uncached behavior",
                   "no permanent payload-validity cache", "Closing that borrower detaches"):
        assert phrase in normalize(repos)
    for phrase in ("one-use completed-publication", "Different coverage or overridden readers",
                   "Every required ZipStore is freshly reopened", "before event delivery",
                   "later status, resume, and recovery", "Every occurrence"):
        assert phrase in normalize(managed)
    assert "repos.md#bounded-snapshot-and-graph-content-reuse" in graph
    assert "Public graph, reference and framed-record decoders always fully validate input" in normalize(graph)
    for callable_, phrases in (
        (Repo.clear_cache, ("live Object cache tiers", "shared epoch", "in-flight")),
        (Repo.close, ("Borrowed", "borrower", "failed close")),
        (decode_cdef_graph, ("fresh private tokens", "complete canonical validation", "CDefGraphCodecError")),
        (publish_managed_state, ("(StateRef, StoreReport)", "fresh authority", "RepoSaveError")),
    ):
        text = normalize(inspect.getdoc(callable_)).lower()
        for phrase in phrases:
            assert phrase.lower() in text, callable_


def test_validation_reuse_owners_keep_one_way_optional_dependencies():
    """Core preparation never imports managed/product policy or optional frameworks."""
    core_files = ("repo.py", "materialization.py", "cdef_codec.py", "reference_values.py",
                  "store/records.py", "store/store.py", "store/dir.py", "store/zip.py")
    managed_files = ("storage.py", "context.py", "runtime.py", "control.py")
    optional = {"tensorflow", "torch", "jax", "flax", "optax", "ray"}
    for package, files in (("core", core_files), ("managed", managed_files)):
        for name in files:
            path = ROOT / "src" / "dryml" / package / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    assert module.split(".")[0] not in optional, (path, node.lineno, module)
                    if package == "core":
                        assert not any(module == forbidden or module.startswith(forbidden + ".")
                                       for forbidden in ("dryml.managed", "dryml.models", "dryml.artifacts",
                                                         "managed", "models", "artifacts")), (path, node.lineno, module)
