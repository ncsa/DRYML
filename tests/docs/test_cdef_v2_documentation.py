"""Keep public CDef V2 documentation free of removed API guidance."""

from pathlib import Path
import re

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
