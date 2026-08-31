"""Keep public CDef V2 documentation free of removed API guidance."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    *(ROOT / "docs" / name for name in (
        "artifacts.md", "objects_and_defs.md", "immutable_definition_graph.md",
        "repos.md", "formats.md", "graph_querying.md",
        "query_index_backend_contracts.md", "sqlite_lowering.md",
        "ref_selector_values.md", "release_notes.md", "table_of_content.md",
        "testing.md",
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
