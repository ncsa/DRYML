"""Keep Node/Store-4 guidance, public contracts and documentation links current."""

from pathlib import Path
import ast
import inspect
import json
import re
import subprocess
import sys
from urllib.parse import urlsplit

from dryml.managed.control import ControlSnapshot
from dryml.core import Definition, Object


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
        "dispatch.md", "object_state_lifecycle_verification.md", "ml_workflow_qualification.md",
    )),
)
RETIRED = re.compile(r"\b(ObjectDef|load_alias|RepoSaveOptions|ephemeral_depth|save_self|dry_args|dry_kwargs)\b")


def test_object_state_docs_have_no_retired_api_examples_or_tutorials():
    """Maintained guides omit retired APIs; the tracked tutorial set is empty."""

    assert not tuple((ROOT / "tutorials").glob("*.ipynb"))
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        assert not RETIRED.search(text), document


def test_local_markdown_links_in_object_state_docs_resolve():
    """Every relative Markdown link in the maintained guides resolves."""

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

    receiver = Definition(Object).concretize()
    snapshot = ControlSnapshot(
        receiver, "run", "c" * 64, "1" * 32, "2" * 32, 1, "running",
    )
    codec = snapshot.to_data()
    text = (ROOT / "docs" / "formats.md").read_text(encoding="utf-8")

    assert f"`{codec['schema']}` v{codec['version']}" in text
    assert "`dryml-managed` v2" in text
    assert "`dryml-managed-pending` v2" in text


def test_validation_reuse_guides_and_public_docstrings_keep_authority_boundaries():
    """Current guides/API contracts explain bounded preparation and fresh authority."""
    from dryml.core import Repo
    from dryml.core.cdef_codec import decode_cdef_graph
    from dryml.managed.storage import _publish_checkpoint

    repos = (ROOT / "docs" / "repos.md").read_text(encoding="utf-8")
    managed = (ROOT / "docs" / "managed_operations.md").read_text(encoding="utf-8")
    graph = (ROOT / "docs" / "immutable_definition_graph.md").read_text(encoding="utf-8")
    normalize = lambda text: " ".join(text.split())
    for phrase in ("no permanent payload-validity cache", "Closing that borrower detaches"):
        assert phrase in normalize(repos)
    for phrase in ("InvocationResult", "run_id", "invocation_id", "start_from", "final_state"):
        assert phrase in normalize(managed)
    assert "repos.md#bounded-snapshot-and-graph-content-reuse" in graph
    assert "Node" in graph
    for callable_, phrases in (
        (Repo.clear_cache, ("live Object cache tiers", "shared epoch", "in-flight")),
        (Repo.close, ("Borrowed", "borrower", "failed close")),
        (decode_cdef_graph, ("fresh private tokens", "complete canonical validation", "CDefGraphCodecError")),
        (_publish_checkpoint, ("(SavedNode, StoreReport)", "provisional", "Reservations")),
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


def test_routine_roster_references_current_maintained_functions():
    """A stale representative must fail visibly instead of silently losing proof."""
    profile = json.loads((ROOT / "tests/test_profiles.json").read_text())["profiles"]["good-enough"]
    missing = []
    for relative, selection in profile["representative_functions"].items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        names = {n.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}
        missing.extend(f"{relative}::{name}" for name in selection["functions"] if name not in names)
    assert not missing, "Stale routine representatives: " + ", ".join(missing)


def test_public_exports_use_one_node_carrier_and_detached_invocation_results():
    """SavedNode is a typing refinement; runtime admission keeps the same graph."""
    from dryml.core import Node, ConcreteNode, SavedNode, require_saved
    from dryml.managed import InvocationResult, ExecutionIdentity, ResultSource
    from typing import get_origin
    soft = Definition(Object)
    concrete = soft.concretize()
    assert isinstance(soft, Node) and isinstance(concrete, ConcreteNode)
    assert get_origin(SavedNode) is not None
    assert require_saved(concrete) is concrete
    assert {"execution", "final_state_ref", "result_reference", "source"} <= set(InvocationResult.__dataclass_fields__)
    assert "run_id" in ExecutionIdentity.__dataclass_fields__
    assert "invocation_id" in ExecutionIdentity.__dataclass_fields__
    assert not {"attempt_id", "return_state_ref"} & set(InvocationResult.__dataclass_fields__)
    assert "store_uuid" in ResultSource.__dataclass_fields__


def test_readme_workflow_executes_in_a_fresh_lightweight_process(tmp_path):
    """The maintained public example restores actual mutated state and identity."""
    text = (ROOT / "README.md").read_text()
    script, = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    script = script.replace('"./checkpoint-store"', repr(str(tmp_path / "store")))
    script += "\nimport sys\nassert not {'tensorflow', 'torch', 'jax', 'ray'} & set(sys.modules)\n"
    source = tmp_path / "readme_workflow.py"
    source.write_text(script)
    completed = subprocess.run([sys.executable, str(source)], cwd=tmp_path,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_query_guides_distinguish_recipe_build_from_saved_node_load():
    """All three query guides name the explicit build and saved-load boundaries."""
    for name in ("graph_querying.md", "repos.md", "objects_and_defs.md"):
        text = " ".join((ROOT / "docs" / name).read_text().split())
        assert "repo.build(" in text, name
        assert "repo.load(saved_node)" in text, name
        assert "repo.load_state_ref(saved_node)" in text, name
    querying = (ROOT / "docs" / "graph_querying.md").read_text()
    assert "model = repo.build(models.one())" in querying
    assert "repo.load_object(models.one())" not in querying


def test_graph_query_recipe_example_builds_stateful_model(tmp_path):
    """Execute the guide against mutable saved state, then verify exact saved loads."""
    readme = (ROOT / "README.md").read_text()
    script, = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    store_path = repr(str(tmp_path / "store"))
    script = script.replace('"./checkpoint-store"', store_path)
    querying = (ROOT / "docs" / "graph_querying.md").read_text()
    example, = (block for block in re.findall(r"```python\n(.*?)```", querying, flags=re.S)
                if "models = repo.query()" in block)
    script += f"\nModel = Counter\nrepo = Repo(DirStore({store_path}))\n"
    script += example
    script += """
assert model.value == 3, "recipe construction must not restore the saved value 5"
assert model.object_id != state.object_id
assert model.last_state_ref is None
recipe = models.one()
assert recipe.object_id is None and recipe.state is None
for load in (repo.load, repo.load_state_ref, repo.load_object):
    try:
        load(recipe)
    except ValueError:
        pass
    else:
        raise AssertionError("a stateful recipe is not a complete SavedNode")
saved_node = repo.query().sel(Definition(Model)).saved().stored().collect().one()
assert saved_node.graph_equal(state)
for load in (repo.load, repo.load_state_ref):
    loaded = load(saved_node)
    assert loaded.value == 5
    assert loaded.object_id == state.object_id
assert not {'tensorflow', 'torch', 'jax', 'ray'} & set(__import__('sys').modules)
"""
    source = tmp_path / "graph_query_workflow.py"
    source.write_text(script)
    completed = subprocess.run([sys.executable, str(source)], cwd=tmp_path,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_curated_backend_rosters_reference_current_paths_and_functions():
    """Curated CI paths stay discoverable without activating optional backends."""
    for roster in ("framework_tests.txt", "native_estimator_tests.txt"):
        for target in (ROOT / "tests" / roster).read_text().splitlines():
            path, *selected = target.split("::")
            source = ROOT / path
            assert source.exists(), (roster, target)
            if selected:
                names = {n.name for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef)}
                assert len(selected) == 1 and selected[0] in names, (roster, target)
