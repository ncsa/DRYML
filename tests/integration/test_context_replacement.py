import ast
from pathlib import Path

from dryml.worlds.legacy import lower_legacy_resource_requirement, lower_legacy_resource_spec


def test_new_world_runtime_code_does_not_import_context():
    root = Path(__file__).parents[2] / "src" / "dryml"
    for package in (root / "worlds", root / "runtime"):
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(alias.name != "dryml.context" for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "dryml.context"


def test_legacy_resource_lowering_routes_to_worlds():
    spec = lower_legacy_resource_spec({"num_cpus": 2, "num_gpus": 1})
    req = lower_legacy_resource_requirement({"num_cpus": 2, "num_gpus": 1})

    assert spec.to_data()["accelerators"] == {"gpu": 1}
    assert req.to_data()["accelerators"]["gpu"] == {"min": 1}
