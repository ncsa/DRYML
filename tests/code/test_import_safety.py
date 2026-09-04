"""Tests for the incremental dependency-light package manifest."""

from __future__ import annotations

import subprocess
import sys


def test_code_exports_only_implemented_u2_apis() -> None:
    """The root manifest exposes the graph but excludes later Stage 3 names."""

    import dryml.code as code

    assert "func_source_extract" not in code.__all__
    assert "CompilerInfo" not in code.__all__
    assert "normalize_target" in code.__all__
    assert "ProgramGraph" in code.__all__
    assert "build_program_graph" not in code.__all__
    assert not hasattr(code, "func_source_extract")
    assert not hasattr(code, "CompilerInfo")


def test_fresh_code_import_loads_no_product_or_optional_packages() -> None:
    """Importing the public package performs no analysis or product imports."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.code; "
            "print(','.join(sorted(name for name in sys.modules if name.startswith('dryml.'))))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = set(filter(None, result.stdout.strip().split(",")))
    assert loaded <= {
        "dryml.code",
        "dryml.code.ast_tools",
        "dryml.code.callable_info",
        "dryml.code.errors",
        "dryml.code.facts",
        "dryml.code.graph",
        "dryml.code.source",
        "dryml.code.targets",
        "dryml._framework_imports",
    }
