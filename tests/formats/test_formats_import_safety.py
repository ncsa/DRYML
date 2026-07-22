import os
from pathlib import Path
import subprocess
import sys


def run_probe(code: str):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env=env)


def test_formats_import_is_lightweight():
    run_probe(
        """
import sys
import dryml.formats
assert "dryml.environments" not in sys.modules
assert "dryml.core" not in sys.modules
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules
assert "jax" not in sys.modules
assert "ray" not in sys.modules
        """
    )


def test_top_level_dryml_formats_export_is_lazy():
    run_probe(
        """
import sys
import dryml
assert "dryml.formats" not in sys.modules
_ = dryml.formats
assert "dryml.formats" in sys.modules
assert "dryml.environments" not in sys.modules
assert "dryml.core" not in sys.modules
        """
    )


def test_formats_docs_page_exists_and_is_linked():
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    text = (docs_dir / "formats.md").read_text()
    toc = (docs_dir / "table_of_content.md").read_text()

    assert "canonical JSON" in text
    assert "content_id" in text
    assert "cdef-v4" in text
    assert '"$literal"' in text
    assert "formats.md" in toc
