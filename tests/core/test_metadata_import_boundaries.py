"""Fresh-process import and observation boundaries for Store metadata."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "store_v3"


def test_fresh_process_inspects_metadata_without_heavy_imports_or_probes(tmp_path):
    """Core metadata inspection stays lightweight and never observes the host."""

    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import shutil
        import sys

        heavy = {{"jax", "tensorflow", "torch"}}
        assert not heavy.intersection(sys.modules)

        import dryml.core as core
        import dryml.environments as environments
        import dryml.records as records
        from dryml.core.store.dir import DirStore

        assert not heavy.intersection(sys.modules)
        assert "dryml.code" not in sys.modules
        assert "dryml.environments.introspection" not in sys.modules
        assert "dryml.environments.probe" not in sys.modules

        source = Path({str(FIXTURE_ROOT)!r})
        destination = Path({str(tmp_path / "store")!r})
        shutil.copytree(source / "dir-store", destination)
        manifest = json.loads((source / "manifest.json").read_text(encoding="ascii"))
        store = DirStore.open_existing(destination)
        repo = core.Repo(store)
        states = list(repo.references().state_refs())
        assert len(states) == len(manifest["snapshots"])
        assert all(repo.get_snapshot_metadata(state, store=store).environment_status == "known" for state in states)
        assert repo.references().where(core.field("snapshot", "requirements_status").eq("conflict")).state_refs().count() == 1

        assert not heavy.intersection(sys.modules)
        assert "dryml.code" not in sys.modules
        assert "dryml.environments.introspection" not in sys.modules
        assert "dryml.environments.probe" not in sys.modules
        store.close()
        """
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
