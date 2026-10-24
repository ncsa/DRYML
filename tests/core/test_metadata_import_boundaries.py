"""Fresh-process import and observation boundaries for Store metadata."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.environments import req


@req(requirements=("metadata-boundary<1",))
class MetadataLeaf(Serializable):
    """Synthetic private-state owner with one passive requirement declaration."""

    def save_state_to_dir_imp(self, destination, *, codec):
        """Write a deterministic payload into the test-owned staging directory."""
        Path(destination, "value").write_bytes(b"payload")


@req(requirements=("metadata-boundary>=1",))
class MetadataConflictRoot(MetadataLeaf):
    """Combine contradictory passive declarations for metadata-only query proof."""

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "store_v3"


def test_fresh_process_inspects_metadata_without_heavy_imports_or_probes(tmp_path, fixed_snapshot_environment):
    """Core metadata inspection stays lightweight and never observes the host."""

    current = tmp_path / "current-store"
    repo = Repo(DirStore(current))
    repo.save_object(MetadataLeaf(repo=repo), alias="plain")
    repo.save_object(MetadataConflictRoot(repo=repo), alias="conflict")
    repo.close(flush=False)
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import shutil
        import sys
        import hashlib
        import pytest

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
        before = {{str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in destination.rglob("*") if p.is_file()}}
        from dryml.core.store.store import StoreAuthorityError
        with pytest.raises(StoreAuthorityError):
            DirStore.open_existing(destination)
        assert before == {{str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in destination.rglob("*") if p.is_file()}}
        store = DirStore.open_existing({str(current)!r})
        repo = core.Repo(store)
        states = list(repo.query().saved().collect())
        assert len(states) == 2
        captures = [store.read_node(store.read_node_name(name).root_digest, kind="saved")
                    for name in ("plain", "conflict")]
        assert all(repo.get_snapshot_metadata(state, store=store).environment_status == "known" for state in captures)
        assert repo.query().where(core.field("snapshot", "requirements_status").eq("conflict")).saved().count() == 1

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
