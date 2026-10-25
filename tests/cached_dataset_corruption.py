"""Prepare malformed codec content without corrupting published Store inventory."""

from hashlib import sha256
import json
from pathlib import Path


def capture_cache_payload(cache, directory):
    """Copy a completed cache through its public writer into an empty test Path.

    The caller may rewrite this private copy without mutating the original native
    StateKey. Writer/authentication failures propagate to the test.
    """

    directory.mkdir()
    cache.save_state_to_dir(str(directory), codec=cache.state_codec)


def publish_cache_payload(cache, repo, directory):
    """Authenticate rewritten codec bytes through current capture and exact load.

    Args:
        cache: Completed CachedDataset whose private copy is being rewritten.
        repo: Repo owning the current native Store used for fresh publication.
        directory: Path to the public-writer copy, after controlled mutation.

    Returns:
        A fresh public exact-load CachedDataset with native source selection.

    Refreshes only the copy's logical chunk size/digest descriptors, restores its
    eager metadata, then uses ordinary public capture to derive a new authoritative
    inventory/StateKey. No published Store files or native readers are patched.
    Publication, native authentication, and metadata validation failures propagate;
    logically malformed chunks must survive these steps and fail during iteration.
    """

    manifest_path = directory / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for descriptor in manifest["chunks"]:
        content = (directory / "chunks" / descriptor["file"]).read_bytes()
        descriptor["size"] = len(content)
        descriptor["sha256"] = sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache.restore_state_from_dir(str(directory), codec=cache.state_codec)
    saved = repo.save_object(cache)
    with repo.default_store.read_state(saved.state.key) as source:
        assert source.state == saved.state
        assert json.loads(
            Path(source.data_directory, "cache-manifest.json").read_text(encoding="utf-8")
        ) == manifest
    restored = repo.load_state_ref(saved, reuse_live="never")
    assert restored.ready and len(restored) == manifest["count"]
    return restored
