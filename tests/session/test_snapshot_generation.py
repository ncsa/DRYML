"""Held-generation session snapshot contracts."""

from dryml.runtime import publication
from dryml.session import snapshot_for_generation


def test_snapshot_for_generation_does_not_reread_current_publication(
        monkeypatch,
):
    """A held generation is projected directly when current would change."""

    held = publication.current()
    monkeypatch.setattr(
        publication,
        "current",
        lambda: (_ for _ in ()).throw(AssertionError("must not reread")),
    )

    snapshot = snapshot_for_generation(held)

    assert snapshot.generation == held.number
    assert snapshot.runtime.mode == held.runtime.mode
