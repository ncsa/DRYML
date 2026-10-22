"""Dependency-light importable fixtures for exact managed worker qualification."""

from pathlib import Path

from dryml.core import Serializable
from dryml.managed import managed_operation


class WorkerValue(Serializable):
    """Persist a counter and record every effect outside the Store payload."""

    def __init__(self, marker, value=0):
        """Initialize the recipe counter and append construction evidence."""
        self.marker, self.value = marker, value
        self.note("construct")

    def note(self, event):
        """Append one deterministic hook/body event to this test's marker."""
        with Path(self.marker).open("a", encoding="ascii") as stream:
            stream.write(event + "\n")

    def save_state_to_dir_imp(self, directory, *, codec):
        """Capture exactly this counter and record the capture boundary."""
        self.note("save")
        Path(directory, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, directory, *, codec):
        """Restore exactly this counter and record the restore boundary."""
        self.note("restore")
        self.value = int(Path(directory, "value").read_text(encoding="ascii"))

    @managed_operation(resumable=True)
    def advance(self, amount=3, *, managed):
        """Checkpoint/fail once when requested; resume from the selected counter."""
        self.note("body")
        if Path(self.marker + ".fail").exists() and not managed.is_resuming:
            self.value += amount
            managed.checkpoint()
            raise ValueError("controlled worker interruption")
        self.value += amount
        return {"value": self.value, "resuming": managed.is_resuming}

    def _managed_post_publication(self, final, repo, report):
        self.note("consumer")


class Branch(Serializable):
    """Share read-only dataset identities while mutating an independent model."""

    def __init__(self, data, repeated, model):
        """Retain exact shared dataset edges and an independent model edge."""
        self.data, self.repeated, self.model = data, repeated, model

    @managed_operation()
    def run(self, *, managed):
        """Return detached topology/identity facts after one independent mutation."""
        assert self.data is self.repeated
        self.model.value += 1
        return str(self.data.object_id), str(self.model.object_id), self.model.value
