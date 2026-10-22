"""Invocation-scoped identity/fence value for the Node managed lifecycle handoff."""

from .errors import ManagedContextError, ManagedControlError


class ManagedContext:
    """Read-only active admission identity and selected authority references.

    Runtime-created contexts borrow control_store/state_repo, retain an active
    operation admission, and expose run/invocation/generation distinctly. Foreign
    process/thread/task or retired use raises ManagedContextError. No Core graph
    reservation lasts through execution. Checkpoint/final policy is integrated by
    U16; the U14 context fails explicitly before legacy publication effects.
    """

    def __init__(self, token, *, admission, control, generation):
        """Bind private admission/control/generation; invalid token raises ManagedContextError.

        Inputs are an active OperationAdmission, ManagedControlStore and integer
        fence. Returns None; no Store IO, state capture or ownership acquisition.
        """
        if token is not _CONTEXT_TOKEN:
            raise ManagedContextError("context_construction", "managed contexts are runtime-created")
        self._admission, self._control, self._generation = admission, control, generation
        self._active = True

    @property
    def active(self):
        """Return bool for the original caller's still-active invocation lifetime."""
        return self._active and self._admission.active

    @property
    def operation_id(self):
        """Return stable operation digest (str), without IO."""
        return self._admission.identity.operation_id

    @property
    def run_id(self):
        """Return stable run UUID hex (str), retained across ordinary resume."""
        return self._admission.run_id

    @property
    def invocation_id(self):
        """Return fresh caller invocation UUID hex (str), distinct from producer."""
        return self._admission.invocation_id

    @property
    def generation(self):
        """Return this context's integer authority fence; it is not a fresh observation."""
        return self._generation

    @property
    def state_repo(self):
        """Return borrowed selected Repo without changing its lifetime or routing."""
        return self._control.state_repo

    @property
    def control_store(self):
        """Return borrowed selected control DirStore without opening/closing it."""
        return self._control.control_store

    def checkpoint(self):
        """Validate the active fence then explicitly refuse unintegrated U16 policy.

        Returns no value. Raises ManagedContextError for inactive callers,
        ManagedConflictError for stale authority, otherwise ManagedControlError.
        Performs only metadata inspection; no capture, grant or payload effects.
        """
        self._require_active()
        current = self._control.inspect(self.operation_id)
        self._control._fence(current, self.operation_id, self._admission,
                             self.run_id, self.invocation_id, self.generation)
        raise ManagedControlError("checkpoint_not_integrated", "Node checkpoint policy awaits U16")

    def interrupt(self, *, cause=None):
        """Validate optional BaseException cause and fence; refuse unintegrated U16 policy.

        Returns no value and raises as checkpoint(), or ManagedContextError for an
        invalid cause. This primitive neither saves state nor marks success.
        """
        if cause is not None and not isinstance(cause, BaseException):
            raise ManagedContextError("invalid_cause", "cause must be BaseException or None")
        self.checkpoint()

    def _require_active(self):
        if not self.active:
            raise ManagedContextError("inactive_context", "managed context is inactive or belongs to another caller")

    def _deactivate(self): self._active = False


_CONTEXT_TOKEN = object()


def _create_context(**kwargs):
    return ManagedContext(_CONTEXT_TOKEN, **kwargs)


__all__ = ["ManagedContext"]
