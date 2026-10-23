"""Artifact completion discovery and explicit SavedNode materialization.

Discovery reads metadata even after private-data cleanup. Materialization raises
on unavailable state; this module never substitutes recomputation for recovery.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from dryml.core import ConcreteDefinition, Node, Serializable
from dryml.core.node import require_saved
from dryml.managed import ManagedContext, managed_operation


class ArtifactRecoveryError(RuntimeError):
    """Raised when exact completed Artifact recovery is ambiguous or invalid.

    Artifact recovery accepts only one managed-completed immutable SavedNode for a
    fully bound recipe. Callers must retain a more specific initial SavedNode when
    identical recipe definitions legitimately have multiple completed results.
    """


class Artifact(Serializable):
    """Abstract repo-backed computed payload with subclass-owned content.

    Concrete subclasses implement one managed ``compute`` operation and a
    boolean ``ready`` property. Artifact supplies exact managed-completion
    discovery and recovery, but does not rerun work, define payload validation,
    or own input-reference policy; those remain with the concrete domain class.
    """

    @classmethod
    def find_completed_state_ref(
            cls, recipe: "Artifact | ConcreteDefinition", *, repo,
            control_store=None) -> Node | None:
        """Return the unique managed-completed exact state for one recipe.

        Args:
            recipe: Fully bound live Artifact or its exact ConcreteDefinition.
                Its definition is queried without materializing Artifact states.
            repo: Connected Repo supplying immutable Node and managed-control
                authority.
            control_store: Optional exact managed control DirStore. Omitting it
                selects the Repo default control Store.

        Returns:
            The one exact completed SavedNode, or ``None`` when no matching managed
            operation has completed. Private-data availability is not required.

        Raises:
            TypeError: If ``recipe`` is not an Artifact or ConcreteDefinition.
            ArtifactRecoveryError: If more than one distinct completed result is
                valid for the recipe, or completion authority names another recipe.
            dryml.managed.ManagedError: If retained completion authority is corrupt
                or its exact Node metadata closure is invalid.

        Side Effects:
            Reads Repo reference evidence and managed control records only. It does
            not construct candidates, load payloads, or materialize input refs.
        """

        definition = _recipe_definition(recipe)
        completed = {}
        for state_ref in repo.query().saved().sel(definition).iter():
            completed_state = _completed_state_for_reference(
                state_ref, repo=repo, control_store=control_store,
            )
            if completed_state is not None:
                completed[completed_state.exact_digest()] = completed_state
        if len(completed) > 1:
            raise ArtifactRecoveryError(
                "Artifact recipe has multiple distinct completed states; recover "
                "through its exact initial StateRef."
            )
        return next(iter(completed.values()), None)

    @classmethod
    def load_completed(
            cls, recipe: "Artifact | ConcreteDefinition", *, repo,
            control_store=None, reuse_live: str = "matching") -> "Artifact | None":
        """Load the unique completed Artifact result for one fully bound recipe.

        Args:
            recipe: Fully bound live Artifact or its exact ConcreteDefinition.
            repo: Connected Repo supplying immutable result authority.
            control_store: Optional exact managed control DirStore.
            reuse_live: Exact-load policy accepted by Repo.load_state_ref. The
                default reuses one matching live completed Artifact identity.

        Returns:
            The unique ready Artifact restored from its exact completed SavedNode,
            or ``None`` when no completed result exists.

        Raises:
            ArtifactRecoveryError: If result selection is ambiguous, restored as a
                different Artifact type, or does not validate as ready.
            dryml.core.RepoLoadError: If the selected exact result is missing or
                corrupt. The error is not replaced with recomputation.
            ValueError: If ``reuse_live`` is not a supported exact-load policy.

        Side Effects:
            Performs at most one exact Artifact restoration after authority
            selection. Matching live exact identities may be returned directly.
        """

        state_ref = cls.find_completed_state_ref(
            recipe, repo=repo, control_store=control_store,
        )
        if state_ref is None:
            return None
        return cls._load_ready_state(state_ref, repo=repo, reuse_live=reuse_live)

    @classmethod
    def recover(
            cls, initial_state_ref: Node, *, repo, control_store=None,
            reuse_live: str = "matching") -> "Artifact":
        """Recover a completed result or its exact initial managed receiver.

        Args:
            initial_state_ref: Complete SavedNode for the initial Artifact retained before
                managed computation or result-row publication.
            repo: Connected Repo supplying initial/final state authority.
            control_store: Optional exact managed control DirStore.
            reuse_live: Exact-load policy accepted by Repo.load_state_ref.

        Returns:
            The ready completed Artifact when its same managed operation completed;
            otherwise the initial Artifact receiver for the same operation, so a
            failed or interrupted invocation can resume through ``compute``.

        Raises:
            TypeError, ValueError: If ``initial_state_ref`` is not a complete SavedNode.
            ArtifactRecoveryError: If the recovered state is not this Artifact type
                or a completed result is not ready.
            dryml.core.RepoLoadError: If the selected final (or uncompleted initial)
                SavedNode's private state is missing or corrupt. An unselected
                initial payload is not opened to recover completed work.
            ValueError: If ``reuse_live`` is not a supported exact-load policy.

        Side Effects:
            Validates retained immutable authority before loading exactly one chosen
            state. It never constructs alternative recipe candidates or invokes
            ``compute``.
        """

        require_saved(initial_state_ref)
        completed_state = _completed_state_for_reference(
            initial_state_ref, repo=repo, control_store=control_store,
        )
        return cls._load_ready_state(
            initial_state_ref if completed_state is None else completed_state,
            repo=repo, reuse_live=reuse_live, require_ready=completed_state is not None,
        )

    @classmethod
    def _load_ready_state(
            cls, state_ref: Node, *, repo, reuse_live: str,
            require_ready: bool = True) -> "Artifact":
        """Restore one already-selected exact Artifact state and validate readiness."""

        if reuse_live not in {"matching", "greedy", "never"}:
            raise ValueError("reuse_live must be 'matching', 'greedy', or 'never'.")
        from dryml.core.repo import RepoSaveError

        try:
            loaded = repo._load_state_ref_with_identity_reservation(
                state_ref, reuse_live=reuse_live,
            )
        except RepoSaveError as error:
            raise ArtifactRecoveryError(
                "Artifact recovery conflicts with an actively owned live state graph."
            ) from error
        if not isinstance(loaded, cls):
            raise ArtifactRecoveryError("Exact Artifact state restored an incompatible receiver type.")
        if require_ready and not loaded.ready:
            raise ArtifactRecoveryError("Managed completed Artifact state is not ready.")
        return loaded

    @managed_operation()
    @abstractmethod
    def compute(self, *args: Any, managed: ManagedContext, **kwargs: Any) -> Any:
        """Compute this Artifact's content through DRYML's managed lifecycle.

        Args:
            *args: Domain-specific positional arguments.
            managed: Framework-provided managed operation context.
            **kwargs: Domain-specific keyword arguments.

        Returns:
            InvocationResult with detached run/invocation metadata, the admitted
            ordinary result in ``.value``, and the completed SavedNode in
            ``.final_state_ref``. Custom summary objects are not durable results;
            concrete bodies must return closed KTD11 values or supported metadata.

        Raises:
            dryml.managed.ManagedError: If managed lifecycle handling fails.

        Side Effects:
            Concrete implementations may update their own content and publish
            immutable state through the managed-operation lifecycle.
        """

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Return whether this Artifact's current content is usable.

        Returns:
            ``True`` only when the subclass considers its current payload
            complete and usable.

        Side Effects:
            Implementations must not compute or materialize input references
            merely to answer readiness.
        """

Artifact.__module__ = "dryml.artifacts"


def _recipe_definition(recipe: Artifact | ConcreteDefinition) -> ConcreteDefinition:
    """Return the exact recipe definition without realizing retained inputs."""

    if isinstance(recipe, Artifact):
        return recipe.definition
    if isinstance(recipe, ConcreteDefinition):
        return recipe
    raise TypeError("Artifact recipe must be an Artifact or ConcreteDefinition.")


def _completed_state_for_reference(state_ref: Node, *, repo, control_store) -> Node | None:
    """Read one receiver's managed completion receipt without materialization."""

    from dryml.managed.control import ManagedControlStore
    from dryml.managed.identity import operation_digest

    control = ManagedControlStore(
        repo.default_store if control_store is None else control_store, repo,
    )
    snapshot = control.inspect(operation_digest(state_ref, "compute"))
    if snapshot is None or snapshot.state != "completed":
        return None
    if snapshot.final is None:
        raise ArtifactRecoveryError("Managed completed authority lacks an exact final StateRef.")
    completed = snapshot.final
    if not completed.object_projection_equal(state_ref):
        raise ArtifactRecoveryError("Managed completion authority has an incompatible final StateRef.")
    return completed


ArtifactRecoveryError.__module__ = "dryml.artifacts"
