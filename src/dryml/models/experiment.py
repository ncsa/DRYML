from __future__ import annotations

import functools
import inspect
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping

from dryml.artifacts import Artifact, Value
from dryml.core import Ref, StateRef, Template
from dryml.core.object import Serializable
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.managed import (
    ManagedConfig,
    ManagedConfigError,
    ManagedRerunRequiredError,
    managed_operation,
)
from dryml.managed.descriptor import ManagedOperation

from .experiment_data import ExperimentData
from .measurements import MeasurementUnavailableError, dataset_size, model_parameter_counts
from .train_spec import TrainState


def _experiment_train_wrapper(target):
    """Prepend Experiment evaluation without mutating caller managed controls."""

    @functools.wraps(target)
    def wrapped(self, *, managed=None):
        config = ManagedConfig() if managed is None else managed
        callbacks = [] if config.callbacks is None else list(config.callbacks)
        if len(callbacks) >= 64:
            raise ManagedConfigError(message="Experiment evaluation leaves room for at most 63 caller callbacks")

        def evaluate(obj, context):
            obj._evaluate_checkpoint(context)

        return target(
            self,
            managed=ManagedConfig(
                state_repo=config.state_repo,
                control_store=config.control_store,
                rerun=config.rerun,
                callbacks=[evaluate, *callbacks],
            ),
        )

    return wrapped


class Experiment(Serializable):
    """Managed training aggregate with exact checkpoint-bound evaluation.

    Args:
        model: Stateful model updated by ``train_fn``.
        train_fn: Training procedure, optionally exposing truthful safe points.
        train_data: Optional Dataset consumed by the training procedure.
        val_data: Optional Dataset used only by the training procedure.
        test_data: Exact saved Dataset reference available to Artifact recipes as
            ``this.test_data``. Missing required data fails binding; it never falls
            back to training or validation data.
         artifacts: String-keyed Template-role Definition mapping, or ``None``.
              Recipes remain inert until a checkpoint binds
             ``this`` to its exact Experiment StateRef.
         checkpoint_every_steps: Positive exact optimizer-step cadence for
             intermediate checkpoints, or ``None`` to suppress intermediate
             checkpoints. The terminal checkpoint always occurs.
        metrics: Legacy trainer-local metric configuration retained separately from
            checkpoint Artifacts.
        capabilities: Additional trainer-owned configuration.

    ``train`` publishes and associates each Experiment checkpoint, writes its
    pending history row, evaluates Artifacts in declaration order, and then marks
    the row terminal. History rows retain exact checkpoint references while their
    projected subject remains non-materializing.
    """

    def __init__(
            self, model, train_fn, train_data=None, val_data=None, *,
            test_data: Ref[StateRef | None] = None,
            artifacts: Mapping[str, Template] | None = None,
            metrics=None, checkpoint_every_steps: int | None = None, **capabilities):
        """Initialize inert training and exact-reference evaluation configuration.

        Args:
            model: Stateful model owned by this Experiment graph.
            train_fn: Training procedure used only by :meth:`train`.
            train_data: Optional Dataset for training.
            val_data: Optional Dataset for trainer-owned validation.
            test_data: Optional exact saved Dataset StateRef for Artifact binding.
            artifacts: Optional inert mapping of named Definition recipes. The
                activated signature boundary orders names canonically and quotes
                every recipe independently.
            metrics: Optional trainer-local metric configuration.
            checkpoint_every_steps: Positive exact optimizer-step cadence, or
                ``None`` for terminal-only checkpointing.
            capabilities: Additional trainer-specific configuration values.

        Side Effects:
            Creates fresh retained TrainState only. It does not materialize recipe
            inputs, compute Artifacts, save a Dataset, or select Store authority.
        """
        if checkpoint_every_steps is not None:
            if type(checkpoint_every_steps) is not int:
                raise TypeError("Experiment checkpoint_every_steps must be an exact int or None.")
            if checkpoint_every_steps <= 0:
                raise ValueError("Experiment checkpoint_every_steps must be positive.")
        super().__init__()
        self.model = model
        self.train_fn = train_fn
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.artifacts = artifacts
        self.metrics = dict(metrics or {})
        self.checkpoint_every_steps = checkpoint_every_steps
        self.capabilities = dict(capabilities)
        self.state = TrainState()

    @_experiment_train_wrapper
    @managed_operation(resumable=True, return_state_ref=True)
    def train(self, *, managed) -> None:
        """Train and evaluate the exact terminal Experiment checkpoint.

        Args:
            managed: Framework-created managed context selecting checkpoint,
                recovery, history, and Artifact authority.

        Returns:
            The managed boundary returns the exact completed terminal StateRef;
            this authored body returns ``None``.

        Raises:
            Exception: Propagates training, checkpoint, Artifact, history, and
                callback failures without reporting a completed training result.

        Side Effects:
            Publishes checkpoint-associated ExperimentData rows and independently
            recoverable Artifact states. A resumed pending observation is repaired
            before any further trainer update.
        """
        self._preflight_artifacts()
        if managed.is_resuming and self.state.pending_observation is not None:
            retained = managed.checkpoint_state_ref
            replayed = managed.checkpoint()
            if replayed != retained:
                raise RuntimeError("Experiment checkpoint replay changed its retained StateRef.")
            if self.state.pending_observation_terminal:
                return
        if self.state.is_trained:
            return
        self.state.phase = TrainState.training
        try:
            callbacks = (self._checkpoint_bridge(managed),) if getattr(
                self.train_fn, "supports_safe_points", True
            ) else ()
            self.train_fn(self, callbacks=callbacks)
        except Exception:
            self.state.phase = TrainState.failed
            raise
        if self.state.phase == TrainState.training:
            self.state.phase = TrainState.trained
        self._stage_checkpoint(managed, terminal=True)

    def _checkpoint_bridge(self, managed):
        """Return a safe-point bridge that applies the retained checkpoint cadence.

        The trainer calls this after a completed optimizer update. ``None`` leaves
        intermediate safe points inert; the terminal checkpoint remains mandatory.
        """

        def checkpoint() -> None:
            cadence = self.checkpoint_every_steps
            if cadence is not None and self.state.step % cadence == 0:
                self._stage_checkpoint(managed)

        return checkpoint

    def _stage_checkpoint(self, managed, *, terminal: bool = False) -> StateRef:
        """Persist one retry-stable observation before entering a managed safe point."""

        prior = self.state.pending_observation
        self.state.stage_observation()
        self.state.pending_observation_time = time.time_ns() // 1_000_000
        self.state.pending_observation_prev_state_ref = self.last_state_ref
        self.state.pending_observation_prev_row_key = (
            None if prior is None else self._occurrence_key(self.state.pending_observation_attempt_id, prior.sequence)
        )
        self.state.pending_observation_attempt_id = managed.attempt_id
        self.state.pending_observation_terminal = terminal
        return managed.checkpoint()

    @staticmethod
    def _occurrence_key(attempt_id: str | None, sequence: int) -> str:
        """Return the durable key for one managed safe-point occurrence."""

        if not attempt_id:
            raise RuntimeError("Experiment observation has no managed attempt identity.")
        return f"v1:{attempt_id}:{sequence}"

    def _evaluate_checkpoint(self, context) -> None:
        """Publish and reconcile one exact checkpoint's Artifact history row."""

        checkpoint = context.checkpoint_state_ref
        observation = self.state.pending_observation
        if checkpoint is None or observation is None:
            raise RuntimeError("Experiment evaluation requires an associated pending observation.")
        row_key = self._occurrence_key(self.state.pending_observation_attempt_id, observation.sequence)
        names, recipes = self._artifact_recipes()
        history = ExperimentData.get_or_create(checkpoint.object_projection(), repo=context.state_repo)
        try:
            parameters = model_parameter_counts(self.model, repo=context.state_repo)
        except (MeasurementUnavailableError, TypeError):
            parameters = None
        history.add_row(
            expected_artifacts=names,
            state_ref=checkpoint,
            prev_state_ref=self.state.pending_observation_prev_state_ref,
            time=self.state.pending_observation_time,
            examples_seen=observation.examples_seen,
            parameters=parameters,
            dataset_size=dataset_size(self.train_data) if self.train_data is not None else dataset_size(_UnknownDataset()),
            training_loss=observation.training_loss,
            row_key=row_key,
            prev_row_key=self.state.pending_observation_prev_row_key,
        )
        history.publish(repo=context.state_repo)
        _experiment_boundary("pending_row_published")
        row = history._rows[row_key]
        try:
            for name, recipe in recipes.items():
                if name in dict(row["eval_artifacts"]):
                    continue
                result_ref, scalars = self._evaluate_artifact(
                    name, recipe, checkpoint, history, row_key, context,
                )
                history.update_row(
                    row_key,
                    eval_artifacts={**dict(history._rows[row_key]["eval_artifacts"]), name: result_ref},
                    scalar_values=scalars,
                    evaluation_status="pending",
                )
                history.publish(repo=context.state_repo)
                _experiment_boundary("result_row_published")
                row = history._rows[row_key]
        except BaseException as error:
            results = dict(history._rows[row_key]["eval_artifacts"])
            completed = tuple(results) == names
            try:
                history.update_row(
                    row_key,
                    eval_artifacts=results,
                    scalar_values=dict(history._rows[row_key]["scalars"]),
                    evaluation_status="completed" if completed else "failed",
                    failed_artifact=None if completed else name,
                )
                history.publish(repo=context.state_repo)
                _experiment_boundary(
                    "completed_status_published" if completed else "failed_status_published"
                )
            except BaseException as status_error:
                raise status_error from error
            raise
        history.update_row(
            row_key,
            eval_artifacts=dict(history._rows[row_key]["eval_artifacts"]),
            scalar_values=dict(history._rows[row_key]["scalars"]),
            evaluation_status="completed",
        )
        history.publish(repo=context.state_repo)
        _experiment_boundary("completed_status_published")

    def _artifact_recipes(self) -> tuple[tuple[str, ...], OrderedDict[str, object]]:
        """Return the configured ordered inert Artifact recipes without resolving them."""

        if self.artifacts is None:
            return (), OrderedDict()
        recipes = OrderedDict(self.artifacts)
        return tuple(recipes), recipes

    def _preflight_artifacts(self) -> None:
        """Validate inert Artifact recipes before training changes retained state.

        Every active recipe root must be the Experiment-owned ``this`` root. A
        synthetic exact checkpoint proves path binding and CDef role compatibility
        without saving, loading, or materializing any Experiment payload.

        Raises:
            TypeError: If a recipe cannot bind to an Experiment checkpoint, resolve
                to a concrete Artifact definition, or expose the required managed
                compute contract.

        Side Effects:
            Performs definition/signature validation only. It does not construct
            an Artifact, save a checkpoint, invoke a trainer, or materialize model
            or Dataset payloads.
        """

        _, recipes = self._artifact_recipes()
        if not recipes:
            return
        proof = StateRef(
            self.object_ref,
            {path: "dryml-" + "0" * 64 for path in self.object_ref.objects},
        )
        for name, recipe in recipes.items():
            roots = set(recipe.names)
            unknown = roots.difference({"this"})
            if unknown:
                raise TypeError(
                    f"Artifact recipe {name!r} has unsupported active roots: "
                    f"{sorted(unknown)!r}."
                )
            try:
                bound = recipe.sub(this=proof) if "this" in roots else recipe
                definition = bound.concretize()
            except Exception as error:
                raise TypeError(
                    f"Artifact recipe {name!r} cannot bind to an Experiment "
                    "checkpoint definition."
                ) from error
            artifact_cls = definition.cls
            resolve_class = getattr(artifact_cls, "resolve", None)
            if not isinstance(artifact_cls, type) and callable(resolve_class):
                artifact_cls = resolve_class()
            if not isinstance(artifact_cls, type) or not issubclass(artifact_cls, Artifact):
                raise TypeError(f"Artifact recipe {name!r} must resolve to an Artifact definition.")
            if inspect.isabstract(artifact_cls):
                raise TypeError(f"Artifact recipe {name!r} resolves to an abstract Artifact class.")
            self._validate_artifact_compute(name, artifact_cls)

    @staticmethod
    def _validate_artifact_compute(name: str, artifact_cls: type[Artifact]) -> None:
        """Require the narrow resumable managed compute contract for one recipe.

        Args:
            name: Configured Artifact name used in validation errors.
            artifact_cls: Concrete Artifact class supplied by a resolved recipe.

        Raises:
            TypeError: If ``compute`` is not a resumable managed operation returning
                its final StateRef, or requires ordinary caller arguments.

        Side Effects:
            Inspects the class declaration only; no receiver is constructed.
        """

        compute = inspect.getattr_static(artifact_cls, "compute", None)
        descriptor = getattr(compute, "_descriptor", compute)
        if not isinstance(descriptor, ManagedOperation):
            raise TypeError(f"Artifact recipe {name!r} requires a managed compute operation.")
        if not descriptor.resumable or not descriptor.return_state_ref:
            raise TypeError(
                f"Artifact recipe {name!r} compute must be resumable and return its final StateRef."
            )
        ordinary = (
            parameter for parameter in descriptor.author_signature.parameters.values()
            if parameter.name not in {
                descriptor.instance_parameter, "managed", descriptor.store_parameter,
            }
        )
        required = [
            parameter.name for parameter in ordinary
            if parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
            } and parameter.default is inspect.Parameter.empty
        ]
        if required:
            raise TypeError(
                f"Artifact recipe {name!r} compute requires ordinary arguments: {required!r}."
            )

    def _evaluate_artifact(self, name, recipe, checkpoint, history, row_key, context):
        """Complete one retained Artifact receiver and validate its exact result.

        Readiness is a payload property, not managed completion authority.  The
        retained initial StateRef identifies the only receiver permitted to
        complete this occurrence; its managed receipt names the only result that
        can enter history.
        """

        bound = recipe.sub(this=checkpoint) if "this" in recipe.names else recipe
        definition = bound.concretize(repo=context.state_repo)
        if self.state.pending_observation_terminal:
            completed = Artifact.load_completed(
                definition,
                repo=context.state_repo,
                control_store=context.control_store,
                reuse_live="never",
            )
            if completed is not None:
                return completed.last_state_ref, self._artifact_scalars(name, completed)
        row = history._rows[row_key]
        initial_refs = dict(row["artifact_inputs"])
        if name in initial_refs:
            initial = initial_refs[name]
        else:
            artifact = context.state_repo.load_or_build(definition, cache="none")
            if not isinstance(artifact, Artifact):
                raise TypeError("Experiment Artifact recipes must resolve to Artifact instances.")
            initial = context.state_repo.save_object(artifact)
            history.update_row(
                row_key,
                artifact_inputs={**initial_refs, name: initial},
                eval_artifacts=dict(row["eval_artifacts"]),
                scalar_values=dict(row["scalars"]),
                evaluation_status="pending",
            )
            history.publish(repo=context.state_repo)
            _experiment_boundary("artifact_input_published")
        artifact = Artifact.recover(
            initial, repo=context.state_repo,
            control_store=context.control_store, reuse_live="never",
        )
        controls = {
            "state_repo": context.state_repo,
            "control_store": context.control_store,
        }
        status = artifact.compute.status(**controls)
        if status.state == "completed":
            result_ref = status.final_state_ref
        else:
            try:
                result_ref = artifact.compute(managed=ManagedConfig(**controls))
            except ManagedRerunRequiredError as error:
                # A retained initial receiver whose failed compute had no safe
                # checkpoint needs its explicitly fenced fresh Artifact attempt.
                if error.reason == "rerun_required":
                    result_ref = artifact.compute(managed=ManagedConfig(rerun=True, **controls))
                elif error.reason == "already_completed":
                    raced_status = artifact.compute.status(**controls)
                    result_ref = raced_status.final_state_ref
                else:
                    raise
            status = artifact.compute.status(**controls)
        if (
                not isinstance(result_ref, StateRef)
                or status.state != "completed"
                or status.final_state_ref is None
                or result_ref != status.final_state_ref):
            raise RuntimeError(
                "Artifact compute did not return its managed completed StateRef."
            )
        artifact = Artifact.recover(
            initial, repo=context.state_repo,
            control_store=context.control_store, reuse_live="never",
        )
        if artifact.last_state_ref != result_ref:
            raise RuntimeError(
                "Artifact recovery did not restore its managed completed StateRef."
            )
        _experiment_boundary("artifact_completed")
        return result_ref, self._artifact_scalars(name, artifact)

    @staticmethod
    def _artifact_scalars(name: str, artifact: Artifact) -> Mapping[str, object]:
        """Extract supported public scalar Value results for one history update."""

        if not isinstance(artifact, Value):
            return {}
        value = artifact.value()
        if isinstance(value, Mapping):
            scalars = {}
            for field, raw_value in value.items():
                scalar = Experiment._history_scalar(raw_value)
                if isinstance(field, str) and scalar is not _UNSUPPORTED:
                    scalars[ExperimentData.scalar_column(name, field)] = scalar
            return scalars
        scalar = Experiment._history_scalar(value)
        return {} if scalar is _UNSUPPORTED else {ExperimentData.scalar_column(name): scalar}

    @staticmethod
    def _history_scalar(value):
        """Return one public finite scalar or the private unsupported marker."""

        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        dimension = Experiment._native_scalar_dimension(value)
        if dimension != 0:
            return _UNSUPPORTED
        item = getattr(value, "item", None)
        if callable(item):
            try:
                normalized = item()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                normalized = _UNSUPPORTED
            if normalized is not _UNSUPPORTED and normalized is not value:
                return Experiment._history_scalar(normalized)
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            try:
                normalized = numpy()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                normalized = _UNSUPPORTED
            if normalized is not _UNSUPPORTED and normalized is not value:
                return Experiment._history_scalar(normalized)
        return _UNSUPPORTED

    @staticmethod
    def _native_scalar_dimension(value: object) -> int | None:
        """Return a bounded native scalar rank without importing optional backends."""

        ndim = getattr(value, "ndim", None)
        if type(ndim) is int:
            return ndim
        shape = getattr(value, "shape", None)
        rank = getattr(shape, "rank", None)
        if type(rank) is int:
            return rank
        if shape is not None:
            try:
                return len(shape)
            except TypeError:
                pass
        return None

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        pickle_save(self.state, os.path.join(dest_dir, "experiment_state.pkl"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        self.state = pickle_load(os.path.join(src_dir, "experiment_state.pkl"))


class _UnknownDataset:
    """Minimal cardinality carrier used when an Experiment has no training data."""

    def __len__(self):
        raise NotImplementedError


_UNSUPPORTED = object()


def _experiment_boundary(stage: str) -> None:
    """Provide a narrow in-process seam for Experiment recovery-window tests.

    Args:
        stage: One completed Experiment-owned durable boundary.

    Side Effects:
        None in production. Tests may replace this function to raise immediately
        after a named publication and prove replay through ordinary checkpoints.
    """


__all__ = ["Experiment"]
