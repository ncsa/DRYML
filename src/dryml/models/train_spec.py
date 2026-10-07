from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from math import isfinite
from typing import ClassVar

from dryml.core.reference_values import StateRef

from .measurements import TrainingObservation


@dataclass(slots=True)
class _EarlyStoppingState:
    """TrainFunction-owned continuation for one early-stopping invocation."""

    best_metric: float | None = None
    wait: int = 0
    best_epoch: int | None = None
    best_model_state: object | None = None
    last_epoch: int | None = None
    stopped_epoch: int | None = None
    accepted_target: int | None = None
    restored: bool = False

    def reset(self) -> None:
        """Clear prior invocation facts before a fresh invocation starts."""

        self.best_metric = None
        self.wait = 0
        self.best_epoch = None
        self.best_model_state = None
        self.last_epoch = None
        self.stopped_epoch = None
        self.accepted_target = None
        self.restored = False

    def to_payload(self) -> dict[str, object]:
        """Return a backend-serializable continuation mapping."""

        return {
            "version": 1,
            "best_metric": self.best_metric,
            "wait": self.wait,
            "best_epoch": self.best_epoch,
            "best_model_state": self.best_model_state,
            "last_epoch": self.last_epoch,
            "stopped_epoch": self.stopped_epoch,
            "accepted_target": self.accepted_target,
            "restored": self.restored,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "_EarlyStoppingState":
        """Validate and rebuild one saved continuation mapping."""

        names = {
            "version", "best_metric", "wait", "best_epoch", "best_model_state",
            "last_epoch", "stopped_epoch", "accepted_target", "restored",
        }
        if type(payload) is not dict or set(payload) != names or payload["version"] != 1:
            raise ValueError("Unsupported early-stopping continuation state.")
        best_metric = payload["best_metric"]
        if best_metric is not None and (
            type(best_metric) not in (int, float) or not isfinite(best_metric)
        ):
            raise ValueError("Saved early-stopping best metric must be finite or None.")
        for name in ("wait", "best_epoch", "last_epoch", "stopped_epoch", "accepted_target"):
            value = payload[name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Saved early-stopping {name} must be nonnegative or None.")
        if type(payload["wait"]) is not int:
            raise TypeError("Saved early-stopping wait must be an exact integer.")
        if type(payload["restored"]) is not bool:
            raise TypeError("Saved early-stopping restored marker must be an exact bool.")
        if (best_metric is None) != (payload["best_epoch"] is None):
            raise ValueError("Saved early-stopping best metric and epoch must agree.")
        if payload["stopped_epoch"] is None:
            if payload["accepted_target"] is not None or payload["restored"]:
                raise ValueError("Saved early-stopping terminal facts require a stop epoch.")
        elif payload["accepted_target"] != payload["stopped_epoch"] + 1:
            raise ValueError("Saved early-stopping accepted target is inconsistent.")
        return cls(
            best_metric=None if best_metric is None else float(best_metric),
            wait=payload["wait"],
            best_epoch=payload["best_epoch"],
            best_model_state=payload["best_model_state"],
            last_epoch=payload["last_epoch"],
            stopped_epoch=payload["stopped_epoch"],
            accepted_target=payload["accepted_target"],
            restored=payload["restored"],
        )


def _validate_early_stopping_config(
    *, monitor: str, patience: int, mode: str, min_delta: float,
    restore_best_weights: bool,
) -> tuple[str, int, str, float, bool]:
    """Validate and normalize the shared saved early-stopping configuration."""

    if type(monitor) is not str or not monitor:
        raise ValueError("early-stopping monitor must be a nonempty string.")
    if monitor != "loss" and not (monitor.startswith("val_") and len(monitor) > 4):
        raise ValueError(
            "early-stopping monitor must be 'loss' or a validation metric prefixed by 'val_'."
        )
    if type(patience) is not int:
        raise TypeError("early-stopping patience must be an exact integer.")
    if patience < 0:
        raise ValueError("early-stopping patience must be nonnegative.")
    if mode not in ("min", "max"):
        raise ValueError("early-stopping mode must be 'min' or 'max'.")
    if type(min_delta) not in (int, float):
        raise TypeError("early-stopping min_delta must be a finite number.")
    min_delta = float(min_delta)
    if not isfinite(min_delta) or min_delta < 0:
        raise ValueError("early-stopping min_delta must be finite and nonnegative.")
    if type(restore_best_weights) is not bool:
        raise TypeError("restore_best_weights must be an exact bool.")
    return monitor, patience, mode, min_delta, restore_best_weights


def _update_early_stopping(
    state: _EarlyStoppingState,
    *,
    epoch: int,
    metrics: Mapping[str, float],
    monitor: str,
    patience: int,
    mode: str,
    min_delta: float,
    capture_best,
    restore_best,
    restore_best_weights: bool,
) -> bool:
    """Apply one idempotent completed-epoch early-stopping decision."""

    if state.last_epoch == epoch:
        stopped = state.stopped_epoch == epoch
    else:
        if monitor not in metrics:
            raise ValueError(f"Early-stopping monitor {monitor!r} is missing from completed epoch metrics.")
        metric = metrics[monitor]
        if type(metric) not in (int, float) or not isfinite(metric):
            raise ValueError(f"Early-stopping monitor {monitor!r} must be finite.")
        metric = float(metric)
        improved = state.best_metric is None or (
            metric < state.best_metric - min_delta
            if mode == "min"
            else metric > state.best_metric + min_delta
        )
        if improved:
            best_model_state = capture_best() if restore_best_weights else None
            state.best_metric = metric
            state.wait = 0
            state.best_epoch = epoch
            state.best_model_state = best_model_state
        else:
            state.wait += 1
        state.last_epoch = epoch
        stopped = not improved and state.wait > patience
        if stopped:
            state.stopped_epoch = epoch
            state.accepted_target = epoch + 1

    if stopped and restore_best_weights and not state.restored:
        if state.best_model_state is None:
            raise ValueError("Early stopping has no completed best Model state to restore.")
        restore_best(state.best_model_state)
        state.restored = True
    return stopped


@dataclass(slots=True)
class TrainState:
    """Retained optimizer-progress accounting for one Experiment training state.

    ``examples_seen``, the observation loss window, and the separate weighted
    completed-epoch loss accumulator advance only after successful optimizer
    updates. ``next_batch`` identifies the next unprocessed batch of ``epoch``;
    this makes a restored deterministic stream resume without reapplying prior
    updates. ``target_epoch`` distinguishes an interrupted invocation from a
    fresh later call. ``pending_epoch_postlude`` and its phase retain the bounded
    validation/progress/native-callback work remaining after a normalized final
    update. ``pending_epoch_metrics`` preserves completed validation facts for a
    retrying progress boundary. U9 owns publication of ``pending_observation``
    to ExperimentData. Its accompanying time, predecessor, attempt, and terminal
    fields are captured before the managed checkpoint, so retry reuses one history
    occurrence rather than assigning fresh facts after publication.
    """

    initial: ClassVar[str | None] = None
    training: ClassVar[str] = "training"
    trained: ClassVar[str] = "trained"
    failed: ClassVar[str] = "failed"

    epoch: int = 0
    step: int = 0
    phase: str | None = initial
    examples_seen: int = 0
    loss_numerator: float = 0.0
    loss_denominator: int = 0
    epoch_loss_numerator: float = 0.0
    epoch_loss_denominator: int = 0
    next_batch: int = 0
    target_epoch: int | None = None
    pending_epoch_postlude: int | None = None
    pending_epoch_postlude_phase: str | None = None
    pending_epoch_metrics: dict[str, float] | None = None
    safe_point_sequence: int = 0
    pending_observation: TrainingObservation | None = None
    pending_observation_time: int | None = None
    pending_observation_prev_state_ref: StateRef | None = None
    pending_observation_prev_row_key: str | None = None
    pending_observation_attempt_id: str | None = None
    pending_observation_terminal: bool = False

    @property
    def is_initial(self) -> bool:
        return self.phase == self.initial

    @property
    def is_training(self) -> bool:
        return self.phase == self.training

    @property
    def is_trained(self) -> bool:
        return self.phase == self.trained

    @property
    def is_failed(self) -> bool:
        return self.phase == self.failed

    def __eq__(self, other):
        if isinstance(other, TrainState):
            return (
                self.epoch == other.epoch
                and self.step == other.step
                and self.phase == other.phase
                and self.examples_seen == other.examples_seen
                and self.loss_numerator == other.loss_numerator
                and self.loss_denominator == other.loss_denominator
                and self.epoch_loss_numerator == other.epoch_loss_numerator
                and self.epoch_loss_denominator == other.epoch_loss_denominator
                and self.next_batch == other.next_batch
                and self.target_epoch == other.target_epoch
                and self.pending_epoch_postlude == other.pending_epoch_postlude
                and self.pending_epoch_postlude_phase == other.pending_epoch_postlude_phase
                and self.pending_epoch_metrics == other.pending_epoch_metrics
                and self.safe_point_sequence == other.safe_point_sequence
                and self.pending_observation == other.pending_observation
                and self.pending_observation_time == other.pending_observation_time
                and self.pending_observation_prev_state_ref == other.pending_observation_prev_state_ref
                and self.pending_observation_prev_row_key == other.pending_observation_prev_row_key
                and self.pending_observation_attempt_id == other.pending_observation_attempt_id
                and self.pending_observation_terminal == other.pending_observation_terminal
            )
        if isinstance(other, str) or other is None:
            return self.phase == other
        return NotImplemented

    def __getstate__(self) -> dict[str, object]:
        """Encode all current fields by name for version-tolerant persistence.

        Returns:
            A field-name mapping accepted by :meth:`__setstate__`.

        Side Effects:
            None. Named state lets a newer slotted schema restore older pickles
            that did not contain postlude or observation fields.
        """

        return {item.name: getattr(self, item.name) for item in fields(self)}

    def __setstate__(self, state: object) -> None:
        """Restore current or legacy slotted TrainState pickle state.

        Args:
            state: Named current/legacy state or Python's legacy slotted-state
                tuple.

        Raises:
            TypeError: If the pickle payload is not a supported state shape.

        Side Effects:
            Initializes every missing newer field from its declared default so
            historical checkpoints remain recoverable.
        """

        if isinstance(state, tuple) and len(state) == 2:
            if state[0] is not None:
                raise TypeError("Unsupported TrainState instance pickle state.")
            state = state[1]
        if isinstance(state, (tuple, list)):
            if len(state) != 3:
                raise TypeError("Unsupported TrainState legacy slot state.")
            state = dict(zip(("epoch", "step", "phase"), state))
        if not isinstance(state, Mapping):
            raise TypeError("Unsupported TrainState pickle state.")
        field_map = {item.name: item for item in fields(self)}
        unknown = set(state).difference(field_map)
        if unknown:
            raise ValueError(f"Unknown TrainState pickle fields: {sorted(unknown)!r}.")
        values = {}
        for name, item in field_map.items():
            if name in state:
                values[name] = state[name]
            elif item.default is not MISSING:
                values[name] = item.default
            else:
                values[name] = item.default_factory()
        self._validate_restored_state(values)
        for item in fields(self):
            value = values[item.name]
            object.__setattr__(self, item.name, value)

    @staticmethod
    def _validate_restored_state(values: Mapping[str, object]) -> None:
        """Validate a complete decoded state before installing any field.

        Args:
            values: Complete current-schema field mapping with legacy defaults.

        Raises:
            TypeError: If a field has an incompatible exact type.
            ValueError: If counters, enum values, or retained lifecycle facts are
                inconsistent.

        Side Effects:
            None. Validation is deliberately complete before ``__setstate__``
            changes this instance, so malformed persistence cannot reach a
            trainer with partially installed progress.
        """

        def counter(name: str, *, nullable: bool = False):
            value = values[name]
            if nullable and value is None:
                return None
            if type(value) is not int:
                raise TypeError(f"TrainState {name} must be an exact integer.")
            if value < 0:
                raise ValueError(f"TrainState {name} must be nonnegative.")
            return value

        epoch = counter("epoch")
        counter("step")
        counter("examples_seen")
        counter("loss_denominator")
        counter("epoch_loss_denominator")
        next_batch = counter("next_batch")
        counter("safe_point_sequence")
        target_epoch = counter("target_epoch", nullable=True)
        postlude = counter("pending_epoch_postlude", nullable=True)
        numerator = values["loss_numerator"]
        if type(numerator) not in (int, float) or not isfinite(numerator):
            raise ValueError("TrainState loss_numerator must be finite.")
        epoch_numerator = values["epoch_loss_numerator"]
        if type(epoch_numerator) not in (int, float) or not isfinite(epoch_numerator):
            raise ValueError("TrainState epoch_loss_numerator must be finite.")
        phase = values["phase"]
        if phase is not None and (
            type(phase) is not str
            or phase not in (TrainState.training, TrainState.trained, TrainState.failed)
        ):
            raise ValueError("TrainState phase is invalid.")
        postlude_phase = values["pending_epoch_postlude_phase"]
        valid_postlude_phases = {"start", "validation", "epoch_end", "progress"}
        if postlude_phase is not None and (
            type(postlude_phase) is not str or postlude_phase not in valid_postlude_phases
        ):
            raise ValueError("TrainState pending_epoch_postlude_phase is invalid.")
        metrics = values["pending_epoch_metrics"]
        if metrics is not None:
            if type(metrics) is not dict:
                raise TypeError("TrainState pending_epoch_metrics must be a dictionary or None.")
            if not all(type(name) is str and type(value) in (int, float) and isfinite(value)
                       for name, value in metrics.items()):
                raise ValueError("TrainState pending_epoch_metrics must contain finite string-keyed values.")
        if target_epoch is not None and target_epoch < epoch:
            raise ValueError("TrainState target_epoch cannot precede epoch.")
        if postlude is None:
            if postlude_phase is not None or metrics is not None:
                raise ValueError("TrainState postlude details require a pending epoch postlude.")
        elif epoch != postlude + 1 or next_batch != 0 or postlude_phase is None:
            raise ValueError("TrainState pending epoch postlude is inconsistent with epoch progress.")
        observation = values["pending_observation"]
        if observation is not None and type(observation) is not TrainingObservation:
            raise TypeError("TrainState pending_observation must be a TrainingObservation or None.")
        observation_time = counter("pending_observation_time", nullable=True)
        predecessor = values["pending_observation_prev_state_ref"]
        if predecessor is not None and type(predecessor) is not StateRef:
            raise TypeError("TrainState pending_observation_prev_state_ref must be a StateRef or None.")
        predecessor_key = values["pending_observation_prev_row_key"]
        if predecessor_key is not None and (type(predecessor_key) is not str or not predecessor_key):
            raise TypeError("TrainState pending_observation_prev_row_key must be a nonempty string or None.")
        attempt_id = values["pending_observation_attempt_id"]
        if attempt_id is not None and (type(attempt_id) is not str or not attempt_id):
            raise TypeError("TrainState pending_observation_attempt_id must be a nonempty string or None.")
        if type(values["pending_observation_terminal"]) is not bool:
            raise TypeError("TrainState pending_observation_terminal must be an exact bool.")
        if observation is None:
            if any(value is not None for value in (observation_time, predecessor, predecessor_key, attempt_id)):
                raise ValueError("TrainState pending observation details require an observation.")
            if values["pending_observation_terminal"]:
                raise ValueError("TrainState terminal observation marker requires an observation.")

    def advance_epoch(self, n: int = 1):
        self.epoch += n

    def advance_step(self, n: int = 1):
        self.step += n

    def record_update(self, *, examples: int, loss: float) -> None:
        """Retain one completed optimizer update and its mean loss contribution.

        Args:
            examples: Exact positive number of examples that contributed to the
                completed update.
            loss: Finite mean loss for those examples.

        Raises:
            TypeError: If ``examples`` is not an exact integer.
            ValueError: If the update size is nonpositive or loss is nonfinite.

        Side Effects:
            Advances step/exposure, the next-batch position, and the weighted
            loss window. Failed/evaluation work must not call this method.
        """

        from math import isfinite

        if type(examples) is not int:
            raise TypeError("examples must be an exact integer.")
        if examples <= 0:
            raise ValueError("examples must be positive.")
        if not isfinite(loss):
            raise ValueError("loss must be finite.")
        self.step += 1
        self.examples_seen += examples
        self.loss_numerator += loss * examples
        self.loss_denominator += examples
        self.epoch_loss_numerator += loss * examples
        self.epoch_loss_denominator += examples
        self.next_batch += 1

    def record_fit(self, *, examples: int) -> None:
        """Retain one successful one-shot fit without claiming update telemetry.

        Args:
            examples: Exact positive number of submitted training examples.

        Raises:
            TypeError: If ``examples`` is not an exact integer.
            ValueError: If ``examples`` is nonpositive.

        Side Effects:
            Advances the coarse successful-fit transition and exposure count only.
            It deliberately leaves loss-window and safe-point facts unchanged
            because a one-shot backend exposes no truthful optimizer boundary.
        """

        if type(examples) is not int:
            raise TypeError("examples must be an exact integer.")
        if examples <= 0:
            raise ValueError("examples must be positive.")
        self.step += 1
        self.examples_seen += examples
        self.next_batch += 1

    def begin_invocation(self, epochs: int) -> int:
        """Return the retained target epoch for this training invocation.

        Args:
            epochs: Nonnegative number of epochs requested by a fresh call.

        Returns:
            The exclusive logical epoch target.  A restored unfinished call
            retains its original target instead of adding ``epochs`` again.

        Raises:
            ValueError: If ``epochs`` is invalid or retained progress is beyond
                the saved invocation target.

        Side Effects:
            Stores a target for a new invocation.  The target remains after an
            interruption so a later restore can finish the same invocation.
        """

        if type(epochs) is not int or epochs < 0:
            raise ValueError("epochs must be a nonnegative exact integer.")
        if self.target_epoch is None:
            self.target_epoch = self.epoch + epochs
        elif self.target_epoch < self.epoch:
            raise ValueError("Saved training progress exceeds its invocation target.")
        return self.target_epoch

    def abandon_new_invocation(self, target_epoch: int, *, initial_epoch: int, initial_step: int) -> None:
        """Clear a target whose fresh invocation retained no recoverable work.

        Args:
            target_epoch: Newly allocated invocation target.
            initial_epoch: Progress epoch before allocating the target.
            initial_step: Optimizer step before allocating the target.

        Raises:
            ValueError: If the target is not the current fresh target or training
                retained progress/postlude work after it was allocated.

        Side Effects:
            Clears only a target created by setup that failed before an optimizer
            update. Existing restored targets remain recoverable.
        """

        if (
            self.target_epoch != target_epoch
            or self.epoch != initial_epoch
            or self.step != initial_step
            or self.pending_epoch_postlude is not None
        ):
            raise ValueError("Cannot abandon an invocation with retained training work.")
        self.target_epoch = None

    def finish_invocation(self, target_epoch: int, *, accept_shortened: bool = False) -> None:
        """Clear a completed invocation target without changing progress.

        Args:
            target_epoch: The target previously returned by
                :meth:`begin_invocation`.

        Raises:
            ValueError: If the invocation has not reached its retained target, or
                an epoch postlude remains unfinished.

        Side Effects:
            Clears ``target_epoch`` so a later caller starts a fresh epoch
            request rather than resuming this completed one.
        """

        if (
            self.target_epoch != target_epoch
            or self.pending_epoch_postlude is not None
            or (self.epoch < target_epoch and not accept_shortened)
        ):
            raise ValueError("Cannot finish an incomplete training invocation.")
        self.target_epoch = None

    def stage_observation(self) -> TrainingObservation:
        """Freeze current accounting and reset the next loss window before capture.

        Returns:
            The immutable observation retained in ``pending_observation``.

        Side Effects:
            Increments safe-point sequence and clears only the next reporting
            loss window. Exposure and optimizer progress remain unchanged.
        """

        self.safe_point_sequence += 1
        observation = TrainingObservation(
            examples_seen=self.examples_seen,
            loss_numerator=self.loss_numerator,
            loss_denominator=self.loss_denominator,
            epoch=self.epoch,
            next_batch=self.next_batch,
            sequence=self.safe_point_sequence,
        )
        self.pending_observation = observation
        self.loss_numerator = 0.0
        self.loss_denominator = 0
        return observation

    def finish_epoch(
        self,
        *,
        postlude_pending: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Normalize an exhausted epoch to its next batch-zero position.

        Args:
            postlude_pending: Whether validation/progress/behavior work remains.
            metrics: Optional finite completed-epoch metrics retained atomically
                with the normalized epoch for postlude recovery.

        Raises:
            ValueError: If metrics are not finite and string-keyed, or metrics
                are supplied without a retained postlude.

        Side Effects:
            Advances ``epoch`` and sets ``next_batch`` to zero. When
            ``postlude_pending`` is true, retains the completed epoch until its
            validation/progress/native-callback postlude succeeds. It does not
            change exposure, optimizer-step, or loss-window accounting.
        """

        if metrics is not None:
            if not postlude_pending:
                raise ValueError("Completed epoch metrics require a pending postlude.")
            if type(metrics) is not dict or not all(
                type(name) is str and type(value) in (int, float) and isfinite(value)
                for name, value in metrics.items()
            ):
                raise ValueError(
                    "TrainState pending_epoch_metrics must contain finite string-keyed values."
                )
            metrics = {name: float(value) for name, value in metrics.items()}
        if postlude_pending and self.epoch_loss_denominator:
            metrics = dict(metrics or {})
            metrics["loss"] = self.epoch_loss_numerator / self.epoch_loss_denominator
        completed_epoch = self.epoch
        self.epoch += 1
        self.next_batch = 0
        self.epoch_loss_numerator = 0.0
        self.epoch_loss_denominator = 0
        if postlude_pending:
            self.pending_epoch_postlude = completed_epoch
            self.pending_epoch_postlude_phase = "start"
            self.pending_epoch_metrics = metrics

    def advance_epoch_postlude(
        self, epoch: int, phase: str, *, metrics: dict[str, float] | None = None
    ) -> None:
        """Record one completed bounded epoch-postlude boundary.

        Args:
            epoch: Logical completed epoch awaiting postlude completion.
            phase: Next retained phase after the boundary, such as
                ``"validation"``, ``"epoch_end"``, or ``"progress"``.
            metrics: Completed validation or epoch metrics retained for a later
                progress retry. ``None`` leaves an existing retained value.

        Raises:
            ValueError: If no matching epoch postlude is pending.

        Side Effects:
            Replaces only the bounded retained phase. It never changes optimizer
            progress or the normalized epoch/batch position.
        """

        if self.pending_epoch_postlude != epoch:
            raise ValueError("No matching epoch postlude is pending.")
        self.pending_epoch_postlude_phase = phase
        if metrics is not None:
            self.pending_epoch_metrics = dict(metrics)

    def finish_epoch_postlude(self, epoch: int) -> None:
        """Mark one normalized epoch's required terminal postlude complete.

        Args:
            epoch: Logical completed epoch whose postlude has succeeded.

        Raises:
            ValueError: If no matching normalized epoch is awaiting postlude.

        Side Effects:
            Clears the retained postlude fact without changing optimizer progress
            or the already normalized next epoch/batch position.
        """

        if self.pending_epoch_postlude != epoch:
            raise ValueError("No matching epoch postlude is pending.")
        self.pending_epoch_postlude = None
        self.pending_epoch_postlude_phase = None
        self.pending_epoch_metrics = None


__all__ = ["TrainState"]
