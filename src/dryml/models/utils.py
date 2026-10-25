"""Model-owned training preparation, Dataset admission, and accounting helpers.

Private one-shot acceptance coordinates backend-owned candidates with TrainState;
it does not publish checkpoints or implement native fitting or persistence codecs.
"""

from __future__ import annotations

from dataclasses import dataclass
from dryml.context import check_context, NoContextError, \
    WrongContextError, ContextIncompatibilityError
from typing import Any

from dryml.core.tensor_spec import iter_specs
from dryml.core.utils.recurse import iter_leaves
from dryml.models.train_spec import TrainState

expected_context_errors = (NoContextError, WrongContextError, ContextIncompatibilityError)


@dataclass(frozen=True, slots=True)
class TrainingPreparation:
    """Once-planned declared x/y handoffs for one native training consumer.

    The carrier executes the selected edges retained in its owning trainer's
    Method graph. Runtime preparation never selects or plans a route.
    """

    producer_specs: tuple[object, object]
    consumer_specs: tuple[object, object]
    conversion_edges: tuple[object | None, object | None]

    @classmethod
    def from_dataset(cls, owner, dataset, backend: str) -> "TrainingPreparation":
        """Plan native x/y handoffs from one already-prepared Dataset.

        Args:
            owner: TrainFunction retaining this invocation's handoff graph.
            dataset: PreparedDataset with captured supervised output specs.
            backend: Native consumer backend for both branches.

        Returns:
            The retained producer/consumer specs and necessary direct edges.

        Raises:
            TypeError: If Methods require input discovery, the output
                is not a pair, or either branch lacks adequate declarations.
            ValueError: If ``backend`` is not a supported backend spelling.

        Side Effects:
            Records owner graph facts only after declaration admission succeeds.
            Opens no source and activates no optional adapter.
        """
        from dryml.data.native import PreparedDataset

        if not isinstance(dataset, PreparedDataset):
            raise TypeError("Native training handoffs require a PreparedDataset.")
        dataset._require_declared_selection()
        specs = dataset.spec
        if not isinstance(specs, tuple) or len(specs) != 2:
            raise TypeError("Native training requires captured (inputs, targets) specifications.")
        return cls.from_specs(owner, *specs, backend)

    @classmethod
    def from_specs(cls, owner, x_spec, y_spec, backend: str) -> "TrainingPreparation":
        """Plan supported direct handoffs from declared x/y specs once.

        Args:
            owner: TrainFunction Method that owns the inspectable graph facts.
            x_spec: Complete feature tree captured after Dataset preparation.
            y_spec: Complete target tree captured after Dataset preparation.
            backend: Native consumer backend for both values.

        Returns:
            An immutable carrier retaining producer/consumer facts and direct
            conversion edges, if either declared source differs from ``backend``.

        Raises:
            TypeError: If a source tree has no one complete supported backend or
                lacks a direct dense handoff to ``backend``.

        Side Effects:
            Selects no runtime values and imports no optional backend; it only
            records conversion planning facts on ``owner.method_graph()``.
        """

        from dryml.core.backend import Backend
        from dryml.methods.conversion import make_edge

        target = Backend(backend)

        def plan(spec):
            source_backends = {tensor_spec.backend for tensor_spec in iter_specs(spec)}
            if len(source_backends) != 1 or None in source_backends:
                raise TypeError("Training values require one complete declared backend.")
            source = next(iter(source_backends))
            return None if source is target else make_edge(spec, target)

        edges = (plan(x_spec), plan(y_spec))
        try:
            owner._retain_preparation_graph((x_spec, y_spec), edges)
        except AttributeError as error:
            raise TypeError("Training preparation requires a Method graph owner.") from error
        return cls(
            producer_specs=(x_spec, y_spec),
            consumer_specs=tuple(
                spec if edge is None else edge.consumer_spec
                for spec, edge in zip((x_spec, y_spec), edges)
            ),
            conversion_edges=edges,
        )

    def prepare(self, x, y):
        """Execute retained x/y conversion edges for one yielded batch.

        Args:
            x: Runtime feature batch matching ``producer_specs[0]``.
            y: Runtime target batch matching ``producer_specs[1]``.

        Returns:
            The prepared ``(x, y)`` pair matching ``consumer_specs``.

        Raises:
            TypeError: If a runtime value violates its retained dense handoff.
            RuntimeError: If a Torch gradient-bearing value would cross a
                framework boundary.

        Side Effects:
            Imports only adapters selected during planning and copies only values
            whose retained edge requires a backend handoff. Zero-edge inputs are
            validated through the existing Method constraint boundary without
            extraction or replacement, preserving identity and native gradients.
        """

        from dryml.methods.conversion import convert
        from dryml.methods.errors import ImplementationSelectionError
        from dryml.methods.implementation import _validate_input_specs
        from dryml.methods.signature import spec_node

        prepared = []
        for value, spec, edge in zip((x, y), self.producer_specs, self.conversion_edges):
            if edge is not None:
                prepared.append(convert(edge, value))
            else:
                try:
                    _validate_input_specs((spec_node(spec),), (value,))
                except ImplementationSelectionError as error:
                    raise TypeError("Runtime training value does not match its producer specification.") from error
                prepared.append(value)
        return tuple(prepared)


def finite_dataset_len(dataset) -> int | None:
    """Return a Dataset's finite yield count without opening a cursor."""

    cardinality = dataset.yield_cardinality()
    return cardinality.require_finite() if cardinality.is_finite else None


def training_cardinality(dataset):
    """Return declared training cardinality without acquiring a cursor.

    Args:
        dataset: Canonical Dataset whose yield cardinality is queried.

    Returns:
        A finite, unknown, or infinite ``Cardinality`` declaration.

    Side Effects:
        Calls only ``dataset.yield_cardinality()``; it does not open, scan, or
        consume a Dataset iterator.
    """

    return dataset.yield_cardinality()


def require_bounded_safe_points(dataset, callbacks) -> None:
    """Reject streams that cannot expose truthful final update safe points.

    Args:
        dataset: Prepared Dataset used by one trainer invocation.
        callbacks: Prevalidated DRYML safe-point callbacks.

    Raises:
        ValueError: If the Dataset is infinite without a finite bound, or its
            cardinality is unknown while callbacks require final-batch recovery.

    Side Effects:
        Reads only declared cardinality and never opens or consumes the Dataset.
    """

    cardinality = training_cardinality(dataset)
    if cardinality.is_infinite:
        raise ValueError("Training on an infinite dataset requires an explicit finite bound.")
    if callbacks and not cardinality.is_finite:
        raise ValueError(
            "Training callbacks require a finite deterministic batch count for final safe-point recovery."
        )


def require_supervised_dataset(dataset, *, batched: bool) -> None:
    """Validate a canonical Dataset pair for a supplied trainer.

    Args:
        dataset: Dataset that must yield exactly ``(inputs, targets)``. Each
            branch may be a nested TensorSpec tree.
        batched: Whether both branches must carry an explicit leading example
            dimension for a native update loop.

    Raises:
        ValueError: If data is absent, noncanonical, or has incompatible batch
            declarations.
        TypeError: If a Dataset specification has no TensorSpec leaves.

    Side Effects:
        Reads public Dataset specification metadata only. It does not inspect
        Dataset operators, open a cursor, or alter selection/order/batching.
    """

    from dryml.data import Dataset

    if dataset is None:
        raise ValueError("Experiment has no train_data.")
    if not isinstance(dataset, Dataset):
        raise TypeError("Training data must be a Dataset.")
    try:
        inputs, targets = dataset.spec
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Training data must be a canonical Dataset yielding (inputs, targets); "
            "use data.as_supervised(...)."
        ) from error
    if not isinstance(dataset.spec, tuple) or len(dataset.spec) != 2:
        raise ValueError(
            "Training data must be a canonical Dataset yielding (inputs, targets); "
            "use data.as_supervised(...)."
        )
    branch_batches = []
    for branch in (inputs, targets):
        specs = tuple(iter_specs(branch))
        if not specs:
            raise TypeError("Training data branches require TensorSpec leaves.")
        flags = {spec.batched for spec in specs}
        if len(flags) != 1:
            raise ValueError("Training data branches must be uniformly batched or unbatched.")
        branch_batches.append(flags.pop())
    if branch_batches[0] != branch_batches[1]:
        raise ValueError("Training inputs and targets must have matching batching declarations.")
    if batched and not branch_batches[0]:
        raise ValueError(
            "Native training requires explicitly batched (inputs, targets) data; "
            "author Batch(dataset, 1) or another Dataset batch before training."
        )


def advance_train_state(exp, *, epochs: int = 0, steps: int = 0, phase: str = TrainState.trained):
    if epochs:
        exp.state.advance_epoch(epochs)
    if steps:
        exp.state.advance_step(steps)
    exp.state.phase = phase


def validate_training_callbacks(callbacks) -> tuple:
    """Freeze and validate every DRYML safe-point callback before training work.

    Args:
        callbacks: Iterable of zero-argument process-local callbacks.

    Returns:
        An immutable callback sequence in caller order.

    Raises:
        TypeError: If the value is not iterable or any member is not callable.

    Side Effects:
        None.  In particular, this check neither opens data nor changes model,
        optimizer, cursor, mode, or retained training state.
    """

    try:
        frozen = tuple(callbacks)
    except TypeError as error:
        raise TypeError("training callbacks must be an iterable of callables.") from error
    if not all(callable(callback) for callback in frozen):
        raise TypeError("training callbacks must be callable.")
    return frozen


def _fit_one_shot(exp, *, fit_candidate, encode_candidate, callbacks=(), observer_session=None):
    """Collect, prepare, and locally accept one opaque fresh-candidate fit.

    Args:
        exp: Experiment with a Model whose ordinary ``_accepted_fit`` instance
            attribute holds all accepted native state, fitted facts, and encoded
            payload bytes (initially None), and an exact TrainState in ``state``.
        fit_candidate: Callable ``(x, y) -> candidate``. Owns fresh native
            construction, fitting, and complete fitted-fact/convergence validation.
            It must not mutate the previously accepted bundle or TrainState.
        encode_candidate: Callable ``(candidate) -> accepted_bundle``. Encodes
            final native bytes and canonical metadata exactly once, returning a
            non-None backend-owned bundle ready for installation. It must not
            install state or mutate the prior bundle. Runs after prep_eval.
        callbacks: Intermediate safe-point requests; any nonempty validated
            collection is unsupported and rejected before source acquisition.
        observer_session: Intermediate observer request; any non-None value is
            unsupported and rejected before source acquisition.

    Returns:
        The exact bundle installed in ``exp.model._accepted_fit``. Backends may
        extract their native return from this bundle for direct trainer callers.

    Raises:
        TypeError: For invalid callbacks, owner slots, or TrainState fields.
        NotImplementedError: For intermediate callbacks or observers.
        ValueError: For invalid accounting, unfinished iterative progress, strict
            Dataset admission/collection, or an absent encoded bundle.
        BaseException: Propagates collection, preparation, native validation,
            encoding, or local installation failure, including interruption.
            Preparation cleanup never replaces an escaping primary failure.

    Side Effects:
        Materializes the canonical Dataset using strict NumPy admission and exact
        collected N. Before native work, validates detached proposed accounting
        using record_fit(examples=N) and finish_epoch(), without loss telemetry or
        an epoch postlude. Calls model prep_train/prep_eval before acceptance.
        Only after cleanup and final encoding succeed, replaces the plain Model
        bundle and Experiment TrainState slots, rolling both back on BaseException.
        These local assignments bypass owner hooks; no I/O, native call, callback,
        or cleanup occurs between them. Preparation hooks must leave accepted
        native/fitted facts intact. Same-model concurrent fit/restore and external
        mutation of the borrowed native state are unsupported. Experiment owns
        phase, terminal association, and publication; later publication failure
        must retain this accepted fit, not roll it back.
    """

    from dryml.data import materialize_supervised

    callbacks = validate_training_callbacks(callbacks)
    if callbacks or observer_session is not None:
        raise NotImplementedError("One-shot training does not support intermediate safe points or observers.")
    model_slots, exp_slots = vars(exp.model), vars(exp)
    if (type(model_slots) is not dict or type(exp_slots) is not dict
            or "_accepted_fit" not in model_slots or "state" not in exp_slots):
        raise TypeError("One-shot acceptance requires ordinary accepted-fit and state instance slots.")
    prior_fit, prior_state = model_slots["_accepted_fit"], exp_slots["state"]
    if type(prior_state) is not TrainState:
        raise TypeError("One-shot acceptance requires an exact TrainState.")
    proposed = TrainState()
    proposed.__setstate__(prior_state.__getstate__())
    if (proposed.next_batch != 0 or proposed.target_epoch is not None
            or proposed.pending_epoch_postlude is not None):
        raise ValueError("One-shot fitting cannot continue unfinished iterative training progress.")
    require_supervised_dataset(exp.train_data, batched=False)
    x, y, examples = materialize_supervised(exp.train_data, strict_numpy=True)
    proposed.record_fit(examples=examples)
    proposed.finish_epoch()
    TrainState._validate_restored_state(proposed.__getstate__()["fields"])

    try:
        exp.model.prep_train()
        candidate = fit_candidate(x, y)
    except BaseException:
        try:
            exp.model.prep_eval()
        except BaseException:
            pass
        raise
    else:
        exp.model.prep_eval()
    accepted = encode_candidate(candidate)
    if accepted is None:
        raise ValueError("One-shot encoding must return an accepted native-state bundle.")

    # Existing plain dictionary slots avoid descriptors or user installation hooks.
    try:
        dict.__setitem__(model_slots, "_accepted_fit", accepted)
        dict.__setitem__(exp_slots, "state", proposed)
    except BaseException:
        dict.__setitem__(model_slots, "_accepted_fit", prior_fit)
        dict.__setitem__(exp_slots, "state", prior_state)
        raise
    return accepted


def record_train_update(
    exp,
    batch,
    loss: float,
    *,
    batched: bool,
    callbacks=(),
    examples: int | None = None,
    complete_epoch: bool = False,
    epoch_metrics: dict[str, float] | None = None,
) -> None:
    """Retain one successful update before invoking process-local safe points.

    Args:
        exp: Experiment whose TrainState receives retained accounting.
        batch: Prepared training value used to determine examples in this update.
        loss: Finite mean loss for the completed update.
        batched: Whether ``batch`` has a leading example dimension.
        callbacks: Zero-argument callbacks run after retained accounting.
        examples: Exact completed-update examples when an owned backend step
            reports them.  When omitted, derive them from ``batch``.
        complete_epoch: Whether this completed update exhausted its logical
            epoch and must normalize state before safe-point callbacks.
        epoch_metrics: Finite completed-epoch metrics to retain before callbacks
            when ``complete_epoch`` is true.

    Raises:
        TypeError: If callbacks are not callable.
        ValueError: If a batched update has no examples.

    Side Effects:
        Advances TrainState optimizer/exposure/loss-window state, then invokes
        callbacks. A callback failure does not roll back the completed update.
    """

    derive_examples = examples is None
    if derive_examples:
        examples = 1
    if derive_examples and batched:
        try:
            first = next(iter_leaves(batch))
            examples = len(first)
        except (StopIteration, TypeError):
            raise ValueError("A batched training update must expose a leading example dimension.") from None
    exp.state.record_update(examples=examples, loss=float(loss))
    if complete_epoch:
        exp.state.finish_epoch(postlude_pending=True, metrics=epoch_metrics)
    elif epoch_metrics is not None:
        raise ValueError("epoch_metrics require a completed epoch update.")
    for callback in callbacks:
        callback()


def signature_discovery(obj: Any, **kwargs):
    try:
        check_context('tf')
        from .tf.utils import tf_signature_discovery
        return tf_signature_discovery(obj, **kwargs)
    except expected_context_errors:
        pass

    raise ValueError("Unable to guess a signature based on the object.")


__all__ = [
    "advance_train_state",
    "finite_dataset_len",
    "record_train_update",
    "require_bounded_safe_points",
    "require_supervised_dataset",
    "signature_discovery",
    "TrainingPreparation",
    "training_cardinality",
    "validate_training_callbacks",
]
