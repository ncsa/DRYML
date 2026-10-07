"""Experimental JAX model state owners and pure training-transition seams.

This module intentionally supplies state ownership and prediction only. The
training implementation owns differentiated updates and coordinated installation.
"""

from __future__ import annotations

import math
import os
from dataclasses import fields
from dataclasses import replace

from dryml.core.backend import Backend
from dryml.core.factory import FactorySpec
from dryml.core.object import Serializable
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.core.utils.stable_hash import stable_hash_function
from dryml.methods import ImplementationSelectionError, traits
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction
from dryml.models.experiment import _notify_host_observers
from dryml.models.progress import TrainingProgress
from dryml.models.train_spec import (
    _EarlyStoppingState,
    _retained_early_stop_completed,
    _update_early_stopping,
    _validate_early_stopping_config,
)
from dryml.models.utils import (
    TrainingPreparation,
    finite_dataset_len,
    require_bounded_safe_points,
    require_supervised_dataset,
    validate_training_callbacks,
)

from .state import (
    _read_tree_payload,
    _restore_tree_payload,
    _write_tree_payload,
    read_owner_envelope,
    read_tree_state,
    write_owner_envelope,
    write_tree_state,
)


def _require_factory(value, name: str) -> FactorySpec:
    if not isinstance(value, FactorySpec):
        raise TypeError(f"{name} must be an explicit FactorySpec. Use F(...).")
    return value


def _tree_to_jax(value, jax):
    return map_leaves(value, jax.numpy.asarray)


def _tree_to_jax_batch(value, input_spec, jax):
    def convert(values):
        leaf, spec = values
        array = jax.numpy.asarray(leaf)
        return array if spec.batched else jax.numpy.expand_dims(array, axis=0)

    return map_leaf_groups((value, input_spec), convert)


def _unbatch_tree(value):
    return map_leaves(value, lambda leaf: leaf[0])


def _validate_plain_array_tree(value, name: str, jax) -> None:
    """Reject functional state outside the supported plain-container contract."""

    def visit(item):
        if type(item) is dict:
            for child in item.values():
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
        elif not isinstance(item, jax.Array):
            raise TypeError(f"JAX Model {name} must be a plain pytree of jax.Array leaves.")

    visit(value)


def _leaf_aliases(value, jax):
    aliases = {}
    result = []
    for leaf in jax.tree_util.tree_leaves(value):
        result.append(aliases.setdefault(id(leaf), len(aliases)))
    return tuple(result)


def _validate_tree_like(value, template, name: str, jax) -> None:
    """Require a complete candidate tree to match a current native template."""

    _validate_plain_array_tree(value, name, jax)
    _validate_plain_array_tree(template, name, jax)
    leaves, tree = jax.tree_util.tree_flatten(value)
    template_leaves, template_tree = jax.tree_util.tree_flatten(template)
    if tree != template_tree or _leaf_aliases(value, jax) != _leaf_aliases(template, jax):
        raise TypeError(f"JAX Model {name} tree topology does not match its current state.")
    for leaf, expected in zip(leaves, template_leaves):
        if leaf.dtype != expected.dtype or leaf.shape != expected.shape:
            raise TypeError(f"JAX Model {name} leaf dtype or shape does not match its current state.")


def _validate_key(value, template, name: str, jax) -> None:
    if not isinstance(value, jax.Array) or not jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        raise TypeError(f"JAX Model {name} must be a typed JAX PRNG key.")
    if (
        value.dtype != template.dtype
        or value.shape != template.shape
        or jax.random.key_impl(value) != jax.random.key_impl(template)
    ):
        raise TypeError(f"JAX Model {name} does not match the owned PRNG key contract.")


def _validate_predictions(value, jax):
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or not all(isinstance(leaf, jax.Array) for leaf in leaves):
        raise TypeError("JAX Model apply predictions must be a nonempty pytree of jax.Array values.")
    return value


def _synchronize_tree(value, jax) -> None:
    """Wait for every candidate array before declaring a JAX update accepted."""

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def _require_finite_tree(value, name: str, jax) -> None:
    """Reject nonfinite floating candidate leaves before eager owner installation."""

    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.numpy.inexact):
            if not bool(jax.device_get(jax.numpy.all(jax.numpy.isfinite(leaf)))):
                raise ValueError(f"JAX Training candidate {name} contains nonfinite values.")


def _restore_train_state(target, source) -> None:
    """Repair a TrainState in place after an interrupted candidate commit."""

    for item in fields(source):
        setattr(target, item.name, getattr(source, item.name))


def _training_commit_boundary(stage: str) -> None:
    """Expose bounded eager commit stages for interruption characterization tests."""

    del stage


def _parameter_template(value, jax) -> dict:
    """Return JSON-safe parameter topology evidence without persisting arrays."""

    paths_and_leaves, tree = jax.tree_util.tree_flatten_with_path(value)
    return {
        "tree": str(tree),
        "aliases": list(_leaf_aliases(value, jax)),
        "leaves": [
            {
                "path": [
                    f"{type(part).__name__}:{getattr(part, 'key', getattr(part, 'idx', part))}"
                    for part in path
                ],
                "dtype": str(leaf.dtype),
                "shape": list(leaf.shape),
            }
            for path, leaf in paths_and_leaves
        ],
    }


def pure_training_transition(transition, model_state, optimizer_state, *args, **kwargs):
    """Run a pure training candidate transition without installing owner state.

    Args:
        transition: Callable receiving the supplied candidate states and inputs.
        model_state: Model parameter, mutable-state, and RNG candidate state.
        optimizer_state: Optimizer slot candidate state.
        *args: Additional pure transition inputs.
        **kwargs: Additional pure transition keyword inputs.

    Returns:
        The transition result, without owner mutation or persistence.

    Raises:
        TypeError: If ``transition`` is not callable.
    """

    if not callable(transition):
        raise TypeError("JAX training transition must be callable.")
    return transition(model_state, optimizer_state, *args, **kwargs)


class Model(BaseModel, Serializable):
    """Experimental functional JAX Model with separately owned native state.

    Args:
        init_fn: ``F(...)`` factory for an initializer accepting an initialization
            key followed by ``init_args`` and ``init_kwargs``.
        apply_fn: ``F(...)`` factory for an apply callable accepting parameters,
            mutable state, RNG state, input tree, and a training boolean.
        *init_args: Authored positional arguments passed to the initializer.
        output_spec: Explicit prediction specification; no forward probe occurs.
        seed: Exact integer used to derive distinct initialization and next-use
            typed JAX keys.
        trainable_mask: Optional boolean pytree matching initialized parameters.
        **init_kwargs: Authored keyword arguments passed to the initializer.

    The initializer returns exactly ``(parameters, mutable_state)``. The apply
    function returns exactly ``(predictions, candidate_mutable_state,
    candidate_next_rng)``. Public prediction returns only predictions and always
    discards candidate state. This API is experimental and may change.

    Raises:
        TypeError: If factories, the seed, output specification, initialized
            state, trainable mask, or later apply results violate the contract.

    Side Effects:
        Imports JAX and executes the initializer after runtime admission. Public
        prediction evaluates candidate state but does not install it.
    """

    native_backend = "jax"

    def __init__(
        self,
        init_fn,
        apply_fn,
        *init_args,
        output_spec=None,
        seed: int = 0,
        trainable_mask=None,
        **init_kwargs,
    ):
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer.")
        if output_spec is None:
            raise TypeError("JAX Model requires an explicit output_spec.")
        self.init_fn = _require_factory(init_fn, "init_fn")
        self.apply_fn = _require_factory(apply_fn, "apply_fn")
        self.init_args = tuple(init_args)
        self.init_kwargs = dict(init_kwargs)
        self.output_spec = output_spec
        self.trainable_mask = trainable_mask

        # Factory resolution and all native values begin only once this runtime is used.
        import jax

        self._init = self.init_fn.build()
        self._apply = self.apply_fn.build()
        if not callable(self._init) or not callable(self._apply):
            raise TypeError("JAX Model init_fn and apply_fn factories must build callables.")
        init_key, self.rng = jax.random.split(jax.random.key(seed))
        initialized = self._init(init_key, *self.init_args, **self.init_kwargs)
        if type(initialized) is not tuple or len(initialized) != 2:
            raise TypeError("JAX Model initializer must return exactly (parameters, mutable_state).")
        self.parameters, self.mutable_state = initialized
        _validate_plain_array_tree(self.parameters, "parameters", jax)
        _validate_plain_array_tree(self.mutable_state, "mutable_state", jax)
        self._refresh_trainable_parameters()

    def _refresh_trainable_parameters(self) -> None:
        """Rebuild derived trainable references after initialization or restore."""

        import jax

        leaves, tree = jax.tree_util.tree_flatten(self.parameters)
        self._parameter_aliases = _leaf_aliases(self.parameters, jax)
        if self.trainable_mask is None:
            self._trainable_mask_leaves = None
            self._trainable_parameters = tuple(leaves)
            return
        enabled, enabled_tree = jax.tree_util.tree_flatten(self.trainable_mask)
        if tree != enabled_tree or not all(type(value) is bool for value in enabled):
            raise TypeError("trainable_mask must be a boolean pytree matching initialized parameters.")
        self._trainable_mask_leaves = tuple(enabled)
        self._trainable_parameters = tuple(
            parameter for parameter, include in zip(leaves, enabled) if include
        )

    def _masked_candidate_parameters(self, parameters, candidate, jax):
        """Retain frozen leaves and shared-leaf topology after an Optax candidate.

        Args:
            parameters: Parameter pytree supplied to the differentiated update.
            candidate: Full parameter candidate returned by Optax.
            jax: Imported JAX module used for pytree reconstruction.

        Returns:
            A candidate with every masked leaf taken from ``parameters`` and each
            originally shared leaf represented once.

        Side Effects:
            None. This is a pure pytree selection used while tracing the native
            transition.
        """

        prior, tree = jax.tree_util.tree_flatten(parameters)
        updated, updated_tree = jax.tree_util.tree_flatten(candidate)
        if tree != updated_tree or len(prior) != len(updated):
            raise TypeError("JAX Optimizer parameter candidate topology changed during update.")
        selected = []
        aliases = {}
        mask = self._trainable_mask_leaves
        for index, (before, after, alias) in enumerate(zip(prior, updated, self._parameter_aliases)):
            if alias not in aliases:
                aliases[alias] = after if mask is None or mask[index] else before
            selected.append(aliases[alias])
        return jax.tree_util.tree_unflatten(tree, selected)

    def _runtime_selection_facts(self, args, kwargs):
        """Select the JAX eager Method route without prediction probing."""

        del args, kwargs
        return Backend.jax, None

    def _candidate_apply(self, value, *, training: bool):
        """Return one validated pure apply candidate without changing Model state.

        This is the training seam for a differentiated update. It validates the full
        mutable/RNG continuation while deliberately leaving parameter installation
        to the coordinated training transition.
        """

        import jax

        if type(training) is not bool:
            raise TypeError("JAX Model training mode must be an exact bool.")
        result = self._apply(
            self.parameters,
            self.mutable_state,
            self.rng,
            _tree_to_jax(value, jax),
            training,
        )
        if type(result) is not tuple or len(result) != 3:
            raise TypeError(
                "JAX Model apply must return exactly "
                "(predictions, candidate_mutable_state, candidate_next_rng)."
            )
        predictions, mutable_state, next_rng = result
        _validate_predictions(predictions, jax)
        self._validate_candidate_state(self.parameters, mutable_state, next_rng)
        return predictions, mutable_state, next_rng

    def _training_apply(self, parameters, mutable_state, rng, value):
        """Apply one functional candidate from explicit state inside native update.

        Args:
            parameters: Candidate parameter pytree being differentiated.
            mutable_state: Current candidate mutable-state pytree.
            rng: Current candidate typed PRNG key.
            value: JAX-native prepared feature batch.

        Returns:
            ``(predictions, mutable_state, next_rng)`` produced without changing
            any Model-owned value.

        Side Effects:
            None. This method is intentionally suitable for the JITted pure
            training transition; eager validation and installation stay outside.
        """

        return self._apply(parameters, mutable_state, rng, value, True)

    def _validate_candidate_state(self, parameters, mutable_state, rng) -> None:
        """Validate one training candidate without installing Model-owned values.

        Args:
            parameters: Candidate parameter tree produced by a future pure update.
            mutable_state: Candidate mutable model-state tree.
            rng: Candidate typed next-use JAX key.

        Raises:
            TypeError: If any candidate tree, leaf dtype/shape, or key contract
                differs from the current Model topology.
        """

        import jax

        _validate_tree_like(parameters, self.parameters, "candidate parameters", jax)
        _validate_tree_like(mutable_state, self.mutable_state, "candidate mutable_state", jax)
        _validate_key(rng, self.rng, "candidate next_rng", jax)

    def _install_candidate_state(self, parameters, mutable_state, rng) -> None:
        """Install a previously computed training candidate after local validation.

        This local operation performs no differentiation, callbacks, I/O, or
        persistence. The trainer coordinates all participating owners around it.
        """

        self._validate_candidate_state(parameters, mutable_state, rng)
        self.parameters = parameters
        self.mutable_state = mutable_state
        self.rng = rng
        self._refresh_trainable_parameters()

    def _call_raw(self, value, *args, **kwargs):
        if args or kwargs:
            raise TypeError("Experimental JAX Model prediction accepts exactly one input.")
        predictions, _, _ = self._candidate_apply(value, training=False)
        return predictions

    @traits(backend="jax")
    def raw_call(self, value, *args, **kwargs):
        """Return snapshot predictions for one raw input without installing state."""
        return self._call_raw(value, *args, **kwargs)

    @traits(backend="jax", batch_mode="batched")
    def batched_call(self, value, *args, **kwargs):
        """Return snapshot predictions for an already-batched selected input."""
        return self._call_raw(value, *args, **kwargs)

    @traits(backend="jax", batch_mode="element")
    def element_call(self, value, *args, **kwargs):
        """Return snapshot predictions for one selected logical element."""
        return self._call_raw(value, *args, **kwargs)

    def find_implementation(self, input_spec=None, *additional_input_specs, backend=None, batch_mode=None, output_spec=None):
        """Select a JAX prediction implementation with explicit element batching.

        Args:
            input_spec: Optional unary input specification.
            *additional_input_specs: Unsupported additional input specifications.
            backend: Optional required backend.
            batch_mode: Optional required element or batched mode.
            output_spec: Optional required result specification.

        Returns:
            A selected local Method implementation.

        Raises:
            ImplementationSelectionError: If the requested traits conflict.
        """

        import dryml.jax

        if additional_input_specs:
            raise ImplementationSelectionError("conflict")
        implementation = super().find_implementation(input_spec, backend=backend, batch_mode=batch_mode, output_spec=output_spec)
        return self._specialize_implementation(implementation, input_spec)

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Build a local learning-time JAX selection without global mutation."""

        import dryml.jax

        implementation = super()._prepare_implementation(input_spec, backend=backend, batch_mode=batch_mode)
        return self._specialize_implementation(implementation, input_spec)

    def _specialize_implementation(self, implementation, input_spec):
        if implementation.name != "element_call" or input_spec is None:
            return implementation

        def invoke_element(value, *args, **kwargs):
            import jax

            return _unbatch_tree(self._call_raw(_tree_to_jax_batch(value, input_spec, jax), *args, **kwargs))

        return replace(implementation, _invoker=invoke_element)

    def trainable_parameters(self, backend: str | None = None):
        """Return current parameter leaves selected by the retained mask.

        Args:
            backend: Optional backend selector.

        Returns:
            Current selected JAX parameter leaves, or an empty tuple for another
            requested backend. The returned objects remain owned by this Model.
        """

        return self._trainable_parameters if backend in (None, "jax") else ()

    def _state_tree(self):
        return {
            "parameters": self.parameters,
            "mutable_state": self.mutable_state,
            "rng": self.rng,
        }

    def _install_state_tree(self, state) -> None:
        self._install_candidate_state(
            state["parameters"], state["mutable_state"], state["rng"],
        )

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Save Model-owned parameters, mutable state, and typed RNG state."""

        del codec
        write_tree_state(dest_dir, "model-state", self._state_tree())

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Validate all Model state before replacing its local values."""

        del codec
        self._install_state_tree(read_tree_state(src_dir, "model-state", self._state_tree()))

    def infer_output_spec(self, input_spec, *additional_input_specs):
        """Return the explicit output spec without executing a model.

        Args:
            input_spec: Unary input specification whose batch metadata propagates.
            *additional_input_specs: Unsupported additional input specifications.

        Returns:
            The configured output specification with propagated batching.
        """

        return super().infer_output_spec(input_spec, *additional_input_specs)


class Optimizer(Serializable):
    """Experimental lazy Optax slot owner for a JAX Model parameter template.

    Args:
        factory: ``F(...)`` factory for an Optax gradient transformation.

    Construction neither imports Optax nor binds slots. Call :meth:`bind` from a
    training transition to build the transformation and initialize/restore slots
    for one matching Model template. This API is experimental and may change.
    """

    def __init__(self, factory):
        self.factory = _require_factory(factory, "factory")
        self.obj = None
        self.state = None
        self._template = None
        self._factory_identity = None
        self._pending_state = None

    def bind(self, model: Model):
        """Bind slots to one Model's current parameters or reject incompatibility.

        Args:
            model: Functional or NNX JAX model exposing current parameters.

        Returns:
            The reconstructed Optax transformation.

        Raises:
            TypeError: If ``model`` is not a JAX Model.
            ValueError: If parameters are empty, the factory is not an ordinary
                Optax transformation, or an existing/pending binding differs.
        """

        if not isinstance(model, Model):
            raise TypeError("JAX Optimizer binding requires a JAX Model.")
        import jax

        factory_identity = stable_hash_function(self.factory)
        if self._factory_identity is not None and self._factory_identity != factory_identity:
            raise ValueError("JAX Optimizer factory identity changed after binding.")
        template = _parameter_template(model.parameters, jax)
        if not template["leaves"]:
            raise ValueError("JAX Optimizer target exposes no parameters.")
        if len(set(template["aliases"])) != len(template["aliases"]):
            raise ValueError(
                "JAX functional training does not support shared parameter leaves; "
                "use distinct parameters or an NNX model."
            )
        if self._template is not None and self._template != template:
            raise ValueError("JAX Optimizer is already bound to an incompatible model parameter template.")
        if self.obj is None:
            import optax

            transformation = self.factory.build(namespace=optax)
            if not callable(getattr(transformation, "init", None)) or not callable(getattr(transformation, "update", None)):
                raise ValueError("JAX Optimizer factory must build an Optax gradient transformation.")
            state = transformation.init(model.parameters)
            if self._pending_state is not None:
                state = _restore_tree_payload(self._pending_state, state)
            self.obj = transformation
            self._template = template
            self._factory_identity = factory_identity
            self.state = state
            self._pending_state = None
        return self.obj

    def _validate_candidate_state(self, value) -> None:
        """Validate candidate Optax slots against the currently bound slot tree.

        Args:
            value: Candidate state returned by one Optax update.

        Raises:
            RuntimeError: If this Optimizer is not bound.
            TypeError: If candidate slots differ in pytree topology, leaf type,
                dtype, or shape from the current bound slot state.
        """

        if self.state is None:
            raise RuntimeError("JAX Optimizer candidate validation requires bound slots.")
        import jax

        leaves, tree = jax.tree_util.tree_flatten(value)
        current, current_tree = jax.tree_util.tree_flatten(self.state)
        if tree != current_tree or len(leaves) != len(current):
            raise TypeError("JAX Optimizer candidate slot topology does not match its current state.")
        for leaf, expected in zip(leaves, current):
            if (
                not isinstance(leaf, jax.Array)
                or not isinstance(expected, jax.Array)
                or leaf.dtype != expected.dtype
                or leaf.shape != expected.shape
            ):
                raise TypeError("JAX Optimizer candidate slot dtype or shape does not match its current state.")

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Save symbolic factory evidence and bound slot state when present."""

        del codec
        write_owner_envelope(
            dest_dir,
            "optimizer-owner",
            factory=self._factory_identity or stable_hash_function(self.factory),
            bound=self._template is not None,
            template=self._template,
        )
        if self._template is not None:
            write_tree_state(dest_dir, "optimizer-state", self.state)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Retain validated binding evidence until a Model binds this owner.

        No Optax import occurs for an unbound optimizer or inference-only graph.
        A later :meth:`bind` reconstructs the symbolic transformation against the
        current Model template before decoding slot leaves.
        """

        del codec
        envelope = read_owner_envelope(src_dir, "optimizer-owner", fields={"factory", "bound", "template"})
        if envelope["factory"] != stable_hash_function(self.factory):
            raise ValueError("Saved JAX Optimizer state belongs to an incompatible factory.")
        if type(envelope["bound"]) is not bool or (envelope["bound"] and not isinstance(envelope["template"], dict)):
            raise ValueError("Saved JAX Optimizer binding evidence is malformed.")
        if not envelope["bound"] and envelope["template"] is not None:
            raise ValueError("Saved unbound JAX Optimizer must not contain a parameter template.")
        pending_state = _read_tree_payload(src_dir, "optimizer-state") if envelope["bound"] else None
        self.obj = None
        self.state = None
        self._template = envelope["template"] if envelope["bound"] else None
        self._factory_identity = envelope["factory"] if envelope["bound"] else None
        self._pending_state = pending_state


class TrainFunction(BaseTrainFunction, Serializable):
    """Own experimental JAX training-behavior continuation state.

    Args:
        continuation: Optional plain pytree converted to JAX arrays.

    Side Effects:
        Imports JAX during admitted construction. This base owner performs no
        updates; concrete training behavior is supplied separately.
    """

    def __init__(self, *, continuation=None):
        import jax

        self.continuation = _tree_to_jax({} if continuation is None else continuation, jax)

    def __call__(self, exp, *, callbacks=()):
        """Reject direct execution because this type owns state but no loop.

        Args:
            exp: Reserved Experiment argument.
            callbacks: Reserved accepted-update callback sequence.

        Raises:
            NotImplementedError: Always; select a concrete JAX trainer instead.
        """

        del exp, callbacks
        raise NotImplementedError("Select a concrete JAX Training implementation; TrainFunction owns state only.")

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Save TrainFunction-owned behavior continuation only."""

        del codec
        write_tree_state(dest_dir, "train-function-state", self.continuation)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Validate continuation topology before replacing behavior state."""

        del codec
        self.continuation = read_tree_state(src_dir, "train-function-state", self.continuation)


class Training(TrainFunction):
    """Train an experimental functional or Flax NNX JAX Model with Optax.

    Args:
        optimizer: Independently owned :class:`Optimizer` slot wrapper.
        loss: Definition-compatible callable, or ``F(...)`` factory returning one,
            that accepts predictions and targets and returns a finite scalar mean.
        epochs: Exact nonnegative epoch count for a fresh invocation.
        metrics: Currently unsupported nonempty metric configuration.
        verbose: Exact integer progress verbosity accepted by TrainingProgress.

    Training consumes only canonical, explicitly batched ``(inputs, targets)``
    Dataset values from its Experiment. It preserves Dataset ordering and batch
    boundaries, including short final batches. Candidate native state is computed
    without owner mutation, synchronized and validated, then installed with Model,
    Optimizer, TrainFunction, and TrainState accounting in one bounded eager
    transition. This experimental API may change.

    Raises:
        TypeError: If construction or invocation values violate the documented
            callable, optimizer, Dataset, callback, or progress contracts.
        ValueError: If counts, loss values, candidate state, or retained recovery
            progress cannot support truthful training.

    Side Effects:
        A positive-update invocation imports JAX and Optax; zero epochs complete
        retained lifecycle state without preparing data or binding optimizer slots.
        Accepted synchronized updates change independently owned Model, Optimizer,
        TrainFunction, and Experiment state before invocation callbacks run.
    """

    __dryml_retired_constructor_parameters__ = (
        "batch_size", "num_examples", "shuffle", "shuffle_seed",
        "shuffle_buffer_size", "x_path", "y_path",
    )

    def __init__(self, *, optimizer, loss, epochs: int = 1, metrics=(), verbose: int = 1):
        """Record the explicit JAX-native training configuration.

        Args:
            optimizer: Independently owned Optax wrapper.
            loss: Scalar mean-loss callable or explicit factory returning one.
            epochs: Exact nonnegative requested epoch count.
            metrics: Reserved empty metric configuration.
            verbose: Exact integer progress verbosity.

        Raises:
            TypeError: If optimizer, loss, metrics, or verbosity is malformed.
            ValueError: If epochs is negative or metrics are requested.

        Side Effects:
            Initializes only TrainFunction-owned continuation state. Optax slots
            remain unbound until a Model is admitted for training.
        """

        if not isinstance(optimizer, Optimizer):
            raise TypeError("JAX Training requires an experimental JAX Optimizer wrapper.")
        if not isinstance(loss, FactorySpec) and not callable(loss):
            raise TypeError("JAX Training loss must be a callable or explicit FactorySpec.")
        if type(epochs) is not int or epochs < 0:
            raise ValueError("epochs must be a nonnegative exact integer.")
        if metrics is None:
            metrics = ()
        if not isinstance(metrics, (tuple, list)):
            raise TypeError("JAX Training metrics must be an empty tuple or list.")
        if metrics:
            raise ValueError("Experimental JAX Training does not yet support metrics.")
        if type(verbose) is not int:
            raise TypeError("verbose must be an exact integer.")
        super().__init__()
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.metrics = ()
        self.verbose = verbose
        self.training_preparation = None
        self.validation_preparation = None

    supports_observers = True

    def _validate_observer_session(self, observer_session) -> None:
        """Require host-callable telemetry before retained training starts."""

        observer_session.require_callables("JAX")

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Train one Experiment and retain only accepted JAX updates.

        Args:
            exp: Experiment providing a functional or NNX JAX Model, canonical
                training/validation Datasets, and retained TrainState.
            callbacks: Zero-argument safe-point callbacks invoked after each
                committed update and any completed-epoch normalization.
            observer_session: Private Experiment-owned invocation telemetry
                session, or ``None``.

        Returns:
            Scalar accepted training losses in invocation order.

        Raises:
            ValueError: If Dataset admission, mean-loss, finite-progress, or
                retained recovery facts are incompatible with this loop.
            TypeError: If Model, callbacks, or runtime candidates violate their
                explicit contracts.

        Side Effects:
            Installs synchronized Model/Optimizer/TrainState updates and may run
            caller callbacks after those updates are truthfully retained.
        """

        callbacks = validate_training_callbacks(callbacks)
        if observer_session is not None:
            observer_session.require_callables("JAX")
        if not isinstance(exp.model, Model):
            raise TypeError("JAX Training requires an experimental functional or NNX JAX Model.")
        train_data = exp.train_data
        require_supervised_dataset(train_data, batched=True)
        require_bounded_safe_points(train_data, callbacks)
        steps_per_epoch = finite_dataset_len(train_data)
        if exp.val_data is not None:
            require_supervised_dataset(exp.val_data, batched=True)
            if exp.val_data.yield_cardinality().is_infinite:
                raise ValueError("Validation on an infinite dataset requires an explicit finite bound.")
            if finite_dataset_len(exp.val_data) == 0:
                raise ValueError("Cannot validate on an empty dataset.")
        start_epoch = exp.state.epoch
        fresh_target = exp.state.target_epoch is None
        initial_step = exp.state.step
        preview_target = exp.state.target_epoch if exp.state.target_epoch is not None else start_epoch + self.epochs
        if steps_per_epoch == 0 and preview_target > start_epoch:
            raise ValueError("Cannot train on an empty dataset.")
        if preview_target == start_epoch and exp.state.pending_epoch_postlude is None:
            target_epoch = exp.state.begin_invocation(self.epochs)
            exp.state.finish_invocation(target_epoch)
            return []
        # Dataset admission and zero-work completion must not build a caller's
        # loss factory or enter native training runtime.
        loss_fn = self._loss_callable()
        self._begin_training_preparation_generation()
        training_preparation = TrainingPreparation.from_specs(
            self, train_data.spec[0], train_data.spec[1], "jax",
        )
        validation_preparation = None
        if exp.val_data is not None:
            validation_preparation = TrainingPreparation.from_specs(
                self, exp.val_data.spec[0], exp.val_data.spec[1], "jax",
            )
        prepared_train = train_data.prepare()
        prepared_val = exp.val_data.prepare() if exp.val_data is not None else None
        self.training_preparation = training_preparation
        self.validation_preparation = validation_preparation

        try:
            import jax
        except ImportError as error:
            raise ImportError("Experimental JAX Training requires the optional 'jax' dependency.") from error
        try:
            import optax
        except ImportError as error:
            raise ImportError(
                "Experimental JAX Training requires Optax; install the optional JAX training dependencies."
            ) from error
        transformation = self.optimizer.bind(exp.model)
        if self._completed_training_behavior_stop(exp):
            exp.state.finish_invocation(
                exp.state.target_epoch, accept_shortened=True,
            )
            return []
        from dryml.jax.training_data import iter_training_batches

        update = self._native_update(jax, optax, exp.model, transformation, loss_fn)
        target_epoch = exp.state.begin_invocation(self.epochs)
        total_steps = None if steps_per_epoch is None else steps_per_epoch * self.epochs
        progress = TrainingProgress(total=total_steps, verbose=self.verbose, desc="JAX training")
        losses = []
        steps = 0

        try:
            stopped = self._finish_pending_epoch(
                jax, exp, prepared_val, loss_fn, progress, target_epoch,
            )
            for epoch in range(start_epoch, target_epoch):
                if stopped:
                    break
                resume_batch = exp.state.next_batch if epoch == start_epoch else 0
                if steps_per_epoch is not None and resume_batch > steps_per_epoch:
                    raise ValueError("Saved JAX batch position exceeds the training epoch.")
                cursor = iter_training_batches(
                    prepared_train, self.training_preparation, epoch=epoch,
                )
                if resume_batch:
                    cursor.skip(resume_batch)
                epoch_loss = 0.0
                epoch_steps = 0
                try:
                    for x, y in cursor:
                        batch_index = exp.state.next_batch
                        examples = train_data.examples_in((x, y))
                        candidate = update(
                            exp.model.parameters,
                            exp.model.mutable_state,
                            exp.model.rng,
                            self.optimizer.state,
                            x,
                            y,
                        )
                        loss, parameters, mutable_state, rng, slots = candidate
                        _synchronize_tree(candidate, jax)
                        loss_float = self._validated_loss(loss, jax)
                        complete_epoch = (
                            steps_per_epoch is not None
                            and exp.state.next_batch + 1 == steps_per_epoch
                        )
                        self._commit_update(
                            exp,
                            parameters,
                            mutable_state,
                            rng,
                            slots,
                            examples=examples,
                            loss=loss_float,
                            complete_epoch=complete_epoch,
                            epoch_metrics=(
                                {"loss": (epoch_loss + loss_float) / (epoch_steps + 1)}
                                if complete_epoch else None
                            ),
                        )
                        for callback in callbacks:
                            callback()
                        losses.append(loss_float)
                        epoch_loss += loss_float
                        epoch_steps += 1
                        steps += 1
                        progress.update(1, {"loss": loss_float})
                        _notify_host_observers(observer_session, {
                            "event": "train_batch_end",
                            "epoch": epoch,
                            "batch": batch_index,
                            "step": exp.state.step,
                            "examples_seen": exp.state.examples_seen,
                            "loss": loss_float,
                        })
                finally:
                    cursor.close()
                if epoch_steps == 0 and resume_batch == 0:
                    continue
                epoch_metrics = {"loss": epoch_loss / epoch_steps} if epoch_steps and not resume_batch else {}
                stopped = self._finish_epoch(
                    jax, exp, epoch, epoch_metrics, prepared_val, loss_fn, progress, target_epoch,
                )
                _notify_host_observers(observer_session, {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "step": exp.state.step,
                    "examples_seen": exp.state.examples_seen,
                    "stopped": stopped,
                    **epoch_metrics,
                })
        except BaseException:
            if fresh_target:
                try:
                    exp.state.abandon_new_invocation(
                        target_epoch, initial_epoch=start_epoch, initial_step=initial_step,
                    )
                except ValueError:
                    pass
            raise
        finally:
            progress.close()

        if (
            steps == 0
            and target_epoch > start_epoch
            and exp.state.epoch < target_epoch
            and not stopped
        ):
            if fresh_target:
                exp.state.abandon_new_invocation(
                    target_epoch, initial_epoch=start_epoch, initial_step=initial_step,
                )
            raise ValueError("Cannot train on an empty dataset.")
        if exp.state.epoch < target_epoch and not stopped:
            raise ValueError("JAX training ended before its retained invocation target.")
        exp.state.finish_invocation(target_epoch, accept_shortened=stopped)
        return losses

    def _loss_callable(self):
        """Build and validate the configured definition-compatible loss callable."""

        value = self.loss.build() if isinstance(self.loss, FactorySpec) else self.loss
        if not callable(value):
            raise TypeError("JAX Training loss factory must build a callable.")
        return value

    @staticmethod
    def _native_update(jax, optax, model, transformation, loss_fn):
        """Return the sole JITted pure parameter-only native update function."""

        def transition(parameters, mutable_state, rng, slots, x, y):
            def objective(candidate_parameters):
                result = model._training_apply(candidate_parameters, mutable_state, rng, x)
                if type(result) is not tuple or len(result) != 3:
                    raise TypeError(
                        "JAX Model training apply must return exactly "
                        "(predictions, candidate_mutable_state, candidate_next_rng)."
                    )
                predictions, candidate_mutable, candidate_rng = result
                _validate_predictions(predictions, jax)
                return loss_fn(predictions, y), (candidate_mutable, candidate_rng)

            (loss, (candidate_mutable, candidate_rng)), gradients = jax.value_and_grad(
                objective, has_aux=True,
            )(parameters)
            updates, candidate_slots = transformation.update(gradients, slots, parameters)
            candidate_parameters = optax.apply_updates(parameters, updates)
            candidate_parameters = model._masked_candidate_parameters(
                parameters, candidate_parameters, jax,
            )
            return loss, candidate_parameters, candidate_mutable, candidate_rng, candidate_slots

        return jax.jit(transition)

    @staticmethod
    def _validated_loss(value, jax) -> float:
        """Return one synchronized finite scalar native loss or reject the update."""

        if getattr(value, "ndim", None) != 0:
            raise ValueError("JAX Training loss must return one scalar mean value.")
        result = float(jax.device_get(value))
        if not math.isfinite(result):
            raise ValueError("JAX Training loss must be finite.")
        return result

    def _commit_update(
        self, exp, parameters, mutable_state, rng, slots, *, examples, loss,
        complete_epoch, epoch_metrics=None,
    ):
        """Install one completely validated candidate or repair every owner on interruption."""

        import jax

        if type(examples) is not int or examples <= 0:
            raise ValueError("JAX Training update examples must be a positive exact integer.")
        if not math.isfinite(loss):
            raise ValueError("JAX Training loss must be finite.")
        exp.model._validate_candidate_state(parameters, mutable_state, rng)
        self.optimizer._validate_candidate_state(slots)
        _validate_plain_array_tree(self.continuation, "continuation", jax)
        _synchronize_tree((parameters, mutable_state, rng, slots, self.continuation), jax)
        _require_finite_tree(parameters, "parameters", jax)
        _require_finite_tree(mutable_state, "mutable_state", jax)
        _require_finite_tree(slots, "optimizer slots", jax)

        prior_model = (exp.model.parameters, exp.model.mutable_state, exp.model.rng)
        prior_slots = self.optimizer.state
        prior_continuation = self.continuation
        prior_state = type(exp.state)()
        _restore_train_state(prior_state, exp.state)
        candidate_state = type(exp.state)()
        _restore_train_state(candidate_state, exp.state)
        candidate_state.record_update(examples=examples, loss=loss)
        if complete_epoch:
            candidate_state.finish_epoch(
                postlude_pending=True,
                metrics=epoch_metrics,
            )
        candidate_state._validate_restored_state(candidate_state.__getstate__())
        committed = False
        try:
            _training_commit_boundary("before_model")
            exp.model._install_candidate_state(parameters, mutable_state, rng)
            _training_commit_boundary("after_model")
            self.optimizer.state = slots
            _training_commit_boundary("after_optimizer")
            _restore_train_state(exp.state, candidate_state)
            _training_commit_boundary("after_accounting")
            committed = True
        except BaseException as error:
            if not committed:
                repair_failures = []
                for repair in (
                    lambda: exp.model._install_candidate_state(*prior_model),
                    lambda: setattr(self.optimizer, "state", prior_slots),
                    lambda: setattr(self, "continuation", prior_continuation),
                    lambda: _restore_train_state(exp.state, prior_state),
                ):
                    try:
                        repair()
                    except BaseException as repair_error:
                        repair_failures.append(repair_error)
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    for repair_error in repair_failures:
                        add_note(
                            "JAX candidate rollback also failed: "
                            f"{type(repair_error).__name__}: {repair_error}"
                        )
            raise

    def _finish_pending_epoch(self, jax, exp, val_data, loss_fn, progress, target_epoch):
        """Complete retained validation/progress work without replaying an update."""

        epoch = exp.state.pending_epoch_postlude
        if epoch is None:
            return False
        metrics = dict(exp.state.pending_epoch_metrics or {})
        if exp.state.pending_epoch_postlude_phase == "start":
            if val_data is not None:
                metrics.update({f"val_{name}": value for name, value in self._evaluate(jax, exp.model, val_data, loss_fn).items()})
            exp.state.advance_epoch_postlude(epoch, "progress", metrics=metrics)
        if exp.state.pending_epoch_postlude_phase == "progress":
            stopped = self._finish_training_behavior_epoch(
                jax, exp, epoch, exp.state.pending_epoch_metrics or metrics
            )
            progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=metrics)
            exp.state.finish_epoch_postlude(epoch)
            return stopped
        return False

    def _finish_epoch(self, jax, exp, epoch, metrics, val_data, loss_fn, progress, target_epoch):
        """Normalize and complete one epoch postlude after its final accepted update."""

        if exp.state.epoch == epoch:
            exp.state.finish_epoch(postlude_pending=True)
        if exp.state.pending_epoch_postlude == epoch:
            if exp.state.pending_epoch_postlude_phase == "start":
                if val_data is not None:
                    metrics.update({f"val_{name}": value for name, value in self._evaluate(jax, exp.model, val_data, loss_fn).items()})
                exp.state.advance_epoch_postlude(epoch, "progress", metrics=metrics)
            if exp.state.pending_epoch_postlude_phase == "progress":
                stopped = self._finish_training_behavior_epoch(
                    jax, exp, epoch, exp.state.pending_epoch_metrics or metrics
                )
                progress.epoch_end(
                    epoch + 1, epochs=target_epoch,
                    metrics=exp.state.pending_epoch_metrics or metrics,
                )
                exp.state.finish_epoch_postlude(epoch)
                return stopped
            return False
        if val_data is not None:
            metrics.update({f"val_{name}": value for name, value in self._evaluate(jax, exp.model, val_data, loss_fn).items()})
        progress.epoch_end(epoch + 1, epochs=target_epoch, metrics=metrics)
        return self._finish_training_behavior_epoch(jax, exp, epoch, metrics)

    def _finish_training_behavior_epoch(self, jax, exp, epoch, metrics):
        """Run saved completed-epoch behavior; ordinary Training has none."""

        del jax, exp, epoch, metrics
        return False

    def _completed_training_behavior_stop(self, exp) -> bool:
        """Return whether saved behavior already completed a shortened target."""

        del exp
        return False

    def _evaluate(self, jax, model, data, loss_fn):
        """Evaluate snapshot candidates without retaining model mutable state or RNG."""

        from dryml.jax.training_data import iter_training_batches

        cursor = iter_training_batches(data, self.validation_preparation)
        total = 0.0
        examples_total = 0
        try:
            for x, y in cursor:
                predictions, _, _ = model._candidate_apply(x, training=False)
                value = loss_fn(predictions, y)
                examples = data.dataset.examples_in((x, y))
                total += self._validated_loss(value, jax) * examples
                examples_total += examples
        finally:
            cursor.close()
        return {} if not examples_total else {"loss": total / examples_total}


class EarlyStoppingTraining(Training):
    """Train JAX with recoverable completed-epoch early stopping.

    Args:
        monitor: Completed-epoch metric name. Initial JAX training supports
            ``"loss"`` and ``"val_loss"``.
        patience: Complete non-improving epochs tolerated before stopping.
        mode: ``"min"`` for decreasing metrics or ``"max"`` for increasing.
        min_delta: Required nonnegative absolute improvement.
        restore_best_weights: Restore best parameters and mutable model state;
            optimizer slots and model RNG remain at the stopping epoch.
        **kwargs: Arguments accepted by :class:`Training`.

    Raises:
        TypeError: If configuration types are invalid.
        ValueError: If configuration, validation data, or completed monitor facts
            are invalid or missing.

    Side Effects:
        Retains decision facts and a bounded best Model-state tree as
        TrainFunction-owned state. Optional restoration occurs once before the
        final Experiment graph checkpoint.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        **kwargs,
    ):
        monitor, patience, mode, min_delta, restore_best_weights = (
            _validate_early_stopping_config(
                monitor=monitor,
                patience=patience,
                mode=mode,
                min_delta=min_delta,
                restore_best_weights=restore_best_weights,
            )
        )
        if monitor not in ("loss", "val_loss"):
            raise ValueError(
                "Experimental JAX early stopping supports only 'loss' and 'val_loss'."
            )
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.early_stopping = _EarlyStoppingState()
        self._pending_best_model_payload = None

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Run or resume one retained early-stopping invocation.

        Args:
            exp: Experiment providing JAX Model, Datasets, and TrainState.
            callbacks: Truthful post-update DRYML safe-point callbacks.
            observer_session: Private Experiment-owned invocation telemetry
                session, or ``None``.

        Returns:
            Per-batch losses accepted before the full or shortened target.

        Raises:
            TypeError: If callbacks or native inputs violate Training contracts.
            ValueError: If recovery, Dataset, or completed monitor facts are
                invalid.

        Side Effects:
            Updates Model, Optimizer, Experiment, and this TrainFunction's saved
            continuation, and may restore best parameters/mutable state while
            retaining the stopping-epoch RNG.
        """

        fresh = exp.state.target_epoch is None
        if fresh:
            self.early_stopping.reset()
            self._pending_best_model_payload = None
        elif self._pending_best_model_payload is not None:
            self.early_stopping.best_model_state = _restore_tree_payload(
                self._pending_best_model_payload,
                (exp.model.parameters, exp.model.mutable_state),
            )
            self._pending_best_model_payload = None
        if self.monitor.startswith("val_") and exp.val_data is None:
            raise ValueError(
                f"Early-stopping monitor {self.monitor!r} requires validation data."
            )
        return super().__call__(
            exp, callbacks=callbacks, observer_session=observer_session,
        )

    def _finish_training_behavior_epoch(self, jax, exp, epoch, metrics):
        def capture():
            snapshot = jax.tree_util.tree_map(
                lambda value: jax.numpy.array(value, copy=True),
                (exp.model.parameters, exp.model.mutable_state),
            )
            _synchronize_tree(snapshot, jax)
            return snapshot

        def restore(snapshot):
            parameters, mutable_state = snapshot
            exp.model._validate_candidate_state(
                parameters, mutable_state, exp.model.rng,
            )
            exp.model._install_candidate_state(
                parameters, mutable_state, exp.model.rng,
            )

        return _update_early_stopping(
            self.early_stopping,
            epoch=epoch,
            metrics=metrics,
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            min_delta=self.min_delta,
            capture_best=capture,
            restore_best=restore,
            restore_best_weights=self.restore_best_weights,
        )

    def _completed_training_behavior_stop(self, exp) -> bool:
        """Recognize a retained stop after its normalized postlude completed."""

        return _retained_early_stop_completed(self.early_stopping, exp.state)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Persist decision metadata and the optional best Model-state tree.

        Args:
            dest_dir: Empty Store-owned local-state directory.
            codec: Opaque selected codec, accepted for hook compatibility.

        Side Effects:
            Writes metadata and a versioned host-array tree inside ``dest_dir``.
        """

        super().save_state_to_dir_imp(dest_dir, codec=codec)
        payload = self.early_stopping.to_payload()
        best_model_state = payload.pop("best_model_state")
        payload["has_best_model_state"] = (
            best_model_state is not None or self._pending_best_model_payload is not None
        )
        pickle_save(payload, os.path.join(dest_dir, "early-stopping.pkl"))
        if best_model_state is not None:
            write_tree_state(dest_dir, "early-stopping-best-model", best_model_state)
        elif self._pending_best_model_payload is not None:
            _write_tree_payload(
                dest_dir,
                "early-stopping-best-model",
                self._pending_best_model_payload,
            )

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Validate metadata and defer best-state topology binding to invocation.

        Args:
            src_dir: Store-owned directory containing continuation payloads.
            codec: Opaque selected codec, accepted for hook compatibility.

        Raises:
            ValueError: If metadata or host-array envelopes are malformed.

        Side Effects:
            Replaces decision facts and retains a validated in-memory payload for
            cross-owner topology validation when the Model is next admitted.
        """

        continuation = read_tree_state(
            src_dir,
            "train-function-state",
            self.continuation,
        )
        payload = pickle_load(os.path.join(src_dir, "early-stopping.pkl"))
        if type(payload) is not dict or type(payload.get("has_best_model_state")) is not bool:
            raise ValueError("Malformed JAX early-stopping continuation state.")
        has_best_model_state = payload.pop("has_best_model_state")
        payload["best_model_state"] = None
        early_stopping = _EarlyStoppingState.from_payload(payload)
        pending_best_model_payload = (
            _read_tree_payload(src_dir, "early-stopping-best-model")
            if has_best_model_state else None
        )
        self.continuation = continuation
        self.early_stopping = early_stopping
        self._pending_best_model_payload = pending_best_model_payload


__all__ = [
    "EarlyStoppingTraining",
    "Model",
    "Optimizer",
    "Training",
    "TrainFunction",
    "pure_training_transition",
]
