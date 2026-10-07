"""Experimental JAX model state owners and pure training-transition seams.

This module intentionally supplies state ownership and prediction only. The
training implementation owns differentiated updates and coordinated installation.
"""

from __future__ import annotations

from dataclasses import replace

from dryml.core.backend import Backend
from dryml.core.factory import FactorySpec
from dryml.core.object import Serializable
from dryml.core.utils.recurse import map_leaf_groups, map_leaves
from dryml.core.utils.stable_hash import stable_hash_function
from dryml.methods import ImplementationSelectionError, traits
from dryml.models import Model as BaseModel
from dryml.models import TrainFunction as BaseTrainFunction

from .state import (
    _read_tree_payload,
    _restore_tree_payload,
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

        if self.trainable_mask is None:
            self._trainable_parameters = tuple(jax.tree_util.tree_leaves(self.parameters))
            return
        leaves, tree = jax.tree_util.tree_flatten(self.parameters)
        enabled, enabled_tree = jax.tree_util.tree_flatten(self.trainable_mask)
        if tree != enabled_tree or not all(type(value) is bool for value in enabled):
            raise TypeError("trainable_mask must be a boolean pytree matching initialized parameters.")
        self._trainable_parameters = tuple(
            parameter for parameter, include in zip(leaves, enabled) if include
        )

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


__all__ = ["Model", "Optimizer", "TrainFunction", "pure_training_transition"]
