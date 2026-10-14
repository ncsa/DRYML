"""Experimental Flax NNX adapter sharing the JAX Model candidate seam."""

from __future__ import annotations

from dryml.core.factory import FactorySpec
from dryml.core.utils.stable_hash import stable_hash_function

from .base import Model, _leaf_aliases, _require_factory, _tree_to_jax, _validate_key, _validate_predictions
from .state import read_owner_envelope, read_tree_state, write_owner_envelope, write_tree_state


def _validate_state_like(value, template, name: str, jax) -> None:
    """Validate native NNX state topology without pretending it is a plain pytree."""

    leaves, tree = jax.tree_util.tree_flatten(value)
    expected, expected_tree = jax.tree_util.tree_flatten(template)
    if tree != expected_tree or len(leaves) != len(expected):
        raise TypeError(f"NNX candidate {name} topology does not match the current module state.")
    for leaf, prior in zip(leaves, expected):
        if not isinstance(leaf, jax.Array) or leaf.dtype != prior.dtype or leaf.shape != prior.shape:
            raise TypeError(f"NNX candidate {name} dtype or shape does not match the current module state.")


class NNXModel(Model):
    """Experimental Flax NNX Model with partitioned state and snapshot prediction.

    Args:
        factory: ``F(...)`` factory for an ``nnx.Module``.
        *factory_args: Additional authored module-factory positional arguments.
        output_spec: Explicit prediction specification; no forward probe occurs.
        seed: Exact integer used for distinct module-construction and external
            training RNG keys.
        **factory_kwargs: Additional authored module-factory keyword arguments.

    The adapter injects its reserved ``rngs`` argument after runtime admission.
    Parameters, remaining mutable variables, and module RNG streams stay
    partitioned in one shared candidate/training seam. Public prediction runs a
    cloned module snapshot and discards every candidate state change. This API is
    experimental and may change.

    Raises:
        TypeError: If the seed, output specification, factory, module, or
            prediction result violates the NNX adapter contract.
        ValueError: If authored configuration supplies the reserved ``rngs`` key.

    Side Effects:
        Imports JAX and Flax and constructs the module after runtime admission.
        Public predictions execute a cloned module and do not install candidates.
    """

    def __init__(self, factory, *factory_args, output_spec=None, seed: int = 0, **factory_kwargs):
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer.")
        if output_spec is None:
            raise TypeError("NNXModel requires an explicit output_spec.")
        self.factory = _require_factory(factory, "factory")
        if "rngs" in self.factory.kwargs or "rngs" in factory_kwargs:
            raise ValueError("NNXModel factory must not configure reserved 'rngs'.")
        self.factory_args = tuple(factory_args)
        self.factory_kwargs = dict(factory_kwargs)
        self.output_spec = output_spec
        self.trainable_mask = None

        import jax
        from flax import nnx

        init_key, self.rng = jax.random.split(jax.random.key(seed))
        kwargs = {**self.factory.kwargs, **self.factory_kwargs, "rngs": nnx.Rngs(init_key)}
        runtime_factory = FactorySpec(self.factory.target, *self.factory.args, *self.factory_args, **kwargs)
        self.obj = runtime_factory.build(instance_type=nnx.Module)
        _, self.parameters, self.mutable_state = nnx.split(
            self.obj, nnx.Param, nnx.Not(nnx.Param),
        )
        self._refresh_trainable_parameters()

    def _refresh_trainable_parameters(self) -> None:
        """Refresh derived references after NNX state installation."""

        import jax

        leaves = jax.tree_util.tree_leaves(self.parameters)
        self._parameter_aliases = _leaf_aliases(self.parameters, jax)
        self._trainable_mask_leaves = None
        self._trainable_parameters = tuple(leaves)

    def _candidate_apply(self, value, *, training: bool):
        """Run NNX on a cloned state and return its validated training candidate."""

        import jax
        from flax import nnx

        if type(training) is not bool:
            raise TypeError("NNXModel training mode must be an exact bool.")
        graphdef, parameters, mutable_state = nnx.split(
            self.obj, nnx.Param, nnx.Not(nnx.Param),
        )
        candidate = nnx.merge(graphdef, parameters, mutable_state)
        if hasattr(candidate, "train") and hasattr(candidate, "eval"):
            candidate.train() if training else candidate.eval()
        predictions = candidate(_tree_to_jax(value, jax))
        _, candidate_parameters, candidate_mutable = nnx.split(
            candidate, nnx.Param, nnx.Not(nnx.Param),
        )
        _validate_predictions(predictions, jax)
        next_rng, _ = jax.random.split(self.rng)
        self._validate_candidate_state(candidate_parameters, candidate_mutable, next_rng)
        return predictions, candidate_mutable, next_rng

    def _training_apply(self, parameters, mutable_state, rng, value):
        """Apply an NNX state candidate without installing it in this Model.

        Args:
            parameters: Candidate NNX ``Param`` state being differentiated.
            mutable_state: Candidate non-parameter NNX state.
            rng: Candidate external typed JAX key.
            value: JAX-native prepared feature batch.

        Returns:
            Predictions, candidate mutable state, and a split next-use external
            key. The live module remains unchanged.

        Side Effects:
            Creates an ephemeral merged NNX module suitable for JAX transforms;
            it never updates the authoritative module or wrapper fields.
        """

        import jax
        from flax import nnx

        graphdef, _, _ = nnx.split(self.obj, nnx.Param, nnx.Not(nnx.Param))
        candidate = nnx.merge(graphdef, parameters, mutable_state)
        if hasattr(candidate, "train"):
            candidate.train()
        predictions = candidate(value)
        _, _, candidate_mutable = nnx.split(candidate, nnx.Param, nnx.Not(nnx.Param))
        next_rng, _ = jax.random.split(rng)
        return predictions, candidate_mutable, next_rng

    def _validate_candidate_state(self, parameters, mutable_state, rng) -> None:
        """Validate one partitioned NNX training candidate without installing it."""

        import jax

        _validate_state_like(parameters, self.parameters, "parameters", jax)
        _validate_state_like(mutable_state, self.mutable_state, "mutable_state", jax)
        _validate_key(rng, self.rng, "candidate next_rng", jax)

    def _install_candidate_state(self, parameters, mutable_state, rng) -> None:
        """Install a validated NNX candidate without introducing a training loop."""

        from flax import nnx

        self._validate_candidate_state(parameters, mutable_state, rng)
        nnx.update(self.obj, parameters, mutable_state)
        self.parameters = parameters
        self.mutable_state = mutable_state
        self.rng = rng
        self._refresh_trainable_parameters()

    def _state_tree(self):
        from flax import nnx

        _, self.parameters, self.mutable_state = nnx.split(
            self.obj, nnx.Param, nnx.Not(nnx.Param),
        )
        self._refresh_trainable_parameters()
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
        """Save partitioned NNX values, not a GraphDef or device object."""

        del codec
        write_owner_envelope(dest_dir, "nnx-owner", factory=stable_hash_function(self.factory))
        write_tree_state(dest_dir, "model-state", self._state_tree())

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Validate NNX identity and local topology before state installation."""

        del codec
        envelope = read_owner_envelope(src_dir, "nnx-owner", fields={"factory"})
        if envelope["factory"] != stable_hash_function(self.factory):
            raise ValueError("Saved NNX Model state belongs to an incompatible factory.")
        self._install_state_tree(read_tree_state(src_dir, "model-state", self._state_tree()))


FlaxModel = NNXModel
"""Experimental spelling for the first-class Flax NNX adapter."""


__all__ = ["FlaxModel", "NNXModel"]
