from __future__ import annotations

from dataclasses import replace

from dryml.core.tensor_spec import Dynamic, batch_spec_tree, iter_specs
from dryml.methods import Method


class Model(Method):
    """Base model Method usable directly and in spec-specialized data nodes.

    Args:
        output_spec: Optional normalized output specification. When its leaves
            are unbatched, an input batch contract is propagated to the result.

    ``output_spec`` avoids backend-specific inference. Subclasses without a
    pure metadata inference route must require it rather than execute a model
    while deriving a specification.
    """

    def __init__(self, output_spec=None):
        self.output_spec = output_spec

    def prep_train(self):
        pass

    def prep_eval(self):
        pass

    def infer_output_spec(self, input_spec):
        if self.output_spec is None:
            return super().infer_output_spec(input_spec)
        if any(spec.batched for spec in iter_specs(self.output_spec)):
            return self.output_spec

        input_batches = {spec.batch for spec in iter_specs(input_spec) if spec.batched}
        if input_batches:
            batch = input_batches.pop() if len(input_batches) == 1 else Dynamic
            return batch_spec_tree(self.output_spec, batch=batch)
        return self.output_spec


class AutoEncoder(Model):
    """Compose an encoder and decoder Model as one locally specialized Method.

    Args:
        encoder: Model that transforms the input into the decoder input.
        decoder: Model that transforms the encoded value into the result.
        output_spec: Optional explicit final output specification.

    Selected AutoEncoder calls select each child once from threaded pure specs;
    they do not change either child's process-local Method preparation state.
    """

    def __init__(self, encoder: Model, decoder: Model, output_spec=None):
        self.encoder = encoder
        self.decoder = decoder
        self.output_spec = output_spec

    def prep_train(self):
        self.encoder.prep_train()
        self.decoder.prep_train()

    def prep_eval(self):
        self.encoder.prep_eval()
        self.decoder.prep_eval()

    def __call__(self, x):
        return self.decoder(self.encoder(x))

    def find_implementation(self, input_spec=None, *, backend=None, batch_mode=None):
        """Select this composite and both children from one threaded input spec.

        Args:
            input_spec: Normalized input specification for the encoder.
            backend: Optional explicit backend selection constraint.
            batch_mode: Optional explicit batch-mode selection constraint.

        Returns:
            A callable implementation that validates the outer input and invokes
            locally selected encoder and decoder implementations.

        Raises:
            ImplementationSelectionError: If this Model or either child cannot
            be selected from the supplied constraints.
        """

        implementation = super().find_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            learning=False,
        )

    def _prepare_implementation(self, input_spec, *, backend, batch_mode):
        """Build the learning-time local child invoker without shared caches."""

        implementation = super()._prepare_implementation(
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
        )
        return self._specialize_implementation(
            implementation,
            input_spec,
            backend=backend,
            batch_mode=batch_mode,
            learning=True,
        )

    def _specialize_implementation(
        self,
        implementation,
        input_spec,
        *,
        backend,
        batch_mode,
        learning,
    ):
        """Attach child callables selected from pure intermediate specifications."""

        if input_spec is None:
            return implementation
        if learning:
            def select(model, spec, selected_backend):
                return model._prepare_implementation(
                    spec,
                    backend=selected_backend,
                    batch_mode=batch_mode,
                )
        else:
            def select(model, spec, selected_backend):
                return model.find_implementation(
                    spec,
                    backend=selected_backend,
                    batch_mode=batch_mode,
                )
        encoder = select(self.encoder, input_spec, backend)
        encoded_spec = self.encoder.infer_output_spec(input_spec)
        decoder = select(self.decoder, encoded_spec, None)

        def invoke_autoencoder(x):
            return decoder(encoder(x))

        return replace(implementation, _invoker=invoke_autoencoder)

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)
        return self.decoder.infer_output_spec(self.encoder.infer_output_spec(input_spec))


__all__ = ["AutoEncoder", "Model"]
