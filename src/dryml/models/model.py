from __future__ import annotations

from dryml.core.methods import Method
from dryml.core.tensor_spec import Dynamic, batch_spec_tree, iter_specs


class Model(Method):
    """Base model API: a model is a Method usable in Dataset Map nodes."""

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

    def bind_first(self, first_value, *, input_spec=None):
        encoder_impl, encoded = self.encoder.bind_first(first_value, input_spec=input_spec)
        encoded_spec = self.encoder.infer_output_spec(input_spec) if input_spec is not None else None
        decoder_impl, decoded = self.decoder.bind_first(encoded, input_spec=encoded_spec)

        def bound_autoencoder(x):
            return decoder_impl(encoder_impl(x))

        return bound_autoencoder, decoded

    def infer_output_spec(self, input_spec):
        if self.output_spec is not None:
            return super().infer_output_spec(input_spec)
        return self.decoder.infer_output_spec(self.encoder.infer_output_spec(input_spec))


__all__ = ["AutoEncoder", "Model"]
