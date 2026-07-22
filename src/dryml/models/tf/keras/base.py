from __future__ import annotations

from dryml.core.factory import FactorySpec
from dryml.models.tf.base import Model


class Sequential(Model):
    @classmethod
    def __prepare_args__(cls, layer_defs=(), output_spec=None):
        args = (FactorySpec.coerce_many(layer_defs),)
        kwargs = {}
        if output_spec is not None:
            kwargs["output_spec"] = output_spec
        return args, kwargs

    def __init__(self, layer_defs=(), output_spec=None):
        from dryml.runtime import import_configured_framework
        tf = import_configured_framework("tensorflow")

        self.layer_defs = tuple(layer_defs)
        layers = []
        for layer_def in self.layer_defs:
            if isinstance(layer_def, FactorySpec):
                layers.append(
                    layer_def.build(
                        namespace=tf.keras.layers,
                        instance_type=tf.keras.layers.Layer,
                    )
                )
                continue

            raise TypeError(
                "Sequential layer definitions must be FactorySpec values. "
                "Tuple and string shorthands should be normalized by __prepare_args__."
            )

        self.obj = tf.keras.Sequential(layers)
        self.model = self.obj
        self.mdl = self.obj
        self.output_spec = output_spec
        self._pending_restore_path = None
        self._restore_checkpoint = None
        self._restore_status = None


__all__ = [
    "Sequential",
]
