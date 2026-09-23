from __future__ import annotations

from dryml.core.factory import FactorySpec
from dryml.models.tf.base import Model


class Sequential(Model):
    """Keras Sequential model constructed from explicit layer factories.

    Args:
        layer_defs: A list or tuple of :class:`~dryml.core.FactorySpec` values
            resolved in ``tf.keras.layers`` when this model is constructed.
        output_spec: Optional explicit DRYML output specification.

    Raises:
        TypeError: If ``layer_defs`` or any of its elements is not an explicit
            FactorySpec, or a built layer is not a Keras Layer.

    Side Effects:
        Imports TensorFlow and constructs the declared Keras layers.
    """

    def __init__(self, layer_defs=(), output_spec=None):
        """Construct the Keras Sequential backend object from explicit factories.

        Args:
            layer_defs: A list or tuple containing only FactorySpec values.
            output_spec: Optional explicit DRYML output specification.

        Raises:
            TypeError: If layer declarations are not explicit factories or a
                constructed object is not a Keras Layer.

        Side Effects:
            Imports TensorFlow and constructs every validated layer.
        """
        import tensorflow as tf

        if not isinstance(layer_defs, (list, tuple)) or not all(
            isinstance(layer_def, FactorySpec) for layer_def in layer_defs
        ):
            raise TypeError(
                "Sequential layer definitions must be a list or tuple of explicit "
                "FactorySpec values. Use F(\"Dense\", 32) or FactorySpec(...)."
            )
        self.layer_defs = tuple(layer_defs)
        layers = [
            layer_def.build(
                namespace=tf.keras.layers,
                instance_type=tf.keras.layers.Layer,
            )
            for layer_def in self.layer_defs
        ]

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
