from __future__ import annotations

from dryml.core2.utils.general import validate_class
from dryml.models.tf.base import Model


class Sequential(Model):
    def __init__(self, layer_defs=(), output_spec=None):
        import tensorflow as tf

        self.layer_defs = tuple(layer_defs)
        layers = []
        for layer_def in self.layer_defs:
            if isinstance(layer_def, tf.keras.layers.Layer):
                layers.append(layer_def)
                continue

            if isinstance(layer_def, str):
                cls = getattr(tf.keras.layers, layer_def)
                args = ()
                kwargs = {}
            elif isinstance(layer_def, type):
                cls = layer_def
                args = ()
                kwargs = {}
            elif len(layer_def) == 2:
                cls, kwargs = layer_def
                cls = getattr(tf.keras.layers, cls) if isinstance(cls, str) else cls
                args = ()
            elif len(layer_def) == 3:
                cls, args, kwargs = layer_def
                cls = getattr(tf.keras.layers, cls) if isinstance(cls, str) else cls
            else:
                raise ValueError(
                    "Layer definitions must be layer names, layer classes, layer instances, "
                    "(cls, kwargs), or (cls, args, kwargs)."
                )
            layers.append(validate_class(cls)(*args, **kwargs))

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
