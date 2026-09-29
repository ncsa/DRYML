from __future__ import annotations

import tensorflow as tf
import inspect
from typing import Any
from collections.abc import Mapping


class keras_train_spec_updater(tf.keras.callbacks.Callback):
    def __init__(self, train_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_spec = train_spec

    def on_epoch_end(self, epoch, logs=None):
        # Advance the train spec at end of an epoch
        self.train_spec.advance()


class keras_callback_wrapper(tf.keras.callbacks.Callback):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def on_epoch_end(self, epoch, logs=None):
        # Call the callback at the end of the epoch
        self.callback()


def save_tf_obj_weights(obj, path):
    """Save the weights of a TensorFlow object (e.g. model or layer) to a file."""
    obj.save_weights(path)


def load_tf_obj_weights(obj, path):
    """Load the weights of a TensorFlow object (e.g. model or layer) from a file."""
    obj.load_weights(path)


def tf_signature_discovery(
    obj: Any,
    *example_args: Any,
    example_kwargs: Mapping[str, Any] | None = None,
    trace: bool = True,
) -> dict[str, Any]:
    """
    Discover TensorFlow/Keras input/output signatures for an arbitrary object.

    Parameters
    ----------
    obj
        Any Python object. Common useful cases:
          - tf.keras.Model
          - tf.keras.layers.Layer
          - @tf.function-decorated callable
          - plain callable that uses TensorFlow ops
          - loaded SavedModel-like object exposing `.signatures`
          - ConcreteFunction-like object exposing structured signatures
    *example_args
        Optional example inputs or TensorSpecs used to force tracing.
        Each leaf may be:
          - tf.TensorSpec / tf.TypeSpec
          - tf.Tensor / tf.Variable
          - numpy array / scalar
          - Python scalar
    example_kwargs
        Optional kwargs used for tracing.
    trace
        If True, try to obtain a ConcreteFunction using example_args/example_kwargs.

    Returns
    -------
    dict
        A structured report containing whatever could be discovered.

    Notes
    -----
    - For subclassed models/layers and generic callables, a true tensor signature
      often requires tracing with example inputs or TensorSpecs.
    - If no TensorFlow-level signature can be found, this falls back to the
      Python call signature where possible.
    """
    import numpy as np
    import tensorflow as tf

    example_kwargs = dict(example_kwargs or {})

    def _safe_python_signature(x: Any) -> str | None:
        try:
            if isinstance(x, tf.keras.Model):
                return str(inspect.signature(x.call))
            if callable(x):
                return str(inspect.signature(x))
        except (TypeError, ValueError):
            return None
        return None

    def _shape_to_tuple(shape: Any) -> Any:
        if shape is None:
            return None
        try:
            # TensorShape
            rank = shape.rank
            if rank is None:
                return None
            return tuple(shape.as_list())
        except Exception:
            pass
        try:
            return tuple(shape)
        except Exception:
            return repr(shape)

    def _dtype_name(dtype: Any) -> str | None:
        if dtype is None:
            return None
        return getattr(dtype, "name", str(dtype))

    def _normalize_leaf_to_spec(x: Any) -> Any:
        # Preserve existing TF specs.
        if isinstance(x, tf.TypeSpec):
            return x

        # Variables / tensors.
        if isinstance(x, tf.Variable):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype, name=getattr(x, "name", None))
        if tf.is_tensor(x):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype, name=getattr(x, "name", None))

        # NumPy arrays / scalars.
        if isinstance(x, np.ndarray):
            return tf.TensorSpec(shape=x.shape, dtype=tf.as_dtype(x.dtype))
        if isinstance(x, np.generic):
            t = tf.convert_to_tensor(x)
            return tf.TensorSpec(shape=t.shape, dtype=t.dtype)

        # Python scalars that TF can convert.
        try:
            t = tf.convert_to_tensor(x)
            return tf.TensorSpec(shape=t.shape, dtype=t.dtype)
        except Exception:
            return x

    def _normalize_specs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        norm_args = tf.nest.map_structure(_normalize_leaf_to_spec, args)
        norm_kwargs = tf.nest.map_structure(_normalize_leaf_to_spec, kwargs)
        return norm_args, norm_kwargs

    def _summarize_leaf(x: Any) -> Any:
        # Concrete spec objects.
        if isinstance(x, tf.TypeSpec):
            out = {
                "kind": type(x).__name__,
                "repr": repr(x),
            }
            if hasattr(x, "shape"):
                out["shape"] = _shape_to_tuple(x.shape)
            if hasattr(x, "dtype"):
                out["dtype"] = _dtype_name(x.dtype)
            if hasattr(x, "name"):
                out["name"] = getattr(x, "name", None)
            return out

        # Runtime tensor values.
        if isinstance(x, tf.Variable):
            return {
                "kind": "Variable",
                "shape": _shape_to_tuple(x.shape),
                "dtype": _dtype_name(x.dtype),
                "name": getattr(x, "name", None),
            }
        if tf.is_tensor(x):
            return {
                "kind": type(x).__name__,
                "shape": _shape_to_tuple(x.shape),
                "dtype": _dtype_name(x.dtype),
                "name": getattr(x, "name", None),
            }

        # Keras symbolic tensors.
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return {
                "kind": type(x).__name__,
                "shape": _shape_to_tuple(getattr(x, "shape", None)),
                "dtype": _dtype_name(getattr(x, "dtype", None)),
                "name": getattr(x, "name", None),
            }

        return {
            "kind": type(x).__name__,
            "repr": repr(x),
        }

    def _summarize_structure(x: Any) -> Any:
        return tf.nest.map_structure(_summarize_leaf, x)

    def _summarize_input_signature(sig: Any) -> dict[str, Any]:
        args, kwargs = sig
        return {
            "args": _summarize_structure(args),
            "kwargs": _summarize_structure(kwargs),
        }

    def _discover_concrete_from_callable(callable_obj: Any) -> dict[str, Any]:
        norm_args, norm_kwargs = _normalize_specs(example_args, example_kwargs)

        if hasattr(callable_obj, "get_concrete_function"):
            fn = callable_obj
        else:
            fn = tf.function(callable_obj)

        cf = fn.get_concrete_function(*norm_args, **norm_kwargs)

        return {
            "function_type": str(getattr(cf, "function_type", None)),
            "structured_input_signature": _summarize_input_signature(
                cf.structured_input_signature
            ),
            "structured_outputs": _summarize_structure(cf.structured_outputs),
        }

    def _discover_symbolic_keras_signature(model: Any) -> dict[str, Any] | None:
        try:
            inputs = getattr(model, "inputs", None)
            outputs = getattr(model, "outputs", None)
        except Exception:
            return None

        if inputs is None or outputs is None:
            return None

        try:
            if len(tf.nest.flatten(inputs)) == 0 or len(tf.nest.flatten(outputs)) == 0:
                return None
        except Exception:
            return None

        info = {
            "built": bool(getattr(model, "built", False)),
            "inputs": _summarize_structure(inputs),
            "outputs": _summarize_structure(outputs),
        }

        try:
            info["input_shape"] = model.input_shape
        except Exception:
            pass

        try:
            info["output_shape"] = model.output_shape
        except Exception:
            pass

        return info

    result: dict[str, Any] = {
        "python_type": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "callable": callable(obj),
        "python_signature": _safe_python_signature(obj),
        "discoveries": {},
        "notes": [],
    }

    # 1) SavedModel-like object exposing named signatures.
    signatures = getattr(obj, "signatures", None)
    if isinstance(signatures, Mapping) and signatures:
        savedmodel_info: dict[str, Any] = {}
        for name, fn in signatures.items():
            if (
                hasattr(fn, "structured_input_signature")
                and hasattr(fn, "structured_outputs")
            ):
                savedmodel_info[name] = {
                    "structured_input_signature": _summarize_input_signature(
                        fn.structured_input_signature
                    ),
                    "structured_outputs": _summarize_structure(fn.structured_outputs),
                }
        if savedmodel_info:
            result["discoveries"]["savedmodel_signatures"] = savedmodel_info

    # 2) Already a ConcreteFunction-like object.
    if (
        hasattr(obj, "structured_input_signature")
        and hasattr(obj, "structured_outputs")
    ):
        result["discoveries"]["concrete_function"] = {
            "structured_input_signature": _summarize_input_signature(
                obj.structured_input_signature
            ),
            "structured_outputs": _summarize_structure(obj.structured_outputs),
            "function_type": str(getattr(obj, "function_type", None)),
        }

    # 3) Keras symbolic model signature if present.
    try:
        import tensorflow as tf  # noqa: F401
        if isinstance(obj, tf.keras.Model):
            sym = _discover_symbolic_keras_signature(obj)
            if sym is not None:
                result["discoveries"]["keras_symbolic"] = sym
    except Exception:
        pass

    # 4) Existing traced signatures on tf.function-like objects.
    pp = getattr(obj, "pretty_printed_concrete_signatures", None)
    if callable(pp):
        try:
            pretty = pp()
            if pretty:
                result["discoveries"]["cached_concrete_signatures"] = {
                    "pretty_printed": pretty
                }
        except Exception:
            pass

    # 5) If example inputs/specs were supplied, try tracing.
    if trace and callable(obj) and (example_args or example_kwargs):
        try:
            result["discoveries"]["traced"] = _discover_concrete_from_callable(obj)
        except Exception as e:
            result["trace_error"] = f"{type(e).__name__}: {e}"

    # 6) Add guidance if nothing TensorFlow-level was found.
    if not result["discoveries"]:
        result["notes"].append(
            "No TensorFlow tensor signature could be discovered from the object as-is."
        )
        if callable(obj):
            result["notes"].append(
                "Pass example_args / example_kwargs as TensorSpecs, tensors, arrays, or scalars to force tracing."
            )

    return result
