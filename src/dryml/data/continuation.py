"""Closed, bounded continuation values and lazy native-buffer reconstruction.

The codec accepts only explicitly supported values, never pickle, live Objects,
callables, generators, or frames. Decoding native values is host-only; executable
backend imports belong exclusively to explicit reattachment. Array payload bytes
are detached at capture, not on ordinary delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import TypeAlias, Union

import numpy as np

from dryml.formats.canonical import canonical_json_bytes, canonical_json_load_bytes, json_ready
from .errors import ContinuationFormatError, UnsupportedContinuationError

ContinuationValue: TypeAlias = Union[
    None, bool, int, float, complex, str, bytes, np.generic, np.ndarray,
    list["ContinuationValue"], tuple["ContinuationValue", ...], dict[str, "ContinuationValue"],
]

_BOUNDS = dict(max_depth=96, max_nodes=500000, max_entries=100000,
               max_string=16 * 1024 * 1024, max_int_bits=4096)
_MAX_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HostBuffer:
    """Inert native-buffer representation; reading it imports no native backend.

    Fields name the registered backend, dtype, shape, layout/placement, and owned
    host bytes. Creation/decoding validates these facts; reconstruction is explicit
    through rebuild_native(). This value contains no live device or resource.
    """

    backend: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    placement: str
    data: bytes

    def __post_init__(self) -> None:
        """Validate host fields without imports or native allocations.

        Raises ContinuationFormatError for unsupported dtype, shape, layout,
        placement or byte fields. Reading the resulting value is backend-inert.
        """
        if (self.backend not in {"torch", "tensorflow", "jax"} or self.layout != "C"
                or type(self.placement) is not str or not self.placement or type(self.data) is not bytes
                or type(self.dtype) is not str or type(self.shape) is not tuple):
            raise ContinuationFormatError("Invalid native host-buffer representation.")
        _dtype_shape(self.dtype, self.shape, len(self.data))


def _dtype_shape(dtype, shape, size):
    try:
        dt = np.dtype(dtype)
        if type(shape) not in (tuple, list) or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid shape")
        if dt.hasobject or dt.fields is not None or dt.metadata or dt.kind not in "biufcSUmM":
            raise ValueError("unsupported storage dtype")
        if dt.itemsize <= 0 or math.prod(shape) * dt.itemsize != size or size > _MAX_BYTES:
            raise ValueError("invalid payload length")
        return dt
    except (ValueError, TypeError, OverflowError) as error:
        raise ContinuationFormatError("Invalid continuation array dtype/shape/bytes.") from error


def _native_host(value):
    module = type(value).__module__.split(".")[0]
    if module == "torch":
        if not isinstance(value, getattr(sys.modules.get("torch"), "Tensor", ())):
            return None
        array = value.detach().cpu().numpy()
        return HostBuffer("torch", array.dtype.str, array.shape, "C", str(value.device), array.tobytes())
    if module == "tensorflow":
        if not isinstance(value, getattr(sys.modules.get("tensorflow"), "Tensor", ())):
            return None
        array = np.asarray(value.numpy())
        return HostBuffer("tensorflow", array.dtype.str, array.shape, "C", str(value.device), array.tobytes())
    if module in {"jax", "jaxlib"}:
        if not isinstance(value, getattr(sys.modules.get("jax"), "Array", ())):
            return None
        array = np.asarray(value)
        return HostBuffer("jax", array.dtype.str, array.shape, "C", str(value.device), array.tobytes())
    return None


def encode_value(value: ContinuationValue) -> tuple[bytes, tuple[bytes, ...]]:
    """Detach a bounded value tree into closed metadata and immutable payloads.

    Args:
        value: Supported scalars/containers, NumPy leaves, or registered native
            buffers. String-keyed dicts and repeated aliases are supported; cycles
            and opaque subclasses are rejected.
    Returns:
        Metadata bytes and owned payload-byte tuple, preserving repeated aliases.
    Raises:
        UnsupportedContinuationError: Unsupported/cyclic/oversized live value.
    Side Effects:
        Copies needed bytes once; native leaves may synchronize only on this call.
    """
    payloads, memo, active = [], {}, set()
    count = [0]
    total_bytes = [0]

    def encode(item, depth=0):
        count[0] += 1
        if count[0] > 100000 or depth > 48:
            raise UnsupportedContinuationError("Continuation value exceeds bounds.")
        typ = type(item)
        if item is None:
            return ["none"]
        if typ in (bool, int, str):
            if typ is int and item.bit_length() > 4096:
                raise UnsupportedContinuationError("Continuation integer exceeds bounds.")
            return [typ.__name__, item]
        if typ is float:
            return ["float", item.hex()]
        if typ is complex:
            return ["complex", item.real.hex(), item.imag.hex()]
        if id(item) in active:
            raise UnsupportedContinuationError("Cyclic continuation value.")
        if id(item) in memo:
            return ["ref", memo[id(item)]]
        index = len(memo)
        memo[id(item)] = index
        active.add(id(item))
        try:
            if typ is bytes:
                tag, attributes, raw = "bytes", [], item
            elif typ is np.ndarray or isinstance(item, np.generic):
                array = np.asarray(item)
                _dtype_shape(array.dtype, array.shape, array.nbytes)
                raw = array.tobytes(order="C")
                tag = "array" if typ is np.ndarray else "scalar"
                attributes = [array.dtype.str, list(array.shape)]
            elif typ in (list, tuple):
                return [typ.__name__, index, [encode(v, depth + 1) for v in item]]
            elif typ is dict:
                if any(type(key) is not str for key in item):
                    raise UnsupportedContinuationError("Continuation mappings require string keys.")
                return ["dict", index, [[key, encode(v, depth + 1)] for key, v in item.items()]]
            else:
                try:
                    host = item if typ is HostBuffer else _native_host(item)
                except (AttributeError, TypeError, ValueError) as error:
                    raise UnsupportedContinuationError("Unsupported native continuation buffer.") from error
                if host is None:
                    raise UnsupportedContinuationError(f"Unsupported continuation value: {typ.__name__}.")
                _dtype_shape(host.dtype, host.shape, len(host.data))
                if host.backend not in {"torch", "tensorflow", "jax"} or host.layout != "C":
                    raise UnsupportedContinuationError("Unsupported native buffer representation.")
                tag, attributes, raw = "native", [host.backend, host.dtype, list(host.shape), host.layout, host.placement], host.data
            total_bytes[0] += len(raw)
            if total_bytes[0] > _MAX_BYTES:
                raise UnsupportedContinuationError("Continuation payload exceeds bounds.")
            payload = len(payloads)
            payloads.append(raw)
            return [tag, index, *attributes, payload]
        finally:
            active.remove(id(item))
    try:
        metadata = canonical_json_bytes(encode(value), **_BOUNDS)
        if len(metadata) > 16 * 1024 * 1024 or len(metadata) + total_bytes[0] > _MAX_BYTES:
            raise UnsupportedContinuationError("Continuation metadata exceeds bounds.")
        return metadata, tuple(payloads)
    except (ContinuationFormatError, ValueError) as error:
        raise UnsupportedContinuationError(str(error)) from error


def decode_value(metadata: bytes, payloads: tuple[bytes, ...]) -> ContinuationValue:
    """Validate/decode codec bytes without importing or rebuilding native tensors.

    Args:
        metadata: Closed tagged metadata from encode_value().
        payloads: Ordered immutable byte payloads with total size at most 1 GiB.
    Returns:
        Fresh independently mutable containers/NumPy buffers; native leaves remain
        inert HostBuffers. Repeated aliases within this decode are preserved.
    Raises:
        ContinuationFormatError: Malformed tags, references, bounds, or payloads.
    Side Effects:
        Allocates host values only; no imports of user targets or optional backends.
    """
    memo, used, pending = {}, set(), set()
    count = [0]
    try:
        if type(metadata) is not bytes or len(metadata) > 16 * 1024 * 1024:
            raise ValueError("invalid metadata length")
        if type(payloads) is not tuple or any(type(p) is not bytes for p in payloads) or sum(map(len, payloads)) > _MAX_BYTES:
            raise ValueError("invalid payloads")
        root = json_ready(canonical_json_load_bytes(metadata, **_BOUNDS), **_BOUNDS)

        def decode(node, depth=0):
            count[0] += 1
            if count[0] > 100000 or depth > 48:
                raise ValueError("continuation value exceeds bounds")
            if type(node) is not list or not node or type(node[0]) is not str:
                raise ValueError("invalid tag")
            tag = node[0]
            if tag == "none" and len(node) == 1:
                return None
            if tag in {"bool", "int", "str"} and len(node) == 2 and type(node[1]) is {"bool": bool, "int": int, "str": str}[tag]:
                return node[1]
            if tag == "float" and len(node) == 2 and type(node[1]) is str:
                return float.fromhex(node[1])
            if tag == "complex" and len(node) == 3 and all(type(v) is str for v in node[1:]):
                return complex(float.fromhex(node[1]), float.fromhex(node[2]))
            if tag == "ref" and len(node) == 2 and type(node[1]) is int and node[1] in memo and node[1] not in pending:
                return memo[node[1]]
            if len(node) < 3 or type(node[1]) is not int or node[1] != len(memo):
                raise ValueError("invalid alias")
            index = node[1]
            memo[index] = None
            pending.add(index)
            if tag in {"list", "tuple", "dict"} and len(node) == 3 and type(node[2]) is list:
                if tag == "dict":
                    value = {}
                    for pair in node[2]:
                        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in value:
                            raise ValueError("invalid mapping")
                        value[pair[0]] = decode(pair[1], depth + 1)
                else:
                    value = [decode(v, depth + 1) for v in node[2]]
                    if tag == "tuple":
                        value = tuple(value)
            elif tag in {"bytes", "array", "scalar", "native"} and len(node) == {"bytes": 3, "array": 5, "scalar": 5, "native": 8}[tag]:
                p = node[-1]
                if type(p) is not int or not 0 <= p < len(payloads) or p in used:
                    raise ValueError("invalid payload index")
                used.add(p)
                raw = payloads[p]
                if tag == "bytes":
                    value = raw
                elif tag == "native":
                    backend, dt, shape, layout, placement = node[2:7]
                    _dtype_shape(dt, shape, len(raw))
                    if backend not in {"torch", "tensorflow", "jax"} or layout != "C" or type(placement) is not str:
                        raise ValueError("invalid native representation")
                    value = HostBuffer(backend, dt, tuple(shape), layout, placement, raw)
                else:
                    dt = _dtype_shape(node[2], node[3], len(raw))
                    value = np.frombuffer(raw, dtype=dt).reshape(node[3]).copy()
                    if tag == "scalar":
                        if value.shape != ():
                            raise ValueError("invalid scalar shape")
                        value = value[()]
            else:
                raise ValueError("unknown or malformed tag")
            memo[index] = value
            pending.remove(index)
            return value
        value = decode(root)
        if used != set(range(len(payloads))):
            raise ValueError("unreferenced payload")
        return value
    except ContinuationFormatError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, RecursionError) as error:
        raise ContinuationFormatError("Malformed continuation value.") from error


def rebuild_native(value: ContinuationValue) -> ContinuationValue:
    """Explicitly rebuild registered native leaves at target reattachment.

    Args:
        value: Host-decoded tree, possibly containing HostBuffer leaves.
    Returns:
        Independent tree preserving aliases and native dtype/shape/placement.
    Raises:
        ImportError: Required backend is unavailable.
        ValueError: Saved placement cannot be reconstructed by that backend.
    Side Effects:
        Lazily imports only required backends and allocates their tensors.
    """
    memo = {}
    def rebuild(item):
        if id(item) in memo:
            return memo[id(item)]
        if type(item) is HostBuffer:
            array = np.frombuffer(item.data, dtype=item.dtype).reshape(item.shape).copy()
            if item.backend == "torch":
                import torch
                result = torch.from_numpy(array).to(item.placement)
            elif item.backend == "tensorflow":
                import tensorflow as tf
                with tf.device(item.placement):
                    result = tf.convert_to_tensor(array)
            else:
                import jax
                devices = [d for d in jax.devices() if str(d) == item.placement]
                if len(devices) != 1:
                    raise ValueError("Saved JAX placement is unavailable.")
                result = jax.device_put(array, devices[0])
        elif type(item) is dict:
            result = {key: rebuild(v) for key, v in item.items()}
        elif type(item) in (list, tuple):
            result = type(item)(rebuild(v) for v in item)
        else:
            result = item
        memo[id(item)] = result
        return result
    return rebuild(value)
