"""A mutable-by-replacement NumPy payload with an array-free DRYML definition."""

from __future__ import annotations

import inspect
import os

import numpy as np

from dryml.core import Expr, Serializable
from dryml.core.dtype import DType
from dryml.core.params import Match


_CODEC = "npy1"
_FILENAME = "array.npy"


class NumpyArray(Serializable):
    """Store one NumPy array in Serializable state rather than constructor metadata.

    Construction records ``shape`` and ``dtype`` as searchable configuration and
    initializes an owned zero array. Array values stay in state, not constructor
    metadata. Assign :attr:`array`, then use the ordinary Object/Repo save API to
    obtain an exact SavedNode. Independent instances have independent ObjectIds
    despite equal shape/dtype definitions.

    The container owns a read-only copy of each assigned ndarray. Returned views
    preserve dtype, shape, byte order and values without exposing an ordinary
    writable alias. Replace ``array`` to edit content; successful replacement
    clears the live last-state receipt, not any previously saved state. Assignment
    and save/restore on the same instance must be caller-serialized, including
    keeping input arrays stable while copying. There is no managed compute step.

    The ``npy1`` codec denotes this container's one-file payload contract, not a
    particular NumPy file-header version. State contains exactly ``array.npy``
    with pickle disabled. Arrays must have NPY-lossless nonobject, unstructured
    dtypes without dtype metadata. Loading is eager; saved payloads are independent
    of input arrays, external files, and live container replacements.
    """

    state_codec = _CODEC

    def __init__(self, shape, dtype) -> None:
        """Initialize read-only zeros for the declared shape and NumPy dtype.

        Args:
            shape: Nonnegative integer or list/tuple of nonnegative integers.
                An empty tuple declares a scalar. Normalized to a tuple.
            dtype: NPY-lossless NumPy dtype spelling, dtype object, scalar type,
                or supported canonical DRYML DType. Canonical NumPy spellings are
                retained in definitions, e.g. "float32"; explicit byte order and
                string widths are preserved when necessary.

        Raises:
            TypeError: For unsupported shapes/dtypes, booleans as dimensions, or
                string dtypes without a concrete nonzero width.
            ValueError: For negative dimensions or an invalid dtype.
            MemoryError: If allocating the zero array fails.

        Side Effects:
            Allocates owned read-only zero content but performs no Store IO.
            Definition authoring normalizes metadata without allocating arrays.
            Repo selection is entirely inherited from the Object framework.
        """
        self._shape = _normalize_shape(shape)
        self._dtype = _numpy_dtype(dtype)
        self._install_array(np.zeros(self._shape, dtype=self._dtype))

    @classmethod
    def __dryml_normalize_definition_arguments__(cls, args, kwargs):
        """Canonicalize shape/dtype metadata without constructing array content.

        The existing Object authoring protocol supplies a positional tuple and
        keyword dict. Return the same two forms after normalizing known values;
        template expressions and query matchers remain inert. Invalid concrete
        configuration raises TypeError/ValueError without allocation or IO.
        """
        bound = inspect.signature(cls.__init__).bind_partial(None, *args, **kwargs)
        if "shape" in bound.arguments:
            bound.arguments["shape"] = _normalize_shape(bound.arguments["shape"], symbolic=True)
        if "dtype" in bound.arguments:
            value = bound.arguments["dtype"]
            if not isinstance(value, (Expr, Match)):
                bound.arguments["dtype"] = _dtype_spelling(_numpy_dtype(value))
        return bound.args[1:], dict(bound.kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the immutable declared shape without reading payload bytes."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return the declared NumPy dtype without reading payload bytes."""
        return self._dtype

    @property
    def has_content(self) -> bool:
        """Return whether usable array content is installed, without IO.

        False also covers a Core-invalidated failed restoration. This describes
        live content, not managed completion or availability of a saved state.
        """
        return not self._restore_failed and self._array is not None

    @property
    def array(self) -> np.ndarray:
        """Return a read-only view of the owned ndarray without copying its bytes.

        Raises:
            RuntimeError: If Core invalidated the instance after failed restore;
                materialize a fresh instance from an exact SavedNode instead.

        Side Effects:
            Allocates only a small ndarray view; never reads Store payloads or
            computes missing content. Changes require assignment through the
            setter, which copies and installs new content.
        """
        self._check_valid()
        return self._array.view()

    @array.setter
    def array(self, value: np.ndarray) -> None:
        """Copy and install one ndarray, clearing the live saved-state receipt.

        Args:
            value: NumPy ndarray (including memmaps), with a nonobject,
                unstructured, NPY-lossless dtype and no dtype metadata.
                Array-subclass behavior is not retained; logical array content
                becomes an owned ndarray. Shape and dtype must exactly match the
                constructor declarations; no resizing or implicit cast occurs.

        Raises:
            TypeError: For nonarrays, unsupported or mismatching dtypes. No implicit conversion
                from lists or arbitrary Python objects is attempted.
            ValueError: If the replacement shape differs from its declaration.
            RuntimeError: If Core has invalidated this instance.
            MemoryError: If allocating the owned copy fails.

        Side Effects:
            Copies logical values into read-only owned memory before replacement.
            Validation/allocation failure leaves content and receipts unchanged.
            Successful replacement invalidates live last-state metadata but does
            not edit constructor definitions, saved Nodes, or Store files.
        """
        self._check_valid()
        self._validate_content(value)
        owned = np.array(value, copy=True, order="K", subok=False)
        self._install_array(owned)

    def _check_valid(self) -> None:
        """Respect Core's failed-restore invalidation before exposing live content."""
        if self._restore_failed:
            raise RuntimeError("NumpyArray was invalidated by failed restore; load a fresh saved instance.")

    def _install_array(self, value: np.ndarray) -> None:
        """Install already-owned decoded/copied bytes only after complete validation."""
        self._validate_content(value)
        value.setflags(write=False)
        self._last_state_hash = None
        self._last_state_ref = None
        self._array = value

    def _validate_content(self, value: np.ndarray) -> None:
        """Keep payload shape/dtype consistent with searchable constructor metadata."""
        _validate_array(value)
        if value.shape != self._shape:
            raise ValueError(f"NumpyArray expected shape {self._shape}, got {value.shape}.")
        if value.dtype != self._dtype:
            raise TypeError(f"NumpyArray expected dtype {self._dtype}, got {value.dtype}.")

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write populated content as one non-pickle ``array.npy`` payload.

        Args:
            dest_dir: Framework-owned empty staging directory.
            codec: Must be this container's ``npy1`` codec identifier.

        Raises:
            ValueError: For unsupported codec.
            RuntimeError: If the instance was invalidated by failed restore.
            OSError: If writing fails. Core owns publication/abort of the staging
                directory; no saved state is replaced by this hook.

        Side Effects:
            Writes array bytes only, without modifying live content or recording
            them in the definition. Uses the default eager Core payload inventory.
        """
        _check_codec(codec)
        array = self.array
        with open(os.path.join(dest_dir, _FILENAME), "wb") as stream:
            np.save(stream, array, allow_pickle=False)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Eagerly restore and validate exactly one non-pickle NumPy array.

        Args:
            src_dir: Framework-validated payload directory containing array.npy.
            codec: Must be ``npy1``; alternative codecs are not guessed.

        Raises:
            ValueError: For unsupported codec, malformed/truncated NPY content,
                object-array encoding, trailing bytes, or a shape mismatch.
            TypeError: For an unsupported or mismatching decoded dtype.
            OSError: If reading fails.

        Side Effects:
            Installs owned read-only content after full decode/validation, with
            no extra array copy. Hook failure leaves previous content untouched;
            the enclosing Core restore still owns graph invalidation and receipt
            installation. No external mmap or file lifetime is retained.
        """
        _check_codec(codec)
        with open(os.path.join(src_dir, _FILENAME), "rb") as stream:
            value = np.lib.format.read_array(stream, allow_pickle=False)
            if stream.read(1):
                raise ValueError("NumpyArray payload has trailing bytes.")
        self._install_array(value)


def _check_codec(codec: str) -> None:
    """Reject codec changes before performing payload IO."""
    if codec != _CODEC:
        raise ValueError("NumpyArray supports only the npy1 state codec.")


def _validate_array(value: np.ndarray) -> None:
    """Admit plain NPY-representable array data without invoking object conversion."""
    if not isinstance(value, np.ndarray):
        raise TypeError("NumpyArray content must be a NumPy ndarray.")
    _validate_dtype(value.dtype)


def _validate_dtype(dtype: np.dtype) -> None:
    """Reject dtype information that NPY cannot preserve before array allocation."""
    if dtype.hasobject or dtype.fields is not None or dtype.metadata is not None:
        raise TypeError("NumpyArray requires a nonobject, unstructured dtype without metadata.")
    if dtype.kind in {"S", "U"} and dtype.itemsize == 0:
        raise TypeError("NumpyArray string dtypes require an explicit nonzero width.")
    descriptor = np.lib.format.dtype_to_descr(dtype)
    if np.lib.format.descr_to_dtype(descriptor) != dtype:
        raise TypeError("NumpyArray dtype cannot be preserved by the non-pickle NPY format.")


def _numpy_dtype(value) -> np.dtype:
    """Resolve supported dtype inputs without allocating a numerical array."""
    dtype = np.dtype(value.np() if isinstance(value, DType) else value)
    _validate_dtype(dtype)
    return dtype


def _dtype_spelling(dtype: np.dtype) -> str:
    """Use readable NumPy names where exact, retaining endian/width descriptors otherwise."""
    try:
        named = np.dtype(dtype.name)
    except TypeError:
        return dtype.str
    return dtype.name if named == dtype else dtype.str


def _normalize_shape(value, *, symbolic=False):
    """Normalize concrete dimensions while leaving optional template/matcher leaves inert."""
    if symbolic and isinstance(value, (Expr, Match)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError("NumpyArray shape must be an integer or a list/tuple of integers.")
    dimensions = []
    for dimension in value:
        if symbolic and isinstance(dimension, (Expr, Match)):
            dimensions.append(dimension)
            continue
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("NumpyArray dimensions must be nonnegative integers, not booleans.")
        if dimension < 0:
            raise ValueError("NumpyArray dimensions must be nonnegative.")
        dimensions.append(int(dimension))
    return tuple(dimensions)


__all__ = ["NumpyArray"]
