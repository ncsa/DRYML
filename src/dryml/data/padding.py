"""Explicit dense padding and geometry-derived masks for selected tensor Methods.

NumPy/TensorFlow accept native ragged values; Torch/JAX consume dense inputs.
Optional frameworks are imported only by their selected execution paths.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec
from dryml.methods import Method, traits


class Padding(Method):
    """Pad sample axes and optionally return a validity mask from input geometry.

    Args:
        length: Nonnegative target size, None for the input maximum, or a tuple
            matching axis. A scalar target is shared by all selected axes.
        axis: Integer sample axis or nonempty tuple of sample axes. Negative
            indices are accepted. The logical batch axis is never selected.
        value: Python scalar fill value, converted by the native backend while
            preserving the input dtype.
        side: 'right' appends padding; 'left' prepends it, on every selected axis.
        return_mask: Return {'values': dense tensor, 'mask': bool tensor} when
            true, otherwise just the tensor. True marks original retained data.
        truncate: Admit oversized inputs by removing the right tail (right
            padding) or left head (left padding). False raises instead.

    Targets are resolved from the whole input before truncation. Masks contain
    the batch axis, if present, and the sample prefix through the last selected
    axis; unchanged trailing feature axes are omitted. No comparison with the
    padding value is used. Empty rows and zero-sized feature axes retain validity
    semantics independently of numeric storage.

    Dense NumPy, TensorFlow, Torch and JAX values are supported. NumPy and
    TensorFlow ragged inputs become dense; every variable ragged sample axis must
    be selected. Fixed uniform ragged axes may remain unselected. Use known-spec
    preparation to establish batch interpretation and reusable Dynamic contracts.

    Raises:
        TypeError: For malformed options, unsupported layouts or non-tensor specs.
        ValueError: For invalid axes/targets, unresolved ragged axes, or oversized
            inputs without truncation. Native TensorFlow assertion/cast errors
            propagate during graph execution; native allocation errors propagate.

    Side Effects:
        Allocates padded results and, when requested, masks; never changes input
        buffers. Pure output inference opens no sources or optional frameworks.
        Native compilation may specialize to concrete shapes independently of IR.
    """

    def __init__(self, length=None, *, axis=0, value=0, side="right", return_mask=False, truncate=False):
        axes = (axis,) if type(axis) is int else tuple(axis) if isinstance(axis, (tuple, list)) else ()
        if not axes or any(type(item) is not int for item in axes):
            raise TypeError("Padding axis must be an integer or nonempty integer sequence.")
        if len(set(axes)) != len(axes):
            raise ValueError("Padding axes must be distinct.")
        lengths = tuple(length) if isinstance(length, (tuple, list)) else (length,) * len(axes)
        if len(lengths) != len(axes):
            raise ValueError("Padding length and axis sequences must have equal length.")
        if any(item is not None and (type(item) is not int or item < 0) for item in lengths):
            raise ValueError("Padding lengths must be nonnegative integers or None.")
        if type(side) is not str or side not in {"left", "right"}:
            raise ValueError("Padding side must be 'left' or 'right'.")
        if type(return_mask) is not bool or type(truncate) is not bool:
            raise TypeError("Padding return_mask and truncate must be exact bools.")
        if type(value) not in (bool, int, float, complex, str, bytes):
            raise TypeError("Padding value must be a Python scalar.")
        self.axis, self.length = axes, lengths
        self.value, self.side = value, side
        self.return_mask, self.truncate = return_mask, truncate

    @property
    def iteration_independent(self) -> bool:
        """Return true: discarded Padding calls have no retained state effects."""
        return True

    def _axes(self, rank, batched):
        sample_rank = rank - int(batched)
        axes = tuple(axis + sample_rank if axis < 0 else axis for axis in self.axis)
        if any(axis < 0 or axis >= sample_rank for axis in axes):
            raise ValueError("Padding axis is outside the sample rank.")
        if len(set(axes)) != len(axes):
            raise ValueError("Padding axes alias after resolving negative indices.")
        return tuple(axis + int(batched) for axis in axes)

    def infer_output_spec(self, input_spec):
        """Infer dense values and optional boolean-mask specs without execution.

        Args:
            input_spec: One dense or ragged TensorSpec of known sample rank.
                Use Select/Project to pad fields of a structured input separately.

        Returns:
            A same-dtype/backend/batch dense spec, or a values/mask spec dict.
            Fixed targets fix dimensions; maximum targets preserve declared sizes
            or Dynamic. Mask names follow its prefix of the sample axis names.

        Raises:
            TypeError: For non-tensor, Python-object or unsupported-layout input.
            ValueError: For unknown rank, invalid axes, incomplete ragged metadata,
                unselected variable ragged axes or provable oversize conflicts.

        Side Effects:
            None. No values, sources or optional backend modules are evaluated.
        """
        if not isinstance(input_spec, TensorSpec) or input_spec.layout not in (Layout.DENSE, Layout.RAGGED):
            raise TypeError("Padding requires one dense or ragged TensorSpec; select structured fields explicitly.")
        if input_spec.shape is None:
            raise ValueError("Padding requires a known sample rank.")
        axes = self._axes(input_spec.full_rank, input_spec.batched)
        offset = int(input_spec.batched)
        if input_spec.layout is Layout.RAGGED:
            if input_spec.backend is not None and input_spec.backend.value not in {"numpy", "tf"}:
                raise TypeError("Padding accepts ragged values only from NumPy or TensorFlow.")
            if input_spec.ragged_rank is None:
                raise ValueError("Padding requires a known ragged_rank.")
            for physical_axis in range(1, input_spec.ragged_rank + 1):
                if physical_axis not in axes and input_spec.full_shape[physical_axis] is Dynamic:
                    raise ValueError("Padding must select every variable ragged sample axis.")
        shape = list(input_spec.shape)
        for physical_axis, target in zip(axes, self.length):
            axis = physical_axis - offset
            if target is not None:
                if not self.truncate and shape[axis] is not Dynamic and shape[axis] > target:
                    raise ValueError("Padding target is smaller than a declared input dimension; enable truncate explicitly.")
                shape[axis] = target
        values = replace(input_spec, shape=tuple(shape), layout=Layout.DENSE,
                         ragged_rank=None, row_splits_dtype=None, sparse_format=None)
        if not self.return_mask:
            return values
        prefix = max(axes) + 1 - offset
        names = None if values.axis_names is None else values.axis_names[:prefix]
        mask = replace(values, dtype="bool", shape=values.shape[:prefix], axis_names=names)
        return {"values": values, "mask": mask}

    def _targets(self, shape, axes):
        targets = list(shape)
        for axis, target in zip(axes, self.length):
            if target is not None:
                if not self.truncate and shape[axis] > target:
                    raise ValueError("Padding target is smaller than input geometry; enable truncate explicitly.")
                targets[axis] = target
        return tuple(targets)

    def _dense(self, x, *, batched, backend):
        axes = self._axes(x.ndim, batched)
        targets = self._targets(tuple(x.shape), axes)
        slices, pads = [], []
        for size, target in zip(x.shape, targets):
            start = max(size - target, 0) if self.side == "left" else 0
            slices.append(slice(start, start + min(size, target)))
            amount = max(target - size, 0)
            pads.append((amount, 0) if self.side == "left" else (0, amount))
        if backend == "torch":
            import torch
            import torch.nn.functional as functional
            def pad(value, widths, fill):
                if any(size == 0 for size in value.shape):
                    # ConstantPadND rejects zero feature axes even when they are unpadded.
                    shape = tuple(size + before + after for size, (before, after) in zip(value.shape, widths))
                    result = torch.full(shape, fill, dtype=value.dtype, device=value.device)
                    result[tuple(slice(before, before + size) for size, (before, _) in zip(value.shape, widths))] = value
                    return result
                return functional.pad(value, tuple(item for pair in reversed(widths) for item in pair), value=fill)
            def ones(shape):
                return torch.ones(shape, device=x.device, dtype=torch.bool)
        else:
            if backend == "jax":
                import jax.numpy as module
            else:
                import numpy as module
            def pad(value, widths, fill):
                return module.pad(value, widths, mode="constant", constant_values=fill)
            def ones(shape):
                return module.ones(shape, dtype=bool)
        values = pad(x[tuple(slices)], pads, self.value)
        if not self.return_mask:
            return values
        prefix = max(axes) + 1
        mask = pad(ones(x.shape[:prefix])[tuple(slices[:prefix])], pads[:prefix], False)
        return {"values": values, "mask": mask}

    def _numpy(self, x, *, batched):
        import numpy as np
        from dryml.numpy.ragged import RaggedTensor

        if not isinstance(x, RaggedTensor):
            return self._dense(x, batched=batched, backend="numpy")
        rank, rr = len(x.shape), x.ragged_rank
        axes = self._axes(rank, batched)
        parts = x.nested_row_splits
        lengths = [np.diff(part) for part in parts]
        shape = list(x.shape)
        for axis in range(1, rr + 1):
            if shape[axis] is None:
                if axis not in axes and len(lengths[axis - 1]):
                    raise ValueError("Padding must select every variable ragged sample axis.")
                shape[axis] = int(lengths[axis - 1].max(initial=0))
        targets = self._targets(shape, axes)
        prefix = max(axes) + 1

        def coordinates(count):
            coords, sizes = [np.arange(len(x), dtype="int64")], [np.full(len(x), len(x), dtype="int64")]
            for part, lens in zip(parts[:count - 1], lengths):
                coords = [np.repeat(coord, lens) for coord in coords]
                sizes = [np.repeat(size, lens) for size in sizes]
                coords.append(np.arange(int(part[-1]), dtype="int64") - np.repeat(part[:-1], lens))
                sizes.append(np.repeat(lens, lens))
            if count > rr + 1:
                tail = tuple(shape[rr + 1:count])
                grid = np.indices(tail, dtype="int64").reshape(len(tail), prod(tail)).T
                population = len(coords[0])
                coords = [np.repeat(coord, prod(tail)) for coord in coords]
                sizes = [np.repeat(size, prod(tail)) for size in sizes]
                for index, width in enumerate(tail):
                    coords.append(np.tile(grid[:, index], population))
                    sizes.append(np.full(population * prod(tail), width, dtype="int64"))
            keep = np.ones(len(coords[0]), dtype=bool)
            for axis in axes:
                if self.side == "left":
                    coords[axis] = coords[axis] + targets[axis] - sizes[axis]
                keep &= (coords[axis] >= 0) & (coords[axis] < targets[axis])
            return tuple(coord[keep] for coord in coords), keep

        value_prefix = max(rr + 1, prefix)
        coords, keep = coordinates(value_prefix)
        blocks = x.flat_values.reshape((len(keep), *shape[value_prefix:]))
        values = np.full(targets, self.value, dtype=x.dtype)
        values[coords] = blocks[keep]
        if not self.return_mask:
            return values
        if prefix != value_prefix:
            coords, _ = coordinates(prefix)
        mask = np.zeros(targets[:prefix], dtype=bool)
        mask[coords] = True
        return {"values": values, "mask": mask}

    def _tensorflow(self, x, *, batched):
        import tensorflow as tf

        rank = x.shape.rank
        if rank is None:
            raise ValueError("Padding requires known TensorFlow rank.")
        axes = self._axes(rank, batched)
        ragged = isinstance(x, tf.RaggedTensor)
        rr = x.ragged_rank if ragged else 0
        parts = tuple(tf.cast(part, tf.int64) for part in x.nested_row_splits) if ragged else ()
        lengths = tuple(part[1:] - part[:-1] for part in parts)
        static = x.shape.as_list()
        checks = []
        if ragged:
            shape = [tf.cast(x.nrows(), tf.int64)]
            for axis, lens in enumerate(lengths, start=1):
                maximum = tf.reduce_max(tf.concat((lens, tf.zeros(1, tf.int64)), axis=0))
                size = tf.constant(static[axis], tf.int64) if static[axis] is not None else maximum
                if axis not in axes:
                    checks.append(tf.debugging.assert_equal(tf.reduce_all(lens == size), True,
                                                           message="Select every variable ragged axis for Padding"))
                shape.append(size)
            shape.extend(tf.unstack(tf.shape(x.flat_values, out_type=tf.int64)[1:], num=rank - rr - 1))
        else:
            shape = tf.unstack(tf.shape(x, out_type=tf.int64), num=rank)
        targets = list(shape)
        for axis, target in zip(axes, self.length):
            if target is not None:
                targets[axis] = tf.constant(target, tf.int64)
                static[axis] = target
                if not self.truncate:
                    checks.append(tf.debugging.assert_less_equal(shape[axis], targets[axis],
                                                               message="Padding target too small; enable truncate explicitly"))
                    # Shape enforcement must fail too if a compiler drops the assertion.
                    targets[axis] = tf.maximum(shape[axis], targets[axis])
        prefix = max(axes) + 1
        with tf.control_dependencies([check for check in checks if check is not None]):
            if not ragged:
                begin = [tf.maximum(size - target, 0) if self.side == "left" else tf.constant(0, tf.int64)
                         for size, target in zip(shape, targets)]
                sizes = [tf.minimum(size, target) for size, target in zip(shape, targets)]
                pads = [(tf.maximum(target - size, 0), 0) if self.side == "left"
                        else (0, tf.maximum(target - size, 0)) for size, target in zip(shape, targets)]
                values = tf.pad(tf.slice(x, begin, sizes), pads, constant_values=tf.cast(self.value, x.dtype))
                if self.return_mask:
                    mask = tf.pad(tf.slice(tf.ones(shape[:prefix], tf.bool), begin[:prefix], sizes[:prefix]),
                                  pads[:prefix], constant_values=False)
            else:
                def coordinates(count):
                    outer = shape[0]
                    coords, sizes = [tf.range(outer, dtype=tf.int64)], [tf.fill([outer], outer)]
                    for part, lens in zip(parts[:count - 1], lengths):
                        coords = [tf.repeat(coord, lens) for coord in coords]
                        sizes = [tf.repeat(size, lens) for size in sizes]
                        coords.append(tf.range(part[-1], dtype=tf.int64) - tf.repeat(part[:-1], lens))
                        sizes.append(tf.repeat(lens, lens))
                    if count > rr + 1:
                        tail = shape[rr + 1:count]
                        population = tf.size(coords[0], out_type=tf.int64)
                        factor = tf.reduce_prod(tf.stack(tail))
                        grid = tf.stack(tf.meshgrid(*(tf.range(size, dtype=tf.int64) for size in tail), indexing="ij"), axis=-1)
                        grid = tf.reshape(grid, [factor, len(tail)])
                        coords = [tf.repeat(coord, factor) for coord in coords]
                        sizes = [tf.repeat(size, factor) for size in sizes]
                        for index, width in enumerate(tail):
                            coords.append(tf.tile(grid[:, index], [population]))
                            sizes.append(tf.fill([population * factor], width))
                    keep = tf.ones(tf.shape(coords[0]), dtype=tf.bool)
                    for axis in axes:
                        if self.side == "left":
                            coords[axis] = coords[axis] + targets[axis] - sizes[axis]
                        keep &= (coords[axis] >= 0) & (coords[axis] < targets[axis])
                    return tf.boolean_mask(tf.stack(coords, axis=1), keep), keep

                value_prefix = max(rr + 1, prefix)
                coords, keep = coordinates(value_prefix)
                blocks = tf.reshape(x.flat_values, [tf.size(keep, out_type=tf.int64), *shape[value_prefix:]])
                values = tf.tensor_scatter_nd_update(tf.fill(targets, tf.cast(self.value, x.dtype)), coords,
                                                     tf.boolean_mask(blocks, keep))
                if self.return_mask:
                    if prefix != value_prefix:
                        coords, _ = coordinates(prefix)
                    mask = tf.tensor_scatter_nd_update(tf.zeros(targets[:prefix], tf.bool), coords,
                                                      tf.ones([tf.shape(coords)[0]], tf.bool))
        values = tf.ensure_shape(values, static)
        if not self.return_mask:
            return values
        return {"values": values, "mask": tf.ensure_shape(mask, static[:prefix])}

    @traits(backend="numpy", batch_mode="element")
    def _numpy_element(self, x):
        return self._numpy(x, batched=False)

    @traits(backend="numpy", batch_mode="batched")
    def _numpy_batched(self, x):
        return self._numpy(x, batched=True)

    @traits(backend="tf", batch_mode="element")
    def _tf_element(self, x):
        return self._tensorflow(x, batched=False)

    @traits(backend="tf", batch_mode="batched")
    def _tf_batched(self, x):
        return self._tensorflow(x, batched=True)

    @traits(backend="torch", batch_mode="element")
    def _torch_element(self, x):
        return self._dense(x, batched=False, backend="torch")

    @traits(backend="torch", batch_mode="batched")
    def _torch_batched(self, x):
        return self._dense(x, batched=True, backend="torch")

    @traits(backend="jax", batch_mode="element")
    def _jax_element(self, x):
        return self._dense(x, batched=False, backend="jax")

    @traits(backend="jax", batch_mode="batched")
    def _jax_batched(self, x):
        return self._dense(x, batched=True, backend="jax")
