"""
Utility functions for data methods
"""

import inspect
from typing import Callable

import numpy as np

from dryml.core2.tensor_spec import Dynamic, TensorSpec, iter_specs, unbatch_spec_tree
from dryml.core2.utils.recurse import map_leaves
from dryml.data.collate import default_collate
from dryml.data.transforms import Select


def _path_select(path):
    if isinstance(path, (tuple, list)):
        return Select(*path)
    return Select(path)


def spec_tree_is_batched(spec_tree) -> bool:
    batches = {spec.batched for spec in iter_specs(spec_tree)}
    return len(batches) == 1 and batches.pop()


def batch_from_spec_tree(spec_tree):
    batches = {spec.batch for spec in iter_specs(spec_tree) if spec.batched}
    if not batches:
        return None
    return batches.pop() if len(batches) == 1 else Dynamic


def sample_from_spec_tree(spec_tree):
    def leaf_sample(spec: TensorSpec):
        if spec.shape is None:
            raise ValueError("Cannot sample from an unknown-rank TensorSpec.")
        sample_shape = tuple(1 if dim is Dynamic else int(dim) for dim in spec.shape)
        shape = spec.full_shape if spec.batched else (1, *sample_shape)
        if spec.batched:
            shape = tuple(1 if dim is Dynamic else int(dim) for dim in shape)
        return np.zeros(shape, dtype=spec.dtype.np())

    return map_leaves(spec_tree, leaf_sample, pred=lambda x: isinstance(x, TensorSpec))


def maybe_unbatch_output_spec(output_spec, input_spec):
    if spec_tree_is_batched(input_spec):
        return output_spec
    return unbatch_spec_tree(output_spec)


def match_input_batch(spec: TensorSpec, input_spec):
    batch = batch_from_spec_tree(input_spec)
    if batch is None:
        return spec
    return spec.with_batch(batch=batch)


def iter_xy(dataset, *, x_path=0, y_path=1):
    x_select = _path_select(x_path)
    y_select = _path_select(y_path)

    it = iter(dataset)
    try:
        first = next(it)
    except StopIteration:
        return

    x_impl, first_x = x_select.bind_first(first, input_spec=dataset.spec)
    y_impl, first_y = y_select.bind_first(first, input_spec=dataset.spec)
    yield first_x, first_y

    for item in it:
        yield x_impl(item), y_impl(item)


def collect_xy(dataset, *, x_path=0, y_path=1):
    x_values = []
    y_values = []

    for x, y in iter_xy(dataset, x_path=x_path, y_path=y_path):
        x_values.append(x)
        y_values.append(y)

    if not x_values:
        raise ValueError("Cannot collect from an empty dataset.")
    return x_values, y_values


def collate_xy(dataset, *, x_path=0, y_path=1, collate=default_collate):
    x_values, y_values = collect_xy(dataset, x_path=x_path, y_path=y_path)
    return collate(x_values), collate(y_values), len(x_values)


def nested_flatten(data):
    flatten_data = []

    def _nested_flatten(data):
        if type(data) is dict:
            for key in data:
                _nested_flatten(data[key])
        elif type(data) is tuple:
            for el in data:
                _nested_flatten(el)
        else:
            flatten_data.append(data)

    _nested_flatten(data)
    return flatten_data


def renest_flat(shape_data, flat_data):
    def _renester(data):
        if type(data) is dict:
            res = {}
            for key in data:
                res[key] = _renester(data[key])
            return res
        elif type(data) is tuple:
            res = []
            for el in data:
                res.append(_renester(el))
            return tuple(res)
        else:
            return flat_data.pop(0)

    res = _renester(shape_data)

    return res


def nested_apply(data, func_lambda, *func_args, **func_kwargs):
    flattened_data = nested_flatten(data)
    flattened_data = list(map(
        lambda el: func_lambda(el, *func_args, **func_kwargs),
        flattened_data))
    return renest_flat(data, flattened_data)


def nestize(f, *func_args, **func_kwargs):
    return lambda el: nested_apply(el, f, *func_args, **func_kwargs)


def nested_slice(data, slicer):
    def _slicer(el):
        return el[slicer]
    return nested_apply(data, _slicer)


def get_data_batch_size(full_data=None, flat_data=None):
    if full_data is None and flat_data is None:
        raise ValueError(
            "At least one of full_data or flat_data needs not be None.")
    if full_data is not None and flat_data is not None:
        raise ValueError("Can't specify both full_data and flat_data.")
    if full_data is not None:
        flat_data = nested_flatten(full_data)
    lengths = set(map(lambda el: len(el), flat_data))
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent element sizes: {lengths}")
    return lengths.pop()


def nested_batcher(data_gen, batch_size, stack_method, drop_remainder=True):
    it = iter(data_gen())
    while True:
        flat_batch_data = None
        flat_batch_shape = None
        num_collected = 0
        try:
            # Fill up batches
            while True:
                el = next(it)
                el_flat = nested_flatten(el)
                if flat_batch_data is None:
                    flat_batch_shape = el
                    flat_batch_data = list(
                        map(lambda e: list(),
                            el_flat))
                for i in range(len(el_flat)):
                    flat_batch_data[i].append(el_flat[i])
                num_collected += 1
                if num_collected >= batch_size:
                    break
        except StopIteration:
            # Catch stop iteration for partial batches
            pass
        if drop_remainder and num_collected != batch_size:
            # Exit now and don't yield
            break
        # if we have a non-empty batch, yield it.
        if flat_batch_data is not None and \
                len(flat_batch_data) > 0:
            flat_batch_data = list(map(
                stack_method,
                flat_batch_data))
            yield renest_flat(
                flat_batch_shape,
                flat_batch_data)
        else:
            break


def nested_unbatcher(data_gen):
    it = iter(data_gen())
    while True:
        try:
            d = next(it)
        except StopIteration:
            return
        flat_d = nested_flatten(d)
        length = get_data_batch_size(flat_data=flat_d)
        for i in range(length):
            new_d = list(map(lambda el: el[i], flat_d))
            yield renest_flat(d, new_d)


def taker(gen_func, n):
    i = 0
    it = iter(gen_func())
    while i < n:
        try:
            yield next(it)
            i += 1
        except StopIteration:
            return
    return


def skiper(gen_func, n):
    i = 0
    it = iter(gen_func())
    while i < n:
        try:
            next(it)
            i += 1
        except StopIteration:
            return
    while True:
        try:
            yield next(it)
        except StopIteration:
            return


def function_inspection(func: Callable):
    if not callable(func):
        raise ValueError("Argument should be a function.")

    sig = inspect.signature(func)
    params = sig.parameters

    explicit_args = 0
    var_args = False
    keyword_args = 0
    var_kwargs = False

    for _, param in params.items():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            if param.default == inspect.Parameter.empty:
                explicit_args += 1
            else:
                keyword_args += 1
        elif param.kind == param.VAR_POSITIONAL:
            var_args = True
        elif param.kind == param.VAR_KEYWORD:
            var_kwargs = True

    return {
        'signature': sig,
        'n_args': explicit_args,
        'var_args': var_args,
        'n_kwargs': keyword_args,
        'var_kwargs': var_kwargs,
    }


def promote_function(func):
    def promoted_func(x, y, *args, **kwargs):
        return func(x, *args, **kwargs), \
               func(y, *args, **kwargs)
    return promoted_func


def func_source_extract(func):
    # Get source code for a given function,
    # and format it in a consistent way for
    # building custom transformations.
    #
    # Args:
    #   func: The function whose source to extract.

    # Get the source code
    code_string = inspect.getsource(func)

    # Possibly strip leading spaces
    code_lines = code_string.split('\n')

    init_strip = code_lines[0]

    for line in code_lines[1:]:
        i = 0
        while i < len(init_strip) and \
                i < len(line) and \
                init_strip[i] == line[i]:
            i += 1
        if i == len(init_strip) or \
                i == len(line):
            continue

        init_strip = init_strip[:i]

    if len(init_strip) > 0:
        code_lines = list(map(
            lambda line: line[len(init_strip):],
            code_lines))

    return '\n'.join(code_lines)
