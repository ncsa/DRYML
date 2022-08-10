"""
Utility functions for data methods
"""


import numpy as np


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

    return _renester(shape_data)


def nested_apply(data, func_lambda, *func_args, **func_kwargs):
    flattened_data = nested_flatten(data)
    flattened_data = list(map(
        lambda el: func_lambda(el, *func_args, **func_kwargs),
        flattened_data))

    return renest_flat(data, flattened_data)


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


def nested_batcher(data_gen, batch_size):
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
        # if we have a non-empty batch, yield it.
        if flat_batch_data is not None and \
                len(flat_batch_data) > 0:
            flat_batch_data = list(map(
                lambda e: np.stack(e, axis=0),
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
