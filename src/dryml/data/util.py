"""
Utility functions for data methods
"""

import inspect
from typing import Callable

from dryml.core.tensor_spec import iter_specs
from dryml.data.collate import default_collate
from dryml.data.methods import Project, Select


def _xy_dataset(dataset, *, x_path=0, y_path=1):
    from dryml.data.dataset import Map

    return Map(dataset, Project(Select(x_path), Select(y_path)))


def as_supervised(dataset, inputs, targets=None, *, input_as_target: bool = False):
    """Return a Dataset graph yielding canonical ``(inputs, targets)`` pairs.

    Args:
        dataset: Source Dataset with a declared element specification, or a soft
            Definition, ConcreteDefinition, ObjectRef, StateRef, symbolic Expr
            (including Par), or explicit ``Ref(...)``/``Mat(...)`` assertion
            expected to identify one. Expressions stay inert until substituted;
            the bound source must materialize a Dataset.
        inputs: One scalar path, :class:`Select`, or nonempty tree of explicit
            ``Select`` leaves describing model inputs.
        targets: Matching target selection. It is required unless
            ``input_as_target`` is true.
        input_as_target: Reuse the selected input object as the target for an
            autoencoder-style Dataset without copying or backend conversion.

    Returns:
        An ordinary :class:`~dryml.data.dataset.Map` Dataset for a live source,
        or an inert Map Definition for a symbolic source. Its values are
        ``(inputs, targets)`` and its Method tree is persisted with the graph.

    Raises:
        TypeError: If ``dataset`` is not a Dataset or supported Dataset authority,
            or a selection tree is malformed. A symbolic source that does not
            materialize a Dataset fails at its normal Definition
            binding/materialization boundary.
        ValueError: If a selection tree is empty or targets conflict with
            ``input_as_target``.

    Side Effects:
        None. This helper does not open or materialize the Dataset, copy values,
        select a backend, or add batching.
    """

    from dryml.core.links import DefLink
    from dryml.core.template import _is_definition_value
    from dryml.data.dataset import Dataset, Map

    symbolic = isinstance(dataset, DefLink) or _is_definition_value(
        dataset, include_authority=True,
    )
    if not isinstance(dataset, Dataset) and not symbolic:
        raise TypeError("as_supervised requires a Dataset or Dataset reference.")
    if isinstance(dataset, DefLink):
        dataset = dataset.target
        if not isinstance(dataset, Dataset) and not _is_definition_value(
                dataset, include_authority=True):
            raise TypeError("as_supervised received an unsupported Ref/Mat target.")
    if type(input_as_target) is not bool:
        raise TypeError("input_as_target must be an exact bool.")
    input_method = _selection_method(inputs, symbolic=symbolic)
    if input_as_target:
        if targets is not None:
            raise ValueError("input_as_target does not accept an explicit target selection.")
        target_method = input_method
    else:
        if targets is None:
            raise ValueError("as_supervised requires targets unless input_as_target is true.")
        target_method = _selection_method(targets, symbolic=symbolic)
    project = (
        Project.defn(input_method, target_method)
        if symbolic else Project(input_method, target_method)
    )
    if symbolic:
        return Map.defn(dataset, project, preserves_examples=True)
    return Map(dataset, project, preserves_examples=True)


def _selection_method(
        selection, *, tree_leaf: bool = False, symbolic: bool = False,
):
    """Build one Project-compatible Method tree while rejecting tuple/list ambiguity."""

    if isinstance(selection, Select):
        return Select.defn(selection.idxs) if symbolic else selection
    if isinstance(selection, (str, int)):
        if tree_leaf:
            raise TypeError("Selection trees require Select.from_path leaves.")
        return Select.defn(selection) if symbolic else Select.from_path((selection,))
    if isinstance(selection, dict):
        if not selection:
            raise ValueError("Selection trees must not be empty.")
        branches = {
            key: _selection_method(branch, tree_leaf=True, symbolic=symbolic)
            for key, branch in selection.items()
        }
        return Project.defn(branches) if symbolic else Project(branches)
    if isinstance(selection, (tuple, list)):
        if not selection:
            raise ValueError("Selection trees must not be empty.")
        if any(not isinstance(branch, Select) for branch in selection):
            raise TypeError("Ambiguous tuple/list selections require Select.from_path leaves.")
        branches = type(selection)(
            _selection_method(branch, tree_leaf=True, symbolic=symbolic)
            for branch in selection
        )
        return Project.defn(branches) if symbolic else Project(branches)
    raise TypeError("Selections must be a scalar path, Select, or tree of Select leaves.")


def iter_xy(dataset, *, x_path=0, y_path=1):
    yield from _xy_dataset(dataset, x_path=x_path, y_path=y_path)


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


def materialize_supervised(dataset):
    """Materialize a canonical supervised Dataset for one-shot consumers.

    Args:
        dataset: Dataset yielding exactly ``(inputs, targets)`` pairs. It may
            yield examples directly or authored batches with dynamic final sizes.

    Returns:
        ``(inputs, targets, examples)`` with authored batches flattened to one
        dense collection and the exact submitted example count.

    Raises:
        ValueError: If the Dataset is noncanonical or empty.
        TypeError: If its values cannot be collated by the declared data backend.

    Side Effects:
        Opens and closes Dataset cursors. Existing Dataset batching and order are
        preserved; no trainer policy is applied.
    """

    if not isinstance(dataset.spec, tuple) or len(dataset.spec) != 2:
        raise ValueError("Expected a canonical Dataset yielding (inputs, targets).")
    branches = tuple(tuple(iter_specs(branch)) for branch in dataset.spec)
    if not all(branches):
        raise TypeError("Supervised Dataset branches require TensorSpec leaves.")
    flags = []
    for branch in branches:
        values = {spec.batched for spec in branch}
        if len(values) != 1:
            raise ValueError("Supervised Dataset branches must be uniformly batched or unbatched.")
        flags.append(values.pop())
    if flags[0] != flags[1]:
        raise ValueError("Supervised Dataset inputs and targets must have matching batching declarations.")
    examples = 0
    x_values = []
    y_values = []
    cursor = dataset.iterator()
    try:
        for value in cursor:
            examples += dataset.examples_in(value)
            x, y = value
            if flags[0]:
                for index in range(dataset.examples_in(value)):
                    x_values.append(nested_slice(x, index))
                    y_values.append(nested_slice(y, index))
            else:
                x_values.append(x)
                y_values.append(y)
    finally:
        cursor.close()
    if not x_values:
        raise ValueError("Cannot materialize an empty supervised Dataset.")
    return default_collate(x_values), default_collate(y_values), examples


_MISSING = object()


class Collect:
    def __init__(self, reducer=None, initial=_MISSING, finalize=None):
        self.reducer = reducer
        self.initial = initial
        self.finalize = finalize

    def __call__(self, data):
        if self.reducer is None:
            return list(data)

        it = iter(data)
        if self.initial is _MISSING:
            try:
                acc = next(it)
            except StopIteration as e:
                raise ValueError("Cannot collect an empty iterable without an initial value.") from e
        else:
            acc = self.initial

        for item in it:
            acc = self.reducer(acc, item)

        if self.finalize is not None:
            return self.finalize(acc)
        return acc


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
