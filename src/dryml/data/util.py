"""Dataset projection, nested-value utilities and owned one-shot collection.

Supervised collection retains permissive consumers by default and offers an
opt-in finite, runtime-validated dense NumPy boundary without foreign conversion.
"""

import inspect
from typing import Callable

import numpy as np

from dryml.core.authoring import authoring_helper
from dryml.core.backend import Backend
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, iter_specs
from dryml.data.collate import collate_for_spec, default_collate
from dryml.data.methods import Project, Select


def _xy_dataset(dataset, *, x_path=0, y_path=1):
    from dryml.data.dataset import Map

    return Map(dataset, Project(Select(x_path), Select(y_path)))


def _supervised_should_author(arguments):
    """Select authoring for a symbolic or exact source, not selection controls."""

    from dryml.core.links import DefLink
    from dryml.core.template import _is_definition_value

    dataset = arguments["dataset"]
    return isinstance(dataset, DefLink) or _is_definition_value(dataset, include_authority=True)


def _supervised_definition(arguments):
    """Build a supervised Definition using the helper's shared graph recipe."""

    return _supervised_graph(**arguments, symbolic=True)


@authoring_helper(
    author_definition=_supervised_definition,
    should_author=_supervised_should_author,
)
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

    return _supervised_graph(
        dataset, inputs, targets, input_as_target=input_as_target, symbolic=False,
    )


def _supervised_graph(dataset, inputs, targets, *, input_as_target, symbolic):
    """Validate and assemble either live or inert supervised projection nodes."""

    from dryml.core.links import DefLink
    from dryml.core.template import _is_definition_value
    from dryml.data.dataset import Dataset, Map

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


def materialize_supervised(dataset, *, strict_numpy: bool = False):
    """Materialize a canonical supervised Dataset for one-shot consumers.

    Args:
        dataset: Dataset yielding exactly ``(inputs, targets)`` pairs. It may
            yield examples directly or authored batches with dynamic final sizes.
        strict_numpy: Exact bool, default false. When true, require finite logical
            yield cardinality and two dense NumPy TensorSpec branches before
            opening a source. Features must be vectors and targets scalars per
            example, with fixed-width real integer or float32/float64 dtypes.
            Validate runtime pairs, native NumPy values, exact dtypes, declared
            rank/shape and batch facts before collation. Dynamic feature width
            must be positive and consistent; dynamic batches may vary in size.
            Dense ndarray subclasses are accepted, but masked arrays are not.

    Returns:
        ``(inputs, targets, examples)`` with authored batches flattened to one
        dense collection and the exact submitted example count. Strict mode
        returns NumPy arrays of shapes ``(examples, features)`` and ``(examples,)``
        retaining each declared dtype and authored order. Unknown example-count
        metadata is allowed when logical yield cardinality is finite.

    Raises:
        ValueError: If the Dataset is noncanonical or empty; in strict mode, also
            for unbounded logical cardinality, excess yields, invalid shapes,
            feature-width drift, or inconsistent batching.
        DatasetExhaustedError: In strict mode, if traversal ends before its
            declared logical yield count, with requested and observed counts.
        TypeError: If strict_numpy is not an exact bool, values cannot be collated,
            or strict declarations/runtime values have unsupported trees, layouts,
            backends or dtypes (including Python scalar/list conversion).
        BaseException: Source acquisition, reading, validation, interruption or
            cleanup failure. A primary failure is preserved when cleanup fails.

    Side Effects:
        Opens and closes Dataset cursors. Existing Dataset batching and order are
        preserved; no trainer policy is applied. Strict mode opens nothing for
        invalid declarations, reads at most the first excess logical yield, and
        respects authored cursor bounds such as Take without reaching underneath
        their EOF. It imports no optional tensor/model framework and adds no
        conversion edges. NaN/Inf policy remains the consumer's responsibility.
        Strict mode copies validated native values before advancing the source,
        preserving each yield even when a producer reuses its NumPy buffers.
        The default retains permissive, including unknown-cardinality, collection.
    """

    from dryml.data.dataset import DatasetCursor

    if type(strict_numpy) is not bool:
        raise TypeError("strict_numpy must be an exact bool.")
    spec = dataset.spec
    if not isinstance(spec, tuple) or len(spec) != 2:
        raise ValueError("Expected a canonical Dataset yielding (inputs, targets).")
    expected = None
    if strict_numpy:
        expected = dataset.yield_cardinality().require_finite()
        _admit_numpy_supervised(spec)
    branches = tuple(tuple(iter_specs(branch)) for branch in spec)
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
    feature_width = None
    cursor = DatasetCursor._from_generator(
        lambda owner: _supervised_yields(dataset, expected, owner),
    )
    with cursor:
        for value in cursor:
            if strict_numpy:
                feature_width = _validate_numpy_supervised(value, spec, feature_width)
            count = dataset.examples_in(value)
            examples += count
            x, y = value
            if strict_numpy:
                x, y = np.asarray(x).copy(), np.asarray(y).copy()
            if flags[0]:
                for index in range(count):
                    x_values.append(nested_slice(x, index))
                    y_values.append(nested_slice(y, index))
            else:
                x_values.append(x)
                y_values.append(y)
    if not x_values:
        raise ValueError("Cannot materialize an empty supervised Dataset.")
    collate = collate_for_spec(spec) if strict_numpy else default_collate
    return collate(x_values), collate(y_values), examples


def _admit_numpy_supervised(spec):
    """Check strict dense declarations without source access or native discovery."""

    for branch in spec:
        if not isinstance(branch, TensorSpec):
            raise TypeError("Strict NumPy supervision requires two TensorSpec branches.")
        if branch.backend is not Backend.numpy or branch.layout is not Layout.DENSE:
            raise TypeError("Strict supervision requires declared dense NumPy inputs and targets.")
        dtype = branch.dtype
        if not ((dtype.kind in {"int", "uint"} and dtype.bits in {8, 16, 32, 64})
                or (dtype.kind == "float" and dtype.bits in {32, 64})):
            raise TypeError("Strict NumPy supervision requires real integer or float32/float64 dtypes.")
        if branch.batch == 0:
            raise ValueError("Strict NumPy supervision cannot declare zero-example batches.")
    x, y = spec
    if x.rank != 1 or y.shape != ():
        raise ValueError("Strict NumPy supervision requires feature vectors and scalar targets.")
    if x.shape[0] != Dynamic and x.shape[0] <= 0:
        raise ValueError("Strict NumPy supervision requires positive feature width.")
    if (x.batched != y.batched or
            (x.batched and x.batch is not Dynamic and y.batch is not Dynamic and x.batch != y.batch)):
        raise ValueError("Strict NumPy supervision requires matching batch declarations.")


def _validate_numpy_supervised(value, spec, feature_width):
    """Validate native facts before any stack can convert values or promote dtypes."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("Expected a canonical runtime (inputs, targets) tuple.")
    for name, runtime, declared in zip(("inputs", "targets"), value, spec):
        if (not isinstance(runtime, (np.ndarray, np.generic))
                or np.ma.isMaskedArray(runtime)):
            raise TypeError(f"Strict {name} must be a dense NumPy ndarray or NumPy scalar.")
        if runtime.dtype != np.dtype(declared.dtype.np()):
            raise TypeError(f"Strict {name} dtype differs from its declared dtype.")
        if not declared.compatible_with_shape(runtime.shape):
            raise ValueError(f"Strict {name} rank/shape differs from its declared shape or batch size.")
    width = value[0].shape[-1]
    if width <= 0 or (feature_width is not None and width != feature_width):
        raise ValueError("Strict NumPy supervision requires positive, consistent feature width.")
    return width


def _supervised_yields(dataset, expected, owner):
    """Check logical counts inside the cursor's primary-error/cleanup boundary."""

    from dryml.data.dataset import DatasetExhaustedError

    source = owner._own(owner._call(dataset.iterator))
    yielded = 0
    while True:
        try:
            value = owner._pull(source)
        except StopIteration as error:
            if expected is not None and yielded != expected:
                raise DatasetExhaustedError(expected, yielded) from error
            return
        if expected is not None and yielded >= expected:
            raise ValueError(f"Supervised Dataset exceeds its declared yield count of {expected}.")
        yielded += 1
        yield value


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
