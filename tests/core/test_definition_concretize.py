from tests.core import core_objects as objects
from dryml.core.definition import Definition, ConcreteDefinition, SKIP_ARGS, selector_match, \
    concretize_func, thaw_concrete
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
import numpy as np


def test_definition_concrete_1():
    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    new_def = definition.concretize()
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition
    assert new_def.identity_version == V2_IDENTITY_VERSION


def test_public_concretization_uses_the_bound_v2_pipeline():
    definition = Definition(objects.TestClass1, 10, test="a")

    private_v2 = definition.concretize()

    assert private_v2.identity_version == V2_IDENTITY_VERSION
    assert definition.concretize().identity_version == V2_IDENTITY_VERSION
    assert definition.concretize() == private_v2


def test_definition_concrete_2():
    definition = Definition(
        objects.TestClass4,
        10,
        test='a')

    new_def = definition.concretize()
    assert selector_match(definition, new_def, verbose=True)
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition
    assert 'uid' in new_def.parameters['kwargs']
    assert 'metadata' in new_def.parameters['kwargs']


def test_definition_concrete_3():
    # Test that concrete definition is hashable
    definition = Definition(objects.TestClass1, 10, test='a').concretize()
    hash(definition)


def test_definition_concrete_4():
    # Test that the same Definition objects product identical ConcreteDefinition objects after concretization
    def_1 = Definition(objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    concrete_def = def_2.concretize()

    conc_def_1 = concrete_def.parameters['test']

    assert concrete_def.parameters['x'].match(conc_def_1, strict=True)

def test_definition_concretize_types_1():
    # Test that concretize properly transforms containers and other types.
    from dryml.core.freeze import FrozenList, FrozenTuple, FrozenSet, FrozenDict, FrozenNDArray

    test_pairs = [
        ([1,2,3], FrozenList),
        ((1,2,3), FrozenTuple),
        (set([1,2,3]), FrozenSet),
        ({1: 2, 3: 4}, FrozenDict),
        (np.array([1,2,3]), FrozenNDArray),
    ]

    for test_input, expected_type in test_pairs:
        result = concretize_func(test_input)
        assert isinstance(result, expected_type)
        assert len(result) == len(test_input)


def test_definition_concretize_types_2():
    # Test that concretize properly transforms containers and other types.
    from dryml.core.freeze import FrozenList, FrozenTuple, FrozenSet, FrozenDict, FrozenNDArray

    test_pairs = [
        (FrozenList([1,2,3]), list),
        (FrozenTuple((1,2,3)), tuple),
        (FrozenSet([1,2,3]), set),
        (FrozenDict({1: 2, 3: 4}), dict),
        (FrozenNDArray.from_array(np.array([1,2,3])), np.ndarray),
    ]

    for test_input, expected_type in test_pairs:
        result = thaw_concrete(test_input)
        assert isinstance(result, expected_type)
        assert len(result) == len(test_input)


def test_definition_concretize_types_3():
    obj = objects.TestWrapper(list, (1, 2, 3))
    assert type(obj.obj) is list
    assert obj.obj == [1,2,3]


# def test_definition_concrete_5():
#     # Test that the same Definition objects produce identical ConcreteDefinition objects after concretization even after the original Definition has been deepcopied
#     def_1 = Definition(objects.TestClass1, 10, test='a')
#     def_2 = Definition(
#         objects.TestClass1,
#         def_1,
#         test=def_1)

#     def_3 = def_2.copy()

#     conc_def = def_3.concretize()

#     conc_def_1 = conc_def.kwargs['test']

#     assert conc_def.args[0].match(conc_def_1, strict=True)


def test_definition_concrete_6():
    def_1 = Definition(objects.TestClass1, SKIP_ARGS, test='a')

    flag = True
    try:
        def_1.concretize()
        flag = False
    except ValueError:
        pass

    assert flag


def test_object_build_from_def_1():
    """
    Test that an object definition with no id results in an object with an id.
    """

    from tests.core import core_objects as objects
    obj = Definition(
        objects.TestClassB,
        1,
        base_msg='Test').build()

    assert 'uid' in obj.definition.parameters['kwargs']
