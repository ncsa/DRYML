import core2_objects as objects
from dryml.core2.definition import Definition, ConcreteDefinition, SKIP_ARGS, selector_match


def test_definition_concrete_1():
    definition = Definition(
        objects.TestClass1,
        10,
        test='a')

    new_def = definition.concretize()
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition


def test_definition_concrete_2():
    definition = Definition(
        objects.TestClass4,
        10,
        test='a')

    new_def = definition.concretize()
    assert selector_match(definition, new_def)
    assert definition != new_def
    assert type(new_def) is ConcreteDefinition
    assert 'uid' in new_def.kwargs
    assert 'metadata' in new_def.kwargs


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

    conc_def_1 = concrete_def.kwargs['test']

    assert concrete_def.args[0] is conc_def_1


def test_definition_concrete_5():
    # Test that the same Definition objects produce identical ConcreteDefinition objects after concretization even after the original Definition has been deepcopied
    def_1 = Definition(objects.TestClass1, 10, test='a')
    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    def_3 = def_2.copy()

    conc_def = def_3.concretize()

    conc_def_1 = conc_def.kwargs['test']

    assert conc_def.args[0] is conc_def_1


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

    import core2_objects as objects
    obj = Definition(
        objects.TestClassB,
        1,
        base_msg='Test').build()

    assert 'uid' in obj.definition.kwargs
