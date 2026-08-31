from tests.core import core_objects as objects
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.definition import Definition
from dryml.core.freeze import FrozenList, FrozenTuple


def test_def_1():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest4(
        objects.TestNest2(
            A=objects.TestNest4(5)))

    obj_def_manual = Definition(
        objects.TestNest4,
        A=Definition(
            objects.TestNest2,
            A=Definition(
                objects.TestNest4,
                A=5)
            )
        )

    obj_def = obj.definition

    assert obj_def != obj_def_manual
    assert 'uid' in obj_def.kwargs

    assert 'uid' not in obj_def_manual.kwargs
    obj_class_def = obj_def.categorical(recursive=True)
    ic(obj_def, obj_class_def, obj_class_def.args, obj_class_def.kwargs)
    assert obj_class_def.match(obj_def_manual)

    obj_class_def = obj_def.categorical(recursive=False)
    obj_def_thawed = obj_def.thaw().without_kwarg('uid')
    assert obj_class_def(obj_def_thawed)


def test_def_2():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest4(('test', 'test'))
    obj_def = obj.definition.categorical()

    assert type(obj_def.parameters["A"]) is FrozenTuple
    assert obj_def.parameters["A"][0] == 'test'
    assert obj_def.parameters["A"][1] == 'test'

    obj = objects.TestNest4(['test', 'test'])
    obj_def = obj.definition.categorical()

    assert type(obj_def.parameters["A"]) is FrozenList
    assert obj_def.parameters["A"][0] == 'test'
    assert obj_def.parameters["A"][1] == 'test'


def test_def_3():
    """
    A case which looks at nested definition
    building indifferent situations
    """

    # Create the data containing objects
    model_obj = objects.TestNest(10)
    opt_obj = objects.TestNest3(20, model=model_obj)
    loss_obj = objects.TestNest2(A='func')
    train_fn_obj = objects.TestNest3(
        optimizer=opt_obj,
        loss=loss_obj,
        epochs=10)

    trainable_obj = objects.TestNest3(
        model=model_obj,
        train_fn=train_fn_obj
    )

    obj_def = trainable_obj.definition

    assert obj_def.kwargs['model'] == obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']
    #assert obj_def.kwargs['model']._obj is not None
    #assert obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']._obj is not None
    #assert obj_def.kwargs['model']._obj is obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']._obj


def test_def_4():
    """
    A case which looks at nested definition
    building indifferent situations
    """

    # Create the data containing objects
    model_obj = objects.TestNest(10)
    opt_obj = objects.TestNest3(20, model=model_obj)
    loss_obj = objects.TestNest2(A='func')
    train_fn_obj = objects.TestNest3(
        optimizer=opt_obj,
        loss=loss_obj,
        epochs=10)

    trainable_obj = objects.TestNest3(
        model=model_obj,
        train_fn=train_fn_obj
    )

    obj_def = trainable_obj.definition
    source_model_def = obj_def.kwargs['model']
    source_nested_model_def = obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']
    assert source_model_def is source_nested_model_def
    assert cdef_node_key(source_model_def) is cdef_node_key(source_nested_model_def)

    # Building from plain definition
    thawed = obj_def.thaw()
    assert thawed.kwargs['model'] is thawed.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']
    rebuilt_def = thawed.concretize()
    rebuilt_model_def = rebuilt_def.kwargs['model']
    rebuilt_nested_model_def = rebuilt_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']
    assert rebuilt_model_def is rebuilt_nested_model_def
    assert cdef_node_key(rebuilt_model_def) is cdef_node_key(rebuilt_nested_model_def)

    trainable_obj_built = thawed.build()

    assert trainable_obj_built['model'] is \
        trainable_obj_built['train_fn']['optimizer']['model']
    assert trainable_obj_built['model'].A == model_obj.A
    assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
    assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
    assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A

    # Building from 'class' definition
    trainable_obj_built = obj_def.categorical(recursive=True).build()

    assert trainable_obj_built['model'] is \
        trainable_obj_built['train_fn']['optimizer']['model']
    assert trainable_obj_built['model'].A == model_obj.A
    assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
    assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
    assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A
