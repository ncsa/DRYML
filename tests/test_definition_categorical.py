import core2_objects as objects
from dryml.core2.definition import Definition


def test_def_1():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest4(
        objects.TestNest2(
            A=objects.TestNest4(5)))

    obj_def_manual = Definition(
        objects.TestNest4,
        Definition(
            objects.TestNest2,
            A=Definition(
                objects.TestNest4,
                5)
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
    obj_def_thawed = obj_def.to_definition()
    del obj_def_thawed.kwargs['uid']
    assert obj_class_def(obj_def_thawed)


def test_def_2():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest4(('test', 'test'))
    obj_def = obj.definition.categorical()

    assert type(obj_def.args[0]) is tuple
    assert obj_def.args[0][0] == 'test'
    assert obj_def.args[0][1] == 'test'

    obj = objects.TestNest4(['test', 'test'])
    obj_def = obj.definition.categorical()

    assert type(obj_def.args[0]) is list
    assert obj_def.args[0][0] == 'test'
    assert obj_def.args[0][1] == 'test'


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

    # Building from plain definition
    trainable_obj_built = obj_def.to_definition().build()

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
