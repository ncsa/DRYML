import dryml
import objects
import copy


def test_def_1():
    """
    Test conditions under which a definition is concrete
    """

    obj_def = dryml.ObjectDef(objects.HelloStr)

    assert not obj_def.is_concrete()

    obj = obj_def.build()

    assert obj.definition().is_concrete()


def test_def_2():
    """
    A definition is only concrete if all of its components are concrete
    """

    obj = objects.TestClassC(
        objects.TestClassC2(10),
        B=objects.TestClassC2(20))

    obj_def = obj.definition()

    assert obj_def.is_concrete()

    obj_def_cpy = copy.copy(obj_def)
    del obj_def_cpy['dry_args'][0]['dry_kwargs']['dry_id']

    assert not obj_def_cpy.is_concrete()

    obj_def_cpy = copy.copy(obj_def)
    del obj_def_cpy['dry_kwargs']['B']['dry_kwargs']['dry_id']

    assert not obj_def_cpy.is_concrete()


def test_def_3():
    """
    A definition is only concrete if all of its components are concrete
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    assert obj.definition().is_concrete()

    obj_def = obj.definition()
    del obj_def['dry_args'][0]['dry_kwargs']['dry_id']

    assert not obj_def.is_concrete()

    obj_def = obj.definition()
    del obj_def['dry_args'][0]['dry_kwargs']['A']['dry_kwargs']['dry_id']

    assert not obj_def.is_concrete()


def test_def_4():
    """
    A case where a fully specified definition wasn't marked as concrete
    """

    obj = dryml.Wrapper(objects.HelloTrainableD, A=5)

    assert obj.definition().is_concrete()


def test_def_5():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest(
        objects.TestNest2(
            A=objects.TestNest(5)))

    obj_def_manual = dryml.ObjectDef(
        objects.TestNest,
        dryml.ObjectDef(
            objects.TestNest2,
            A=dryml.ObjectDef(
                objects.TestNest,
                5)
            )
        )

    obj_def = obj.definition()

    assert obj_def != obj_def_manual
    assert 'dry_id' in obj_def.kwargs

    assert 'dry_id' not in obj_def_manual.kwargs
    obj_class_def = obj_def.get_cat_def()
    assert obj_class_def == obj_def_manual

    obj_class_def = obj_def.get_cat_def(recursive=False)
    del obj_def.kwargs['dry_id']
    assert obj_class_def == obj_def


def test_def_6():
    """
    A case which looks at stripping id methods
    """

    obj = objects.TestNest(('test', 'test'))
    obj_def = obj.definition().get_cat_def()

    assert type(obj_def.args[0]) is tuple
    assert obj_def.args[0][0] == 'test'
    assert obj_def.args[0][1] == 'test'

    obj = objects.TestNest(['test', 'test'])
    obj_def = obj.definition().get_cat_def()

    assert type(obj_def.args[0]) is list
    assert obj_def.args[0][0] == 'test'
    assert obj_def.args[0][1] == 'test'


def test_def_7():
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

    obj_def = trainable_obj.definition()

    assert obj_def.kwargs['model'] is \
        obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']


def test_def_8():
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

    obj_def = trainable_obj.definition()

    # Building from plain definition
    trainable_obj_built = obj_def.build()

    assert trainable_obj_built['model'] is \
        trainable_obj_built['train_fn']['optimizer']['model']
    assert trainable_obj_built['model'].A == model_obj.A
    assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
    assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
    assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A

    # Building from 'class' definition
    trainable_obj_built = obj_def.get_cat_def(recursive=True).build()

    assert trainable_obj_built['model'] is \
        trainable_obj_built['train_fn']['optimizer']['model']
    assert trainable_obj_built['model'].A == model_obj.A
    assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
    assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
    assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A
