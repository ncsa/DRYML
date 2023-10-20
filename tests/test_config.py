from dryml.core2 import Definition
import core2_objects as objs
import copy


def test_def_5():
    """
    A case which looks at stripping id methods
    """

    obj = objs.TestNest4(
        objs.TestNest2(
            A=objs.TestNest4(5)))

    obj_def_manual = Definition(
        objs.TestNest4,
        Definition(
            objs.TestNest2,
            A=Definition(
                objs.TestNest4,
                5)
            )
        )

    obj_def = obj.definition

    assert obj_def != obj_def_manual
    assert 'uid' in obj_def.kwargs

    assert 'uid' not in obj_def_manual.kwargs
    obj_class_def = obj_def.categorical(recursive=True)
    assert obj_class_def == obj_def_manual

    obj_class_def = obj_def.categorical(recursive=False)
    del obj_def.kwargs['uid']
    assert obj_class_def == obj_def


# def test_def_6():
#     """
#     A case which looks at stripping id methods
#     """

#     obj = objects.TestNest(('test', 'test'))
#     obj_def = obj.definition().get_cat_def()

#     assert type(obj_def.args[0]) is tuple
#     assert obj_def.args[0][0] == 'test'
#     assert obj_def.args[0][1] == 'test'

#     obj = objects.TestNest(['test', 'test'])
#     obj_def = obj.definition().get_cat_def()

#     assert type(obj_def.args[0]) is list
#     assert obj_def.args[0][0] == 'test'
#     assert obj_def.args[0][1] == 'test'


# def test_def_7():
#     """
#     A case which looks at nested definition
#     building indifferent situations
#     """

#     # Create the data containing objects
#     model_obj = objects.TestNest(10)
#     opt_obj = objects.TestNest3(20, model=model_obj)
#     loss_obj = objects.TestNest2(A='func')
#     train_fn_obj = objects.TestNest3(
#         optimizer=opt_obj,
#         loss=loss_obj,
#         epochs=10)

#     trainable_obj = objects.TestNest3(
#         model=model_obj,
#         train_fn=train_fn_obj
#     )

#     obj_def = trainable_obj.definition()

#     assert obj_def.kwargs['model'] is \
#         obj_def.kwargs['train_fn'].kwargs['optimizer'].kwargs['model']


# def test_def_8():
#     """
#     A case which looks at nested definition
#     building indifferent situations
#     """

#     # Create the data containing objects
#     model_obj = objects.TestNest(10)
#     opt_obj = objects.TestNest3(20, model=model_obj)
#     loss_obj = objects.TestNest2(A='func')
#     train_fn_obj = objects.TestNest3(
#         optimizer=opt_obj,
#         loss=loss_obj,
#         epochs=10)

#     trainable_obj = objects.TestNest3(
#         model=model_obj,
#         train_fn=train_fn_obj
#     )

#     obj_def = trainable_obj.definition()

#     # Building from plain definition
#     trainable_obj_built = obj_def.build()

#     assert trainable_obj_built['model'] is \
#         trainable_obj_built['train_fn']['optimizer']['model']
#     assert trainable_obj_built['model'].A == model_obj.A
#     assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
#     assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
#     assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A

#     # Building from 'class' definition
#     trainable_obj_built = obj_def.get_cat_def(recursive=True).build()

#     assert trainable_obj_built['model'] is \
#         trainable_obj_built['train_fn']['optimizer']['model']
#     assert trainable_obj_built['model'].A == model_obj.A
#     assert trainable_obj_built['train_fn']['optimizer'][0] == opt_obj[0]
#     assert trainable_obj_built['train_fn']['epochs'] == train_fn_obj['epochs']
#     assert trainable_obj_built['train_fn']['loss'].A == loss_obj.A
