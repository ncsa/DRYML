import dryml
from dryml.core2.definition import Definition, SKIP_ARGS
import os
import core2_objects as objs


# def test_selector_5():
#     """
#     Nested class selection
#     """

#     obj = objs.TestNest4(objs.HelloTrainableD(A=objs.TestNest4(10)))
#     obj_def = obj.definition.copy()

#     sel = Definition(
#         objs.TestNest4,
#         *objs.__args__)

#     assert sel(obj)

#     del obj_def.args[0].kwargs['uid']

#     sel = Definition(
#         objs.TestNest4,
#         *obj_def.args)

#     assert sel(obj)

#     obj_def = obj.definition.copy()
#     del obj_def.args[0].kwargs['A'].kwargs['uid']

#     assert sel(obj)


# def test_selector_6():
#     """
#     A Selector usage pattern
#     """

#     obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

#     obj_def = dryml.ObjectDef(
#         objects.TestNest,
#         dryml.ObjectDef(
#             objects.HelloTrainableD,
#             A=dryml.ObjectDef(
#                 objects.TestNest,
#                 10)
#         )
#     )

#     sel = dryml.Selector.build(obj_def)

#     assert sel(obj)


# def test_selector_7():
#     """
#     Test real life example
#     """

#     import dryml.data

#     # Create transform object
#     def test_func(x):
#         return x*2

#     real_obj = dryml.data.transforms.FuncTransform.from_function(
#         test_func, framework='tf')

#     obj_def = dryml.ObjectDef(
#         dryml.data.transforms.FuncTransform,
#         real_obj.dry_args[0])

#     assert dryml.Selector.from_def(obj_def)(real_obj)


# def test_selector_8():
#     """
#     Test parent/child class selection
#     """

#     parent_obj = objects.TestBase()

#     assert dryml.Selector(objects.TestBase)(parent_obj)
#     assert not dryml.Selector(objects.TestClassA)(parent_obj)

#     assert dryml.Selector.from_def(
#         dryml.ObjectDef(
#             objects.TestBase))(parent_obj)
#     assert not dryml.Selector.from_def(
#         dryml.ObjectDef(
#             objects.TestClassA))(parent_obj)


# def test_selector_9():
#     """
#     Test parent/child class selection in an argument
#     """

#     parent_obj = objects.TestNest(objects.TestBase)

#     assert dryml.Selector.from_def(
#        dryml.ObjectDef(
#            objects.TestNest,
#            objects.TestBase))(parent_obj)
#     assert not dryml.Selector.from_def(
#        dryml.ObjectDef(
#            objects.TestNest,
#            objects.TestClassA))(parent_obj)


# def test_selector_10():
#     """
#     Test parent/child class selection in a keyword argument
#     """

#     parent_obj = objects.TestNest2(A=objects.TestBase)

#     assert dryml.Selector.from_def(
#        dryml.ObjectDef(
#            objects.TestNest2,
#            A=objects.TestBase))(parent_obj)
#     assert not dryml.Selector.from_def(
#        dryml.ObjectDef(
#            objects.TestNest2,
#            A=objects.TestClassA))(parent_obj)


# def test_selector_build_1():
#     """
#     Test that we can construct Selectors from various objects
#     """

#     obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

#     sel = dryml.Selector.from_obj(obj)

#     assert sel(obj)


# def test_selector_build_2():
#     """
#     Test that we can construct Selectors from various objects
#     """

#     obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

#     sel = dryml.Selector.from_def(obj.definition())

#     assert sel(obj)


# def test_selector_build_5():
#     """
#     Test that we can construct Selectors from nested objects
#     """

#     obj1 = objects.TestNest2(A=1)

#     obj_def = dryml.ObjectDef(
#         objects.TestNest3,
#         obj1,
#         dryml.ObjectDef(
#             objects.TestNest2,
#             A=2),
#         dryml.ObjectDef(
#             objects.TestNest,
#             dryml.ObjectDef(
#                 objects.TestNest2,
#                 A=5,)
#             )
#         )

#     obj = obj_def.build()

#     sel = dryml.Selector.build(obj_def)

#     assert sel(obj, verbosity=2)


# def test_selector_build_6():
#     """
#     Test that we can construct Selectors from nested objects
#     """

#     obj1 = objects.TestNest2(A=1)

#     obj_def = dryml.ObjectDef(
#         objects.TestNest3,
#         obj1,
#         dryml.ObjectDef(
#             objects.TestNest2,
#             A=2),
#         dryml.ObjectDef(
#             objects.TestNest,
#             dryml.ObjectDef(
#                 objects.TestNest2,
#                 A=5,)
#             )
#         )

#     obj = obj_def.build()

#     sel = dryml.Selector.build(obj_def)

#     assert sel(obj, verbosity=2)
