import dryml
from dryml.core2 import Definition, SKIP_ARGS
import os
import core2_objects as objs


def test_selector_1():
    "Class selection"
    obj1 = objs.TestClassA(base_msg="Test1", item=5)
    obj2 = objs.TestClassB([1, 2, 3], base_msg="Test1")

    sel = Definition(objs.TestClassA, SKIP_ARGS)

    # Test selectors work with built classes
    assert sel(obj1)
    assert sel(obj1.definition)
    assert not sel(obj2)
    assert not sel(obj2.definition)

    dryml.core2.save_object(obj1, dest='test1.dry')
    dryml.core2.save_object(obj2, dest='test2.dry')

    # Test selectors work with loaded classes

    obj1_loaded = dryml.core2.load_object(dest='test1.dry')
    obj2_loaded = dryml.core2.load_object(dest='test2.dry')

    assert sel(obj1_loaded)
    assert not sel(obj2_loaded)

    os.remove('test1.dry')
    os.remove('test2.dry')


# def test_selector_2():
#     "args selection"
#     obj1 = objects.TestClassB(1, base_msg="Test1")
#     obj2 = objects.TestClassB([1, 2, 3], base_msg="Test2")

#     sel = dryml.Selector(cls=objects.TestClassB, args=(1,))

#     assert sel(obj1)
#     assert sel(obj1.definition())
#     assert not sel(obj2)
#     assert not sel(obj2.definition())


# def test_selector_3():
#     "kwargs selection"
#     obj1 = objects.TestClassA(base_msg="Test1", item='a')
#     obj2 = objects.TestClassA(base_msg="Test2", item=[10, 10, 10])

#     sel = dryml.Selector(
#         cls=objects.TestClassA,
#         kwargs={'item': 'a'})

#     assert sel(obj1)
#     assert sel(obj1.definition())
#     assert not sel(obj2)
#     assert not sel(obj2.definition())

#     sel = dryml.Selector(
#         cls=objects.TestClassA,
#         kwargs={'item': [10, 10, 10]})

#     assert not sel(obj1)
#     assert not sel(obj1.definition())
#     assert sel(obj2)
#     assert sel(obj2.definition())


# def test_selector_4():
#     "superclass selection"
#     obj1 = objects.TestClassA(base_msg="Test1", item='a')
#     obj2 = objects.TestClassA(base_msg="Test2",
#                               item=[10, 10, 10])
#     obj3 = objects.TestClassB(0, base_msg="Test3")
#     obj4 = objects.TestClassB([10, 10], base_msg="Test4")
#     obj5 = objects.HelloInt(msg=5)
#     obj6 = objects.HelloInt(msg=20)
#     obj7 = objects.HelloStr(msg='test')
#     obj8 = objects.HelloStr(msg='2test')

#     sel = dryml.Selector(
#         cls=objects.TestBase)

#     assert sel(
#         obj1,
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj1.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj2,
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj2.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj3,
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj3.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj4,
#         cls_str_compare=False,
#         verbosity=2)
#     assert sel(
#         obj4.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj5,
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj5.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj6,
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj6.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj7,
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj7.definition(),
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj8,
#         cls_str_compare=False,
#         verbosity=2)
#     assert not sel(
#         obj8.definition(),
#         cls_str_compare=False,
#         verbosity=2)

#     sel = dryml.Selector(
#         cls=objects.HelloObject)

#     assert not sel(
#         obj1,
#         cls_str_compare=False)
#     assert not sel(
#         obj1.definition(),
#         cls_str_compare=False)
#     assert not sel(
#         obj2,
#         cls_str_compare=False)
#     assert not sel(
#         obj2.definition(),
#         cls_str_compare=False)
#     assert not sel(
#         obj3,
#         cls_str_compare=False)
#     assert not sel(
#         obj3.definition(),
#         cls_str_compare=False)
#     assert not sel(
#         obj4,
#         cls_str_compare=False)
#     assert not sel(
#         obj4.definition(),
#         cls_str_compare=False)
#     assert sel(
#         obj5,
#         cls_str_compare=False)
#     assert sel(
#         obj5.definition(),
#         cls_str_compare=False)
#     assert sel(
#         obj6,
#         cls_str_compare=False)
#     assert sel(
#         obj6.definition(),
#         cls_str_compare=False)
#     assert sel(
#         obj7,
#         cls_str_compare=False)
#     assert sel(
#         obj7.definition(),
#         cls_str_compare=False)
#     assert sel(
#         obj8,
#         cls_str_compare=False)
#     assert sel(
#         obj8.definition(),
#         cls_str_compare=False)


# def test_selector_5():
#     """
#     Nested class selection
#     """

#     obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))
#     obj_def = obj.definition()

#     sel = dryml.Selector(
#         cls=objects.TestNest,
#         args=obj_def['dry_args'])

#     assert sel(obj)

#     del obj_def['dry_args'][0]['dry_kwargs']['dry_id']

#     sel = dryml.Selector(
#         cls=objects.TestNest,
#         args=obj_def['dry_args'])

#     assert sel(obj)

#     obj_def = obj.definition()
#     del obj_def['dry_args'][0]['dry_kwargs']['A']['dry_kwargs']['dry_id']

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
