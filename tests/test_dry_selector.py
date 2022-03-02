import dryml
import os
import objects


def test_selector_1():
    "Class selection"
    obj1 = objects.TestClassA(base_msg="Test1", item=5)
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test1")

    sel = dryml.DrySelector(cls=objects.TestClassA)

    # Test selectors work with built classes
    assert sel(obj1)
    assert sel(obj1.definition())
    assert not sel(obj2)
    assert not sel(obj2.definition())

    dryml.save_object(obj1, 'test1')
    dryml.save_object(obj2, 'test2')

    # Test selectors work with loaded classes

    obj1_loaded = dryml.load_object('test1')
    obj2_loaded = dryml.load_object('test2')

    assert sel(obj1_loaded)
    assert not sel(obj2_loaded)

    os.remove('test1.dry')
    os.remove('test2.dry')


def test_selector_2():
    "args selection"
    obj1 = objects.TestClassB(1, base_msg="Test1")
    obj2 = objects.TestClassB([1, 2, 3], base_msg="Test2")

    sel = dryml.DrySelector(cls=objects.TestClassB, args=(1,))

    assert sel(obj1)
    assert sel(obj1.definition())
    assert not sel(obj2)
    assert not sel(obj2.definition())


def test_selector_3():
    "kwargs selection"
    obj1 = objects.TestClassA(base_msg="Test1", item='a')
    obj2 = objects.TestClassA(base_msg="Test2", item=[10, 10, 10])

    sel = dryml.DrySelector(
        cls=objects.TestClassA,
        kwargs={'item': 'a'})

    assert sel(obj1)
    assert sel(obj1.definition())
    assert not sel(obj2)
    assert not sel(obj2.definition())

    sel = dryml.DrySelector(
        cls=objects.TestClassA,
        kwargs={'item': [10, 10, 10]})

    assert not sel(obj1)
    assert not sel(obj1.definition())
    assert sel(obj2)
    assert sel(obj2.definition())


def test_selector_4():
    "superclass selection"
    obj1 = objects.TestClassA(base_msg="Test1", item='a')
    obj2 = objects.TestClassA(base_msg="Test2",
                              item=[10, 10, 10])
    obj3 = objects.TestClassB(0, base_msg="Test3")
    obj4 = objects.TestClassB([10, 10], base_msg="Test4")
    obj5 = objects.HelloInt(msg=5)
    obj6 = objects.HelloInt(msg=20)
    obj7 = objects.HelloStr(msg='test')
    obj8 = objects.HelloStr(msg='2test')

    sel = dryml.DrySelector(
        cls=objects.TestBase,
        cls_str_compare=False,
        verbosity=2)

    assert sel(obj1)
    assert sel(obj1.definition())
    assert sel(obj2)
    assert sel(obj2.definition())
    assert sel(obj3)
    assert sel(obj3.definition())
    assert sel(obj4)
    assert sel(obj4.definition())
    assert not sel(obj5)
    assert not sel(obj5.definition())
    assert not sel(obj6)
    assert not sel(obj6.definition())
    assert not sel(obj7)
    assert not sel(obj7.definition())
    assert not sel(obj8)
    assert not sel(obj8.definition())

    sel = dryml.DrySelector(
        cls=objects.HelloObject,
        cls_str_compare=False)

    assert not sel(obj1)
    assert not sel(obj1.definition())
    assert not sel(obj2)
    assert not sel(obj2.definition())
    assert not sel(obj3)
    assert not sel(obj3.definition())
    assert not sel(obj4)
    assert not sel(obj4.definition())
    assert sel(obj5)
    assert sel(obj5.definition())
    assert sel(obj6)
    assert sel(obj6.definition())
    assert sel(obj7)
    assert sel(obj7.definition())
    assert sel(obj8)
    assert sel(obj8.definition())


def test_selector_5():
    """
    Nested class selection
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))
    obj_def = obj.definition()

    sel = dryml.DrySelector(
        cls=objects.TestNest,
        args=obj_def['dry_args'])

    assert sel(obj)

    del obj_def['dry_args'][0]['dry_kwargs']['dry_id']

    sel = dryml.DrySelector(
        cls=objects.TestNest,
        args=obj_def['dry_args'])

    assert sel(obj)

    obj_def = obj.definition()
    del obj_def['dry_args'][0]['dry_kwargs']['A']['dry_kwargs']['dry_id']

    assert sel(obj)


def test_selector_6():
    """
    A Selector usage pattern
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    obj_def = dryml.DryObjectDef(
        objects.TestNest,
        dryml.DryObjectDef(
            objects.HelloTrainableD,
            A=dryml.DryObjectDef(
                objects.TestNest,
                10)
        )
    )

    sel = dryml.DrySelector.build(obj_def)

    assert sel(obj)


def test_selector_build_1():
    """
    Test that we can construct DrySelectors from various objects
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    sel = dryml.DrySelector.from_obj(obj)

    assert sel(obj)


def test_selector_build_2():
    """
    Test that we can construct DrySelectors from various objects
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    sel = dryml.DrySelector.from_def(obj.definition())

    assert sel(obj)


def test_selector_build_3():
    """
    Test that we can construct DrySelectors from various objects
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    sel = dryml.DrySelector.from_dict({
        'cls': objects.TestNest,
        'dry_args': [{'cls': objects.HelloTrainableD,
                      'dry_kwargs': {'A': {'cls': objects.TestNest}}}]
    })

    assert sel(obj)


def test_selector_build_4():
    """
    Test that we can construct DrySelectors from various objects
    """

    obj = objects.TestNest(objects.HelloTrainableD(A=objects.TestNest(10)))

    sel = dryml.DrySelector.build(obj)

    assert sel(obj)

    sel = dryml.DrySelector.build(obj.definition())

    assert sel(obj)

    sel = dryml.DrySelector.build({
        'cls': objects.TestNest,
        'dry_args': [{'cls': objects.HelloTrainableD,
                      'dry_kwargs': {'A': {'cls': objects.TestNest}}}]
    })

    assert sel(obj)
