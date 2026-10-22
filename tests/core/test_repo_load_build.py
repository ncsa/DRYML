import pytest

from tests.core import core_objects as objects
from dryml.core.definition import Definition
from dryml.core.repo import Repo
from dryml.core import save_object, load_object


def test_partial_node_build_and_apply_have_distinct_unspecified_state_semantics(tmp_path):
    from dryml.core import ObjectId
    from dryml.core.store.dir import DirStore
    from tests.core.test_state_ref_restore_into import RestoreIntoValue, RestoreIntoIndependentPair

    repo = Repo(DirStore(tmp_path / "store"))
    donor = RestoreIntoValue(1, repo=repo)
    donor.value = 7
    selected = repo.save(donor).state
    first = Definition(RestoreIntoValue, 1)
    recipe = Definition(RestoreIntoIndependentPair, first=first, second=Definition(RestoreIntoValue, 2))
    identity = ObjectId()
    partial = recipe.bind(identities={"$.first": identity}, states={"$.first": selected})
    import sys
    allocated = []
    previous = sys.getprofile()

    def count_identity(frame, event, arg):
        if event == "call" and frame.f_code is ObjectId.__init__.__code__ and frame.f_locals["namespace"] is None:
            allocated.append(True)
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(count_identity)
    try:
        built = partial.build(repo=repo, cache="none")
    finally:
        sys.setprofile(previous)
    assert len(allocated) == 1
    assert built.first.value == 7 and built.second.value == 2
    assert built.first.object_id == identity
    assert built.second.object_id is not None
    built.first.value, built.second.value = 90, 91
    before = built.first.object_id, built.second.object_id
    repo.apply_state(built, partial.concretize().state_spec())
    assert built.first.value == 7 and built.second.value == 91
    assert (built.first.object_id, built.second.object_id) == before


def test_build_from_definition_repo_1():
    # Test we can build one object
    definition = Definition(
        objects.TestClass1,
        10, test='a')

    repo = Repo() 
    obj = definition.build(repo=repo)
    assert repo._num_constructions == 1
    assert definition(obj.definition)
    assert definition(obj)


def test_build_from_definition_repo_2():
    # Test we can build a nested object
    definition = Definition(
        objects.TestClass1,
        Definition(
            objects.TestClass1,
            10,
            test='b'),
        test='a')

    repo = Repo() 
    obj = definition.build(repo=repo)
    assert repo._num_constructions == 2
    assert definition(obj.definition, verbose=True)
    assert definition(obj, verbose=True)


def test_build_from_definition_repo_3():
    # Test instance caching
    def_1 = Definition(
        objects.TestClass1,
        10,
        test='a')

    def_2 = Definition(
        objects.TestClass1,
        def_1,
        test=def_1)

    repo = Repo()

    obj = def_2.build(repo=repo)
    assert obj.x is obj.test
    assert repo._num_constructions == 2


def test_object_args_passing_1():
    from tests.core import core_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    assert obj.definition.parameters["layers"] == 1


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_object_args_passing_2(tmp_path):
    from tests.core import core_objects as objs

    obj = objs.TestClassB(1, base_msg="Test1")

    from dryml.core.store.dir import DirStore
    repo = Repo(DirStore(tmp_path / "store"))
    saved = repo.save(obj)
    obj_loaded = repo.load(saved)

    assert obj_loaded.definition.parameters["layers"] == 1


def test_object_args_passing_3():
    """
    Test passing nested dryobjects as arguments
    """
    from tests.core import core_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest(obj1)

    obj3 = objs.TestNest(obj2)

    obj1_def = obj1.definition

    obj1_cpy = obj1_def.build(cache="none")

    assert obj1.definition.graph_equal(obj1_cpy.definition)
    assert obj1 is not obj1_cpy

    obj2_def = obj2.definition

    obj2_cpy = obj2_def.build(cache="none")

    assert obj2.definition.graph_equal(obj2_cpy.definition)
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A) is type(obj1)
    assert type(obj2_cpy.A) is type(obj1)

    obj3_cpy = obj3.definition.build(cache="none")

    assert obj3.definition.graph_equal(obj3_cpy.definition)
    assert obj3 is not obj3_cpy
    assert obj3.A is not obj3_cpy.A
    assert type(obj3.A) is type(obj2)
    assert type(obj3_cpy.A) is type(obj2)
    assert obj3.A.A is not obj3_cpy.A.A
    assert type(obj3.A.A) is type(obj1)
    assert type(obj3_cpy.A.A) is type(obj1)


def test_object_args_passing_4():
    """
    Test passing nested dryobjects as arguments, within a list
    """
    from tests.core import core_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([obj1])

    obj1_cpy = obj1.definition.build(cache="none")

    assert obj1.definition.graph_equal(obj1_cpy.definition)
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(cache="none")

    assert obj2.definition.graph_equal(obj2_cpy.definition)
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0]) is type(obj1)
    assert type(obj2_cpy.A[0]) is type(obj1)


def test_object_args_passing_5():
    """
    Test passing nested dryobjects as arguments, within a nested list
    """
    from tests.core import core_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest([[obj1]])

    obj1_cpy = obj1.definition.build(cache="none")

    assert obj1.definition.graph_equal(obj1_cpy.definition)
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(cache="none")

    assert obj2.definition.graph_equal(obj2_cpy.definition)
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A[0][0]) is type(obj1)
    assert type(obj2_cpy.A[0][0]) is type(obj1)


def test_object_args_passing_6():
    """
    Test passing nested dryobjects as arguments, within a dict
    """
    from tests.core import core_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': obj1})

    obj1_cpy = obj1.definition.build(cache="none")

    assert obj1.definition.graph_equal(obj1_cpy.definition)
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(cache="none")

    assert obj2.definition.graph_equal(obj2_cpy.definition)
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A']) is type(obj1)
    assert type(obj2_cpy.A['A']) is type(obj1)


def test_object_args_passing_7():
    """
    Test passing nested dryobjects as arguments, within a dict with a list
    """
    from tests.core import core_objects as objs

    obj1 = objs.TestNest(10)

    obj2 = objs.TestNest({'A': [[obj1]]})

    obj1_cpy = obj1.definition.build(cache="none")

    assert obj1.definition.graph_equal(obj1_cpy.definition)
    assert obj1 is not obj1_cpy

    obj2_cpy = obj2.definition.build(cache="none")

    assert obj2.definition.graph_equal(obj2_cpy.definition)
    assert obj2 is not obj2_cpy
    assert obj2.A is not obj2_cpy.A
    assert type(obj2.A['A'][0][0]) is type(obj1)
    assert type(obj2_cpy.A['A'][0][0]) is type(obj1)
