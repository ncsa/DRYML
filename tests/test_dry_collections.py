from dryml import DryList
from dryml import DryRepo
import objects


def test_dry_list_1():
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")
    the_list = DryList(obj1, obj2)

    print(the_list.definition())
    new_list = the_list.definition().build()
    assert new_list[0].definition() == obj1.definition()
    assert new_list[1].definition() == obj2.definition()


def test_dry_list_2():
    # Create a repo
    repo = DryRepo()

    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")

    # Add them to the repo
    repo.add_object(obj1)
    repo.add_object(obj2)

    # Create a list object
    the_list = DryList(obj1, obj2)

    # Create a new list object from a definition, but use a repo
    new_list = the_list.definition().build(repo=repo)

    # The list object should've extracted the objects themselves
    # From the repo instead of constructing them
    assert the_list[0] is new_list[0]
    assert the_list[1] is new_list[1]

    # BUT we should have a 'new' list object.
    assert the_list is not new_list


def test_dry_list_3():
    # Create a repo
    repo = DryRepo()

    print("Creating objects")
    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")

    print("Add objects to repo")
    # Add them to the repo
    repo.add_object(obj1)
    repo.add_object(obj2)

    print("create list object")
    # Create a list object
    the_list = DryList(obj1, obj2)

    print("add list object to repo")
    # Add that to the repo
    repo.add_object(the_list)

    print("Build new list object")
    # Create a new list object from a definition, but use a repo
    new_list = the_list.definition().build(repo=repo)

    # The list object should've extracted the objects themselves
    # From the repo instead of constructing them
    assert the_list[0] is new_list[0]
    assert the_list[1] is new_list[1]

    # BUT we should not have a 'new' list object.
    assert the_list is new_list
