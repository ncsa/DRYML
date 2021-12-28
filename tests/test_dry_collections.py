import pytest
import os
from dryml import DryList, DryRepo, load_object
import objects


def test_dry_list_1():
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")
    the_list = DryList(obj1, obj2)

    new_list = the_list.definition().build()
    assert new_list[0].definition() == obj1.definition()
    assert new_list[1].definition() == obj2.definition()
    assert new_list is not the_list
    assert new_list[0] is not obj1
    assert new_list[1] is not obj2


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

    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")

    # Add them to the repo
    repo.add_object(obj1)
    repo.add_object(obj2)

    # Create a list object
    the_list = DryList(obj1, obj2)

    # Add that to the repo
    repo.add_object(the_list)

    # Create a new list object from a definition, but use a repo
    new_list = the_list.definition().build(repo=repo)

    # The list object should've extracted the objects themselves
    # From the repo instead of constructing them
    assert the_list[0] is new_list[0]
    assert the_list[1] is new_list[1]

    # BUT we should not have a 'new' list object.
    assert the_list is new_list


@pytest.mark.usefixtures("create_temp_dir")
def test_dry_list_4(create_temp_dir):
    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj1.save_self(os.path.join(create_temp_dir, 'obj1.dry'))
    obj2 = objects.HelloStr(msg="a test")
    obj2.save_self(os.path.join(create_temp_dir, 'obj2.dry'))

    # Create a list object
    the_list = DryList(obj1, obj2)

    # Save the list object
    the_list.save_self(os.path.join(create_temp_dir, 'obj_list.dry'))

    # Load list from disk without a repo
    new_list = load_object(os.path.join(create_temp_dir, 'obj_list.dry'))

    assert len(the_list) == len(new_list)
    assert the_list[0].definition() == new_list[0].definition()
    assert the_list[1].definition() == new_list[1].definition()
    assert the_list[0] is not new_list[0]
    assert the_list[1] is not new_list[1]


@pytest.mark.usefixtures("create_temp_dir")
def test_dry_list_5(create_temp_dir):
    # Create repo
    repo = DryRepo(create_temp_dir)

    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg="a test")

    # Add objects to the repo
    repo.add_object(obj1)
    repo.add_object(obj2)

    # Create a list object
    the_list = DryList(obj1, obj2)

    # Add the list to the repo
    repo.add_object(the_list, filepath="obj_list.dry")

    # Save objects
    repo.save()

    new_list = load_object(os.path.join(create_temp_dir, 'obj_list.dry'),
                           repo=repo)

    assert the_list is new_list


@pytest.mark.usefixtures("create_temp_dir")
def test_dry_list_6(create_temp_dir):
    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg='a test')

    list_1 = DryList(obj1)
    list_2 = DryList(obj2, list_1)

    list_2.save_self(os.path.join(create_temp_dir, 'list.dry'))

    loaded_list = load_object(os.path.join(create_temp_dir, 'list.dry'))

    assert list_2.definition() == loaded_list.definition()


@pytest.mark.usefixtures("create_temp_dir")
def test_dry_list_7(create_temp_dir):
    # Create Repo
    repo = DryRepo(create_temp_dir)

    # Create objects
    obj1 = objects.HelloInt(msg=5)
    obj2 = objects.HelloStr(msg='a test')

    # Add objects to Repo
    repo.add_object(obj1)
    repo.add_object(obj2)

    # Create lists
    list_1 = DryList(obj1)

    # Add list to repo
    repo.add_object(list_1)

    # Create second list, and save it
    list_2 = DryList(obj2, list_1)
    list_2.save_self(os.path.join(create_temp_dir, 'list.dry'))

    loaded_list = load_object(os.path.join(create_temp_dir, 'list.dry'),
                              repo=repo)

    assert list_2.definition() == loaded_list.definition()
    assert list_2 is not loaded_list
    assert list_2[0] is loaded_list[0]
    assert list_2[1] is loaded_list[1]
