import pytest
import dryml

import objects


@pytest.mark.usefixtures("create_temp_dir")
def test_compound_object_save_restore_1(create_temp_dir):
    # First create a definition object
    obj_def = dryml.ObjectDef(
        objects.TestNest,
        dryml.ObjectDef(
            objects.TestNest2,
            A=3
        )
    )

    # Next, create a compound object
    obj = obj_def.build()

    # Create object repo
    repo = dryml.Repo(directory=create_temp_dir)

    # Add object to the repo
    repo.add_object(obj)

    # Check that there are now two objects stored.
    assert len(repo) == 2

    # Save the objects
    repo.save()

    # Delete repo
    del repo

    # Recreate repo object
    repo = dryml.Repo(directory=create_temp_dir)

    # Check there are still two objects
    assert len(repo) == 2

    # Use original definition as a selector to grab the constructed object
    obj_ret = repo.get(selector=obj_def)

    assert type(obj_ret) is not list

    # Check new fetched object is the same as originally created object
    assert obj_ret.dry_id == obj.dry_id
    assert obj_ret.A.dry_id == obj.A.dry_id
    assert obj_ret.A.A == obj.A.A
