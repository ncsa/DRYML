import pytest
import core2_objects as objects
from dryml.core2.repo import Repo

@pytest.mark.usefixtures("create_temp_dir")
def test_context_1(create_temp_dir):
    from dryml.context.new.process import compute_context

    # First create an object and save it.
    obj1 = objects.TestDefer2('a')
    obj1.set_val(10)

    # Save the object to the temp directory
    repo = Repo(create_temp_dir)
    repo.save_object(obj1)

    # Unload the object
    obj1.__unload__()
    assert not obj1.__initialized__

    # Create a function to edit the value
    @compute_context(ctx_update_objs=True)
    def edit_value(obj):
        obj.set_val(20)

    edit_value(obj1, call_repo=create_temp_dir)

    assert obj1.data == 20
    assert obj1.__initialized__
