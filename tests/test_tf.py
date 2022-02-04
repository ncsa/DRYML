import pytest
import dryml
import objects_tf


def test_tfbase_1():
    """Test save/reload tfbase type objects"""

    test_obj = objects_tf.TestTF1()

    # Attempt to change class of the object.
    test_obj2 = dryml.change_object_cls(test_obj, objects_tf.TestTF1)

    assert test_obj2.definition() == test_obj.definition()


@pytest.mark.usefixtures("create_temp_named_file")
def test_tfbase_2(create_temp_named_file):
    """Test saving over an existing file"""

    test_obj = objects_tf.TestTF1()

    # Attempt to save model to file twice
    test_obj.save_self(create_temp_named_file)
    test_obj.save_self(create_temp_named_file)
