import dryml
import objects_tf


def test_tfbase_1():

    # Test save/reload tfbase type objects
    test_obj = objects_tf.TestTF1()

    # Attempt to change class of the object.
    test_obj2 = dryml.change_object_cls(test_obj, objects_tf.TestTF1)
