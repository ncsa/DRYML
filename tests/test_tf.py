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


@pytest.mark.usefixtures("ray_server")
def test_context_object_compute_check_tf_1():
    """
    Checks whether Eval behaves correctly when transformed to numpy
    """
    import ray
    import numpy as np
    import dryml.models.tf
    import tensorflow as tf

    num_classes = 5
    num_dims = 10

    model_def = dryml.DryObjectDef(
        dryml.models.tf.keras.Trainable,
        model=dryml.DryObjectDef(
            dryml.models.tf.keras.SequentialFunctionalModel,
            input_shape=(num_dims,),
            layer_defs=[
                ('Dense', {'units': 32, 'activation': 'relu'}),
                ('Dense', {'units': 32, 'activation': 'relu'}),
                ('Dense', {'units': num_classes, 'activation': 'softmax'}),
            ]),
        optimizer=dryml.DryObjectDef(
            dryml.models.tf.ObjectWrapper,
            tf.keras.optimizers.Adam),
        loss=dryml.DryObjectDef(
            dryml.models.tf.ObjectWrapper,
            tf.keras.losses.SparseCategoricalCrossentropy),
        train_fn=dryml.DryObjectDef(
            dryml.models.tf.keras.BasicTraining,
            epochs=1))

    # Create data for eval
    data = np.random.random((20, num_dims))

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def test_method(model_def, data):
        model = model_def.build()

        import dryml.data
        ds = dryml.data.NumpyDataset(data)

        with dryml.context.ContextManager({'tf': {}}):
            result = model.eval(ds.batch()).numpy().peek()
            assert type(result) is np.ndarray

    ray.get(test_method.remote(model_def, data))


@pytest.mark.usefixtures("ray_server")
@pytest.mark.usefixtures("create_name")
def test_context_object_compute_check_tf_2(create_name):
    import ray
    import numpy as np
    import dryml.models.tf
    import tensorflow as tf

    num_classes = 5
    num_dims = 10

    model_def = dryml.DryObjectDef(
        dryml.models.tf.keras.Trainable,
        model=dryml.DryObjectDef(
            dryml.models.tf.keras.SequentialFunctionalModel,
            input_shape=(num_dims,),
            layer_defs=[
                ('Dense', {'units': 32, 'activation': 'relu'}),
                ('Dense', {'units': 32, 'activation': 'relu'}),
                ('Dense', {'units': num_classes, 'activation': 'softmax'}),
            ]),
        optimizer=dryml.DryObjectDef(
            dryml.models.tf.ObjectWrapper,
            tf.keras.optimizers.Adam),
        loss=dryml.DryObjectDef(
            dryml.models.tf.ObjectWrapper,
            tf.keras.losses.SparseCategoricalCrossentropy),
        train_fn=dryml.DryObjectDef(
            dryml.models.tf.keras.BasicTraining,
            epochs=1))

    # Create data for eval
    data = np.random.random((20, num_dims))

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def test_method_1(model_def, data, dest_name):
        model = model_def.build()
        # Just created, there should be no model object
        assert model.model.mdl is None
        assert model.__dry_compute_data__ is None

        import dryml.data
        ds = dryml.data.NumpyDataset(data)

        with dryml.context.ContextManager({'tf': {}}):
            result = model.eval(ds.batch()).numpy().peek()

            assert model.model.mdl is not None
            assert model.__dry_compute_data__ is None

        assert model.model.mdl is None
        assert model.__dry_compute_data__ is not None

        model.save_self(dest_name)

        return result

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def test_method_2(data, dest_name):
        import dryml
        import dryml.models.tf
        model = dryml.load_object(dest_name)
        assert type(model) is dryml.models.tf.keras.Trainable

        # Just created, there should be no model object
        assert model.model.mdl is None
        # But we saved model data, so that should be there.
        assert model.__dry_compute_data__ is not None

        orig_compute_data = model.__dry_compute_data__

        import dryml.data
        ds = dryml.data.NumpyDataset(data)

        with dryml.context.ContextManager({'tf': {}}):
            result = model.eval(ds.batch()).numpy().peek()

            assert model.model.mdl is not None
            # We loaded the model into memory the
            # compute data should still be there though.
            assert model.__dry_compute_data__ is not None
            # It should also still be the same object
            assert id(orig_compute_data) == id(model.__dry_compute_data__)

        # We should have no model loaded anymore
        assert model.model.mdl is None
        assert model.__dry_compute_data__ is not None
        # Data should now be a different object. since we have to
        # write entirely new zip files
        assert id(orig_compute_data) != id(model.__dry_compute_data__)

        return result

    # Evaluate model and return result
    result_1 = ray.get(test_method_1.remote(model_def, data, create_name))
    result_2 = ray.get(test_method_2.remote(data, create_name))

    assert np.all(result_1 == result_2)
