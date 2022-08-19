import dryml
import objects
import mc
import numpy as np
import pickle
import pytest


def test_compute_context_consistency_1():
    obj = objects.TestClassE()

    assert obj.__dry_compute_data__ is None

    @dryml.compute
    def func1(obj):
        obj.set_val(20)

    func1(obj)

    assert obj.__dry_compute_data__ is None


def test_compute_context_consistency_2():
    obj = objects.TestClassE()

    assert obj.__dry_compute_data__ is None

    @dryml.compute_context(ctx_update_objs=True)
    def func1(obj):
        obj.set_val(20)

    func1(obj)

    assert obj.__dry_compute_data__ is not None

    @dryml.compute
    def func2(obj):
        assert obj.val == 20


# Model and definition generators


def sklearn_regressor(definition=False):
    def gen():
        try:
            import dryml.models.sklearn
            import sklearn.neighbors
        except ImportError:
            pytest.skip("sklearn not supported")

        # Create model
        model = dryml.models.sklearn.Trainable(
            model=dryml.models.sklearn.RegressionModel(
                sklearn.neighbors.KNeighborsRegressor,
                n_neighbors=5,
                weights='uniform',
                algorithm='ball_tree'),
            train_fn=dryml.models.sklearn.BasicTraining())

        if definition:
            return model.definition().get_cat_def(recursive=True)
        else:
            return model
    return gen


def tf_regressor(definition=False):
    def gen():
        try:
            import dryml.models.tf
            import tensorflow as tf
        except ImportError:
            pytest.skip("tensorflow not supported")

        # Create model
        model = dryml.models.tf.keras.Trainable(
            model=dryml.models.tf.keras.SequentialFunctionalModel(
                input_shape=(1,),
                layer_defs=[
                    ('Dense', {'units': 32, 'activation': 'relu'}),
                    ('Dense', {'units': 32, 'activation': 'relu'}),
                    ('Dense', {'units': 1, 'activation': 'linear'})]),
            optimizer=dryml.models.tf.ObjectWrapper(tf.keras.optimizers.Adam),
            loss=dryml.models.tf.ObjectWrapper(
                tf.keras.losses.MeanSquaredError),
            train_fn=dryml.models.tf.keras.BasicTraining(epochs=1),
            metrics=[
                dryml.models.tf.ObjectWrapper(
                    tf.keras.metrics.MeanSquaredError)
            ])

        if definition:
            return model.definition().get_cat_def(recursive=True)
        else:
            return model
    return gen


def sklearn_classifier(definition=False):
    def gen(num_dims, num_classes):
        try:
            import dryml.models.sklearn
            import sklearn.neighbors
            import dryml.data.transforms
        except ImportError:
            pytest.skip("sklearn not available")

        model = dryml.models.sklearn.Trainable(
            model=dryml.models.sklearn.ClassifierModel(
                sklearn.neighbors.KNeighborsClassifier,
                n_neighbors=5,
                algorithm='ball_tree'),
            train_fn=dryml.models.sklearn.BasicTraining())

        best_cat = dryml.data.transforms.BestCat()

        model = dryml.models.DryPipe(model, best_cat)

        if definition:
            return model.definition().get_cat_def(recursive=True)
        else:
            return model
    return gen


def tf_classifier(definition=False):
    def gen(num_dims, num_classes):
        try:
            import dryml.models.tf
            import dryml.data.transforms
            import dryml.data.tf.transforms
            import tensorflow as tf
        except ImportError:
            pytest.skip("tf not available")

        model = dryml.models.tf.keras.Trainable(
            model=dryml.models.tf.keras.SequentialFunctionalModel(
                input_shape=(num_dims,),
                layer_defs=[
                    ('Dense', {'units': 32, 'activation': 'relu'}),
                    ('Dense', {'units': 32, 'activation': 'relu'}),
                    ('Dense', {'units': num_classes, 'activation': 'softmax'}),
                ]),
            optimizer=dryml.models.tf.ObjectWrapper(tf.keras.optimizers.Adam),
            loss=dryml.models.tf.ObjectWrapper(
                tf.keras.losses.SparseCategoricalCrossentropy),
            train_fn=dryml.models.tf.keras.BasicTraining(epochs=1))

        best_cat = dryml.data.tf.transforms.BestCat()

        model = dryml.models.DryPipe(model, best_cat)

        if definition:
            return model.definition().get_cat_def(recursive=True)
        else:
            return model
    return gen


@pytest.mark.parametrize(
    "model_gen",
    [sklearn_regressor(definition=False),
     tf_regressor(definition=False)])
def test_train_supervised_1_context(model_gen):
    # Generate the model
    model = model_gen()

    # Generate data.
    num_train = 2000
    num_test = 500
    train_data = mc.gen_dataset_1(num_examples=num_train)
    test_data = mc.gen_dataset_1(num_examples=num_test)

    def test_model(test_data, model):
        import dryml.metrics

        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.mean_squared_error(model, test_ds)

    @dryml.compute_context(
        ctx_use_existing_context=False,
        ctx_update_objs=True)
    def train_method(train_data, test_data, model):
        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            supervised=True)

        # Train model
        model.prep_train()
        model.train(train_ds)

        return test_model(test_data, model)

    @dryml.compute_context(ctx_use_existing_context=False)
    def test_method(test_data, model):
        return test_model(test_data, model)

    train_loss = train_method(train_data, test_data, model)
    test_loss = test_method(test_data, model)

    assert train_loss == test_loss


@pytest.mark.usefixtures("create_temp_dir")
@pytest.mark.parametrize(
    "model_gen",
    [sklearn_regressor(definition=True),
     tf_regressor(definition=True)])
def test_train_supervised_1_context_repo(create_temp_dir, model_gen):
    # Generate data.
    num_train = 2000
    num_test = 500
    train_data = mc.gen_dataset_1(num_examples=num_train)
    test_data = mc.gen_dataset_1(num_examples=num_test)

    model_def = model_gen()

    def test_model(test_data, model):
        import dryml.metrics
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.mean_squared_error(model, test_ds)

    model_reqs = model_def.build().dry_context_requirements()

    @dryml.compute_context(
        ctx_context_reqs=model_reqs,
        ctx_use_existing_context=False,
        ctx_update_objs=True)
    def train_method(train_data, test_data, model_def, work_dir):
        # Create repo
        import dryml
        repo = dryml.DryRepo(directory=work_dir)

        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            batch_size=num_train,
            supervised=True)

        # Build and fetch model
        model = model_def.build(repo=repo)

        # Train model
        model.prep_train()
        model.train(train_ds)

        # Save model to repo.
        repo.save(model)

        return test_model(test_data, model)

    @dryml.compute_context(
        ctx_context_reqs=model_reqs,
        ctx_use_existing_context=False)
    def test_method(test_data, model_def, work_dir):
        import dryml
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Fetch model based on definition
        model = dryml.utils.head(repo.get(model_def, build_missing_def=False))

        return test_model(test_data, model)

    train_loss = train_method(
        train_data, test_data, model_def, create_temp_dir)
    test_loss = test_method(
        test_data, model_def, create_temp_dir)

    assert train_loss == test_loss


@pytest.mark.usefixtures("create_temp_dir")
@pytest.mark.usefixtures("ray_server")
@pytest.mark.parametrize(
    "model_gen",
    [sklearn_regressor(definition=True),
     tf_regressor(definition=True)])
def test_train_supervised_1_ray(create_temp_dir, model_gen):
    import ray

    # Generate data.
    num_train = 2000
    num_test = 500
    train_data = mc.gen_dataset_1(num_examples=num_train)
    test_data = mc.gen_dataset_1(num_examples=num_test)

    model_def = model_gen()

    def test_model(test_data, model):
        import dryml.metrics
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.mean_squared_error(model, test_ds)

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def train_method(train_data, test_data, model_def, work_dir):
        # Create repo
        import dryml
        repo = dryml.DryRepo(directory=work_dir)

        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            batch_size=num_train,
            supervised=True)

        # Build and fetch model
        model = model_def.build(repo=repo)

        # Create context
        with dryml.context.ContextManager(model.dry_context_requirements()):
            # Train model
            model.prep_train()
            model.train(train_ds)

            acc = test_model(test_data, model)

        # Save model to repo.
        repo.save(model)

        return acc

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def test_method(test_data, model_def, work_dir):
        import dryml
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Fetch model based on definition
        model = dryml.utils.head(repo.get(model_def, build_missing_def=False))

        with dryml.context.ContextManager(model.dry_context_requirements()):
            acc = test_model(test_data, model)

        return acc

    train_loss = ray.get(train_method.remote(
        train_data, test_data, model_def, create_temp_dir))
    test_loss = ray.get(test_method.remote(
        test_data, model_def, create_temp_dir))

    assert train_loss == test_loss


@pytest.mark.parametrize(
    "model_gen",
    [sklearn_classifier(definition=False),
     tf_classifier(definition=False)])
def test_train_supervised_2_context(model_gen):
    num_classes = 10
    num_dims = 2

    # generate the model
    model = model_gen(num_dims, num_classes)

    # Generate classes
    L = -10.
    H = 10.
    Max_W = 5.
    centers = (np.random.random((num_classes, num_dims))*(H-L))+L
    widths = np.random.random((num_classes, num_dims))*Max_W

    num_train = 5000
    train_data = mc.gen_dataset_2(
        num_examples=num_train,
        centers=centers,
        widths=widths)

    num_test = 100
    test_data = mc.gen_dataset_2(
        num_examples=num_test,
        centers=centers,
        widths=widths)

    def test_model(test_data, model):
        import dryml.metrics
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.categorical_accuracy(model, test_ds)

    @dryml.compute_context(
        ctx_use_existing_context=False,
        ctx_update_objs=True)
    def train_method(train_data, test_data, model):
        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            batch_size=num_train,
            supervised=True)

        # Train model
        model.prep_train()
        model.train(train_ds)

        return test_model(test_data, model)

    @dryml.compute_context(ctx_use_existing_context=False)
    def test_method(test_data, model):
        return test_model(test_data, model)

    random_guessing_accuracy = 1./num_classes

    train_accuracy = train_method(train_data, test_data, model)
    test_accuracy = test_method(test_data, model)

    assert train_accuracy == test_accuracy
    assert train_accuracy > random_guessing_accuracy


@pytest.mark.usefixtures("create_temp_dir")
@pytest.mark.parametrize(
    "model_gen",
    [sklearn_classifier(definition=True),
     tf_classifier(definition=True)])
def test_train_supervised_2_context_repo(create_temp_dir, model_gen):
    num_classes = 10
    num_dims = 2

    model_def = model_gen(num_dims, num_classes)

    # Generate classes
    L = -10.
    H = 10.
    Max_W = 5.
    centers = (np.random.random((num_classes, num_dims))*(H-L))+L
    widths = np.random.random((num_classes, num_dims))*Max_W

    num_train = 5000
    train_data = mc.gen_dataset_2(
        num_examples=num_train,
        centers=centers,
        widths=widths)

    num_test = 100
    test_data = mc.gen_dataset_2(
        num_examples=num_test,
        centers=centers,
        widths=widths)

    def test_model(test_data, model):
        import dryml.metrics
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.categorical_accuracy(model, test_ds)

    model_reqs = model_def.build().dry_context_requirements()

    @dryml.compute_context(
        ctx_context_reqs=model_reqs,
        ctx_use_existing_context=False,
        ctx_update_objs=True)
    def train_method(train_data, test_data, model_def, work_dir):
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Build model from definition
        model = model_def.build(repo=repo)

        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            batch_size=num_train,
            supervised=True)

        # Train model
        model.prep_train()
        model.train(train_ds)

        # Save model
        repo.save(model)

        return test_model(test_data, model)

    @dryml.compute_context(
        ctx_context_reqs=model_reqs,
        ctx_use_existing_context=False)
    def test_method(test_data, model_def, work_dir):
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Get model from the repo
        model = repo.get(model_def, build_missing_def=False)

        # Run test
        return test_model(test_data, model)

    random_guessing_accuracy = 1./num_classes

    train_accuracy = train_method(
        train_data, test_data, model_def, create_temp_dir)
    test_accuracy = test_method(
        test_data, model_def, create_temp_dir)

    assert train_accuracy == test_accuracy
    assert train_accuracy > random_guessing_accuracy


@pytest.mark.usefixtures("create_temp_dir")
@pytest.mark.usefixtures("ray_server")
@pytest.mark.parametrize(
    "model_gen",
    [sklearn_classifier(definition=True),
     tf_classifier(definition=True)])
def test_train_supervised_2_ray(create_temp_dir, model_gen):
    import ray

    num_classes = 10
    num_dims = 2

    model_def = model_gen(num_dims, num_classes)

    # Generate classes
    L = -10.
    H = 10.
    Max_W = 5.
    centers = (np.random.random((num_classes, num_dims))*(H-L))+L
    widths = np.random.random((num_classes, num_dims))*Max_W

    num_train = 5000
    train_data = mc.gen_dataset_2(
        num_examples=num_train,
        centers=centers,
        widths=widths)

    num_test = 100
    test_data = mc.gen_dataset_2(
        num_examples=num_test,
        centers=centers,
        widths=widths)

    def test_model(test_data, model):
        import dryml.metrics
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        return dryml.metrics.categorical_accuracy(model, test_ds)

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=1)
    def train_method(train_data, test_data, model_def, work_dir):
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Build model from definition
        model = model_def.build(repo=repo)

        # Create datasets
        train_ds = dryml.data.NumpyDataset(
            train_data,
            batch_size=num_train,
            supervised=True)

        with dryml.context.ContextManager(model.dry_context_requirements()):
            # Train model
            model.prep_train()
            model.train(train_ds)

            acc = test_model(test_data, model)

        # Save model
        repo.save(model)

        return acc

    @ray.remote(num_cpus=1, num_gpus=0, max_calls=0)
    def test_method(test_data, model_def, work_dir):
        # Create repo
        repo = dryml.DryRepo(directory=work_dir)

        # Get model from the repo
        model = repo.get(model_def, build_missing_def=False)

        # Run test
        with dryml.context.ContextManager(model.dry_context_requirements()):
            acc = test_model(test_data, model)

        return acc

    random_guessing_accuracy = 1./num_classes

    train_accuracy = ray.get(train_method.remote(
        train_data, test_data, model_def, create_temp_dir))
    test_accuracy = ray.get(test_method.remote(
        test_data, model_def, create_temp_dir))

    assert train_accuracy == test_accuracy
    assert train_accuracy > random_guessing_accuracy


@pytest.mark.usefixtures("create_temp_dir")
@pytest.mark.usefixtures("create_temp_named_file")
def test_train_test_pattern_1(create_temp_dir, create_temp_named_file):
    """
    Detect a situation, when an object is created accidentally
    with a 'parent' class.
    """

    from dryml import Workshop
    import dryml.data.transforms

    # First, generate problem data
    # Generate classes
    num_classes = 10
    num_dims = 2
    L = -10.
    H = 10.
    Max_W = 5.
    centers = (np.random.random((num_classes, num_dims))*(H-L))+L
    widths = np.random.random((num_classes, num_dims))*Max_W

    num_train = 5000
    train_data = mc.gen_dataset_2(
        num_examples=num_train,
        centers=centers,
        widths=widths)

    num_test = 100
    test_data = mc.gen_dataset_2(
        num_examples=num_test,
        centers=centers,
        widths=widths)

    # Save dataset to temp file so we can load it in subordinate processes
    with open(create_temp_named_file, 'wb') as f:
        f.write(pickle.dumps((train_data, test_data)))

    # Create workshop
    shop = Workshop(work_dir=create_temp_dir)

    # Create flatten step
    flattener = dryml.data.transforms.Flatten()
    shop.repo.add_object(flattener)

    # Define a function to create a model definition
    def build_model_def_1(
            shop, n_neighbors=2, algorithm='auto', num_examples=-1):
        import dryml.models.sklearn
        import sklearn.neighbors

        flattener = dryml.utils.head(shop.repo.get(
            selector=dryml.DryObjectDef(dryml.data.transforms.Flatten)))

        mdl_def = dryml.DryObjectDef(
            dryml.models.sklearn.Model,
            sklearn.neighbors.KNeighborsClassifier,
            n_neighbors=n_neighbors,
            algorithm=algorithm
            )

        mdl_def = dryml.DryObjectDef(
            dryml.models.sklearn.Trainable,
            model=mdl_def,
            train_fn=dryml.DryObjectDef(
                dryml.models.sklearn.BasicTraining,
                num_examples=num_examples))

        mdl_def = dryml.DryObjectDef(
            dryml.models.DryPipe,
            flattener,
            mdl_def,)

        return mdl_def

    # Define a function to create a model definition
    def build_model_def_2(
            shop, n_neighbors=2, algorithm='auto', num_examples=-1):
        import dryml.models.sklearn
        import sklearn.neighbors

        flattener = dryml.utils.head(shop.repo.get(
            selector=dryml.DryObjectDef(dryml.data.transforms.Flatten)))

        mdl_def = dryml.DryObjectDef(
            dryml.models.sklearn.ClassifierModel,
            sklearn.neighbors.KNeighborsClassifier,
            n_neighbors=n_neighbors,
            algorithm=algorithm
            )

        mdl_def = dryml.DryObjectDef(
            dryml.models.sklearn.Trainable,
            model=mdl_def,
            train_fn=dryml.DryObjectDef(
                dryml.models.sklearn.BasicTraining,
                num_examples=num_examples))

        mdl_def = dryml.DryObjectDef(
            dryml.models.DryPipe,
            flattener,
            mdl_def,)

        return mdl_def

    # Create definition
    model_def = build_model_def_1(shop, n_neighbors=2, algorithm='ball_tree')

    # Define function to train model
    @dryml.compute_context(ctx_context_reqs={'default': {}})
    def train_model(model_def, repo_dir, data_filepath):
        # Load data from file
        with open(data_filepath, 'rb') as f:
            (train_data, test_data) = pickle.loads(f.read())
        train_ds = dryml.data.NumpyDataset(train_data, supervised=True)

        # Create repo
        shop = Workshop(work_dir=repo_dir)

        # Build the object
        model_obj = model_def.build(repo=shop.repo)

        # Train model object
        model_obj.train(train_ds)

        # Save all objects
        shop.repo.save(model_obj)

    # Train the model
    train_model(model_def, create_temp_dir, create_temp_named_file)

    # load objects
    shop.repo.load_objects_from_directory()

    # Check we stored one object in the repo
    assert dryml.utils.count(
        shop.repo.get(model_def, build_missing_def=False)) == 1

    # Check we can't see the object in another way.
    model_def_2 = build_model_def_2(shop, n_neighbors=2, algorithm='ball_tree')
    try:
        assert dryml.utils.count(shop.repo.get(
            model_def_2,
            sel_kwargs={'verbosity': 10},
            build_missing_def=False)) == 0
    except KeyError:
        pass
