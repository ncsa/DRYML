import dryml
import objects
import mc


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


def test_train_supervised_model_sklearn_1():
    # Generate data.
    num_train = 2000
    num_test = 500
    train_data = mc.gen_dataset(num_examples=num_train)
    test_data = mc.gen_dataset(num_examples=num_test)

    import dryml.models.sklearn
    import sklearn.neighbors

    # Create model
    model = dryml.models.sklearn.SklearnTrainable(
        model=dryml.models.sklearn.SklearnRegressionModel(
            sklearn.neighbors.KNeighborsRegressor,
            n_neighbors=5,
            weights='uniform',
            algorithm='ball_tree'),
        train_fn=dryml.models.sklearn.SklearnBasicTraining())

    def test_model(test_data, model):
        import numpy as np
        test_ds = dryml.data.NumpyDataset(test_data, supervised=True)

        eval_data = model.eval(test_ds)

        eval_data = eval_data.batch(batch_size=32).numpy()

        total_loss = 0.
        num_examples = 0
        for batch_e_y, batch_y in eval_data:
            total_loss += np.sum((batch_e_y-batch_y)**2)
            num_examples += batch_e_y.shape[0]

        return total_loss/num_examples

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
        model.train(train_ds)

        return test_model(test_data, model)

    @dryml.compute_context(ctx_use_existing_context=False)
    def test_method(test_data, model):
        return test_model(test_data, model)

    train_loss = train_method(train_data, test_data, model)
    test_loss = test_method(test_data, model)

    assert train_loss == test_loss
