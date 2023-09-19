# DRYML

![Py 3.8 tests](https://github.com/ncsa/dryml/actions/workflows/test38.yaml/badge.svg)
![Py 3.9 tests](https://github.com/ncsa/dryml/actions/workflows/test39.yaml/badge.svg)
![Py 3.10 tests](https://github.com/ncsa/dryml/actions/workflows/test310.yaml/badge.svg)
![Py 3.11 tests](https://github.com/ncsa/dryml/actions/workflows/test311.yaml/badge.svg)
[![codecov](https://codecov.io/gh/ncsa/dryml/branch/main/graph/badge.svg?token=ELz0TSuOzo)](https://codecov.io/gh/ncsa/dryml)

Don't Repeat Yourself Machine Learning: A Machine Learning library to reduce code duplication, automate testing, perform hyper paramter searches, and ease model serialization.

DRYML aims to empower the Machine Learning practitioner to spend less time writing boilerplate code, and more time implementing new techniques. DRYML provides a model serialization framework along with serialization implementation for many common ML frameworks and model types, a framework for defining and training models on a specific problem, and a system to compare models from different ML frameworks on the same footing.

## DRYML Programming Philosophy

### Easy Object Serialization

All DRYML `Object`s can be uniquely identified, and serialized to disk. Once saved, you can load `Object`s directly from disk without having to first build a holding object as originally constructed. This allows for instance, Neural Net objects to be initialized with the correct parameters before loading the model weights without user intervention. Loading an `Object` consists of a single command `load_object`. Basic save/load logic is available for some major ML platforms, but the user is able and encouraged to implement new `Object`s with custom save/load methods and so is extendable to any ML system.

### Reuse Model Components

DRYML borrows from the Entity Component System programming pattern, and many Model types are created from components which are attached to the model, and can be reused. These can include training procedure, optimization algorithm, loss function, and the underlying NN model itself. This compartmentalization allows us to enable hyperparameter searches over nearly any parameter of your ML algorithm from the NN topology, from the optimizer learning rate, to the training procedure. It also allows the user to define a complex training procedure once, and then reuse it for multiple related problems in the future. DRYML also has defined training procedures for some common ML frameworks.

### Compare Models Between Frameworks

Many ML Problems can be tackled by different ML Frameworks. DRYML's API places all supported Frameworks on equal footing. All models output data in the form of DRYML Datasets. This means metrics on Datasets can be reused for models in different frameworks and models from different frameworks can be compared directly, allowing the ML practictioner to make decisions about which method is best for their problem. Models can also be easily chained together

### Allow Frameworks to work together

Modern ML frameworks such as tensorflow and pytorch are greedy about GPU use. DRYML implements a context system to enforce constraints on these frameworks when possible. The context system also provides a resource request API to allow the user to request the types of resources each framework is allowed to use. This allows elements from multiple frameworks to co-exist within the same data pipeline. For example, we can use a pytorch dataset and preprocessing with a tensorflow machine learning model.

## Bringing ML Frameworks together

The following ML Frameworks are currently supported, or planned to be supported

* Tensorflow (Initial support complete)
* Pytorch (Initial support complete)
* Sklearn (Initial support complete)
* XGBoost (Initial support complete)
* Statsmodels (support planned)

## DRYML Major Components

### DRYML `Object` and `ObjectDef`

The DRYML API provides the `Object` class which automatically implements all basic machinery for automatic object serialization. Any class you create which you want to serialize must inherit from the `Object` class. Here's a simple example:

```python
>>> from dryml import Object
>>> class Data(Object):
...     def __init__(self, data):
...         pass
... 
>>> data_obj = Data([1, 2, 3, 4, 5])
>>> data_obj.dry_args
([1, 2, 3, 4, 5],)
>>> data_obj.dry_kwargs
{'dry_id': 'e18d670c-b3b8-41b3-a941-3c2f7bf0b11e'}
```

We can see that `Object` gives the new `Data` class some extra powers! It remembers the arguments used to create it, and it receives a unique identifier (if you don't specify it yourself!). `data_obj` can also be easily serialized to disk with the `save_self` member method or `save_object` global method. Let's see that here:

```python
>>> from dryml import save_object, load_object
>>> save_object(data_obj, 'test.dry')
True
>>> new_obj = load_object('test.dry')
>>> new_obj
{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': ([1, 2, 3, 4, 5],), 'dry_kwargs': {'dry_id': 'e18d670c-b3b8-41b3-a941-3c2f7bf0b11e'}}
```

Now, why not just use `pickle` or `dill`? There is one major issue with that. `pickle` and `dill` tries to save every python object contained within the object you're trying to save. This will fail if your model object contains data which isn't supported by these major serialization platforms! Tensorflow tensors for example aren't supported.

If we want to add the ability for an `Object` to store an internal state, we need to implement the `save_object_imp` and `load_object_imp` methods as well.

`data_obj` has another ability too. It has the method `definition` which builds an `ObjectDef` object matching the arguments `data_obj` was constructed with. Let's take a quick look at that.

```python
>>> obj_def = data_obj.definition()
>>> obj_def
{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': ([1, 2, 3, 4, 5],), 'dry_kwargs': {'dry_id': 'e18d670c-b3b8-41b3-a941-3c2f7bf0b11e'}}
>>> new_obj_2 = obj_def.build()
>>> new_obj_2
{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': ([1, 2, 3, 4, 5],), 'dry_kwargs': {'dry_id': 'e18d670c-b3b8-41b3-a941-3c2f7bf0b11e'}}
>>> new_obj_2.dry_args
([1, 2, 3, 4, 5],)
```

So we can see that `ObjectDef` is a factory object creating objects matching the arguments used to initially construct `data_obj`!. 

We can create new `ObjectDef`s directly and use it to create new objects with different definitions! Let's look at that here:

```python
>>> from dryml import ObjectDef
>>> obj_def_2 = ObjectDef(Data, 3)
>>> obj_def_2
{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (3,), 'dry_kwargs': {}}
>>> test_obj_2 = obj_def_2.build()
>>> test_obj_2.dry_args
(3,)
```

This is great for creating copies of an object which contains internal data that is randomly initialized. We can for example, create many copies of the same neural network, train them, and see how well each network trains.


### DRYML `Repo` and `Selector`

A major issue working with many machine learning models is we want to try different things, which means different models and parameters. This can get unyieldy as the variety of models we're interested in trying gets larger. DRYML introduces the `Repo` and `Selector` types to help solve this issue. Any `Object` can be added to a `Repo`, and `Repo`s methods can be used to automate saving, loading, and application of a method across a collection of `Objects`. `Selector` is a type which can match properties of an `Object`. It's created similarly to `ObjectDef` and is a callable object. When passed an `ObjectDef` or `Object`, it indicates with a `bool` whether that object is compatible with the `Selector`. With `Selector`, `Repo` can return only those objects contained which match the `Selector`. Let's look at this now.

First, let's create a `Repo` and save a few objects.

```python
>>> repo = Repo(directory='./test', create=True)
>>> obj_def = ObjectDef(Data, 1)
>>> obj_def
{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (1,), 'dry_kwargs': {}}
>>> for i in range(5):
...     obj = obj_def.build()
...     repo.add_object(obj)
... 
>>> obj_def_2 = ObjectDef(Data, 2)
>>> for i in range(5):
...     obj = obj_def_2.build()
...     repo.add_object(obj)
... 
>>> len(repo)
10
```

Do now our repo has 10 objects, 5 of each type. Let's use a `Selector` to grab only those `Objects` with a 2.

```python
>>> repo.get(Selector(Data, obj_args=(2,)))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: __init__() got an unexpected keyword argument 'obj_args'
>>> repo.get(Selector(Data, args=(2,)))
[{'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (2,), 'dry_kwargs': {'dry_id': '99d796f9-6bc6-4341-947e-94b1b89a9ff3'}}, {'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (2,), 'dry_kwargs': {'dry_id': 'b9208924-8714-448a-b280-d63eefa758a7'}}, {'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (2,), 'dry_kwargs': {'dry_id': 'deb0b98b-6ec7-4a98-a98f-81cd0c9b3f3f'}}, {'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (2,), 'dry_kwargs': {'dry_id': '77561cf0-bdf5-4ae0-b7d8-9b9302544cd8'}}, {'cls': <class '__main__.Data'>, 'dry_mut': False, 'dry_args': (2,), 'dry_kwargs': {'dry_id': 'c256d4c2-45d0-495a-8b67-e29f4d5e824f'}}]
>>> len(repo.get(Selector(Data, args=(2,))))
5
```

And now we can work with the selected models!

### DRYML Dataset

The DRYML API provides the `Dataset` class which represents a machine learning dataset. It presents a number of useful methods for working with data, and also transformations to change datasets defined within major machine learning systems like `tensorflow` or `pytorch` into a more relevant framework or data type. We'll create a small Dataset here, and look at the `unbatch`, and `peek` methods.

```python
>>> import numpy as np
>>> num_examples = 32
>>> data_shape = (10, 10)
>>> data = np.random.random((num_examples,)+data_shape)
>>> data.shape
(32, 10, 10)
>>> from dryml.data import NumpyDataset
>>> data_ds = NumpyDataset(data)
>>> data_ds.peek().shape
(32, 10, 10)
>>> type(data_ds.peek())
<class 'numpy.ndarray'>
>>> type(data_ds.tf().unbatch().peek())
<class 'tensorflow.python.framework.ops.EagerTensor'>
>>> data_ds.tf().peek().shape
TensorShape([32, 10, 10])
>>> data_ds.tf().unbatch().peek().shape
TensorShape([10, 10])
>>> type(data_ds.torch().peek())
<class 'torch.Tensor'>
```

We can also see that `tf` turns the Dataset into a `TFDataset` which is backed by a `tf.data.Dataset`. Thus the elements retrievable become tensorflow `Tensor`s. Similarly, `torch` turns the `Dataset` into a `TorchDataset` which is backed by a `torch.utils.data.IterableDataset`.

### DRYML Context

The DRYML API provides a context system to manage the allocation of compute devices like GPUs. It also provides a decorator `dryml.context.compute_context` which can be applied to any method we want to launch in a separate process with a specific set of context requirements. This allows the user to prevent code which allocates memory on a device like a GPU from running unless you explicitly allow it. `Object` supports this as well with the `load_compute_imp` and `save_compute_imp` methods which manage an `Object`'s transition into and out of 'compute mode' in which an `Object` may allocate memory on a device. This is especially useful for situations when we may want to compare results from a `tensorflow` based model and a `pytorch` based model. We can wrap a method with `compute_context` and then request a context with requirements `{'tf': {}}` for the `tensorflow` model, and requirements `{'torch': {}}` for the `pytorch` model.

The user can specify the amount or even specific resources with a context requirement for example:

```python
{
    'tf': {'gpu/1': 0.5},
    'torch': {'gpu/1': 0.5},
}
```

Let's look at a simple example. We'll first define a method which loads a tensorflow dataset, but first checks whether the context can support it. It then returns the first element of that dataset. Then we'll decorate that method so we can execute it in a separate process with certain contexts. We'll then try a couple contexts and see what happens!

```python
>>> def test_method():
...     import dryml
...     import tensorflow_datasets as tfds
...     from dryml.data.tf import TFDataset
...     dryml.context.context_check({'tf': {}})
...     (test_ds,) = tfds.load(
...         'mnist',
...         split=['test'],
...         shuffle_files=True,
...         as_supervised=True)
...     test_ds = TFDataset(
...         test_ds,
...         supervised=True)
...     return test_ds.unbatch().numpy().peek()
...
>>> test_method = dryml.context.compute_context()(test_method)
>>> test_method()
Exception encountered in context thread! pid: 724396
Traceback (most recent call last):
  File "/data0/matthew/Software/NCSA/DRYML/src/dryml/context/process.py", line 34, in run
    super().run()
  File "/data0/matthew/Software/NCSA/DRYML/venv_dryml_dev/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/data0/matthew/Software/NCSA/DRYML/src/dryml/context/process.py", line 186, in __call__
    self.final_call(f, ctx_send_q, ctx_ret_q, *args, **kwargs)
  File "/data0/matthew/Software/NCSA/DRYML/src/dryml/context/process.py", line 129, in final_call
    res = f(*args, **kwargs)
  File "<stdin>", line 5, in test_method
  File "/data0/matthew/Software/NCSA/DRYML/src/dryml/context/context_tracker.py", line 420, in context_check
    raise ContextIncompatibilityError(
dryml.context.context_tracker.ContextIncompatibilityError: Context doesn't satisfy requirements {'tf': {}}

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/data0/matthew/Software/NCSA/DRYML/src/dryml/context/process.py", line 370, in wrapped_func
    raise e
dryml.context.context_tracker.ContextIncompatibilityError: Context doesn't satisfy requirements {'tf': {}}
>>> x, y = test_method(call_context_reqs={'tf': {}})
2022-10-06 15:27:33.318424: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
>>> x.shape
(28, 28, 1)
>>> y.shape
()
```
