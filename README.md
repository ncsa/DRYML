# DRYML

![Py 3.7 tests](https://github.com/ncsa/dryml/actions/workflows/test37.yaml/badge.svg)
![Py 3.8 tests](https://github.com/ncsa/dryml/actions/workflows/test38.yaml/badge.svg)
![Py 3.9 tests](https://github.com/ncsa/dryml/actions/workflows/test39.yaml/badge.svg)
![Py 3.10 tests](https://github.com/ncsa/dryml/actions/workflows/test310.yaml/badge.svg)
[![codecov](https://codecov.io/gh/ncsa/dryml/branch/main/graph/badge.svg?token=ELz0TSuOzo)](https://codecov.io/gh/ncsa/dryml)

Don't Repeat Yourself Machine Learning: A Machine Learning library to reduce code duplication, automate testing, perform hyper paramter searches, and ease model serialization.

DRYML aims to empower the Machine Learning practitioner to spend less time writing boilerplate code, and more time implementing new techniques to push ML forward. DRYML provides a model serialization framework along with serialization implementation for many common ML frameworks and model types, a framework for defining and training models on a specific problem, and a system to compare models from different ML frameworks on the same footing.

## DRYML Programming Philosophy

### Easy Object Serialization

Nearly all objects within DRYML can be uniquely identified, and serialized to disk. Once saved, you can load objects directly from disk without having to build Neural Net objects in their entirety initialized with the correct parameters before loading the model weights. The serialization format stores all needed parameters so model objects can be automatically constructed without user intervention, and then weights can then be loaded directly. Loading a model consists of a single command. Basic save/load logic is available for all major ML platforms, but the user is able to simply implement new objects with custom save/load methods and so is extendable to any ML system.

### Reuse Model Components

DRYML borrows much from the Entity Component System, and many Model types are used by creating components which are attached to the model. These can include training procedure, optimization algorithm, loss function, and the underlying NN model itself. This compartmentalization allows us to enable hyperparameter searches over nearly any parameter of your ML algorithm from the NN topology, to the optimizer learning rate, to the training procedure. It also allows the user to define a complex training procedure once, and then reuse it for multiple related problems in the future. DRYML also has defined common training procedures for some common ML frameworks

### Compare Models Between Frameworks

Many ML Problems can be tackled by different ML Frameworks. DRYML's API places all supported Frameworks on equal footing. Models from different frameworks can be compared directly, allowing the ML practictioner to make decisions about which method is best for their problem.

## Bringing ML Frameworks together

The following ML Frameworks are currently supported, or planned to be supported

* Tensorflow (Initial support complete)
* Pytorch (Initial support complete)
* Sklearn (Initial support complete)
* XGBoost (Initial support complete)
* Statsmodels (support planned)
