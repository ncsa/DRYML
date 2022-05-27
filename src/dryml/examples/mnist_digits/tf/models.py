from dryml.models.tf import keras_sequential_functional_class

DigitsKeras = keras_sequential_functional_class(
    "DigitsKeras", (28, 28, 1), (10,))
