from dryml.models.tf.keras import keras_sequential_functional_class

DigitsKeras = keras_sequential_functional_class(
    "DigitsKeras", (28, 28, 1), (10,))
