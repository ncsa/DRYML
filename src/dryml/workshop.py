class Workshop(object):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def data_prep(self):
        # A 
        return

# toy usage:
#
# Create workshop object
# shop = Workshop()
#
# Load models related to this workshop from a directory
# Options to restrict?
# shop.load_models(directory, **kwargs)
# OR create new models and add them to the workshop
# for model in models:
#     shop.add_model(model)
#
# Method where central data repository is initialized
# shop.data_prep()
#
# Train models (with training options??
# shop.train_models(**kwargs)
#
# Measure model performance
# shop.model_performance()
