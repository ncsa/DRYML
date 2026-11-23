import dryml
from dryml import Repo

dryml.context.set_context({
    'default': {},
    'torch': {'gpu/0': 1.},
    'tf': {}
})

from dryml.models import Pipe

from dryml.examples.sentiment_analysis.workshop import train as train_model

if __name__ == "__main__":
    from dryml.models.torch.text import TextVectorizer
    from dryml.examples.sentiment_analysis.torch.models import SentimentTorchModel
    from dryml.models.torch.generic import BasicTraining, TorchOptimizer, Trainable as TorchTrainable
    from dryml.models.torch.generic import Wrapper
    import torch.nn as nn
    import torch

    torch_vectorizer = TextVectorizer(max_tokens=10000, sequence_length=250, dry_id="imdb_vectorizer")

    torch_optimizer = TorchOptimizer(torch.optim.Adam, SentimentTorchModel, lr=0.001)
    loss_fn = Wrapper(nn.BCELoss)

    sentiment_torch_trainable = TorchTrainable(
        model=SentimentTorchModel,
        train_fn=BasicTraining(epochs=10, optimizer=torch_optimizer, loss=loss_fn)
    )
    
    from dryml.data.torch.transforms import TorchDevice
    from dryml.context import context

    dev = context().get_torch_devices()[0]

    torch_pipe = Pipe(torch_vectorizer, TorchDevice(device=dev), sentiment_torch_trainable)
    trained_torch_pipe = train_model(torch_pipe)
    
    repo = Repo(directory="sentiment_models", create=True)
    repo.add_object(trained_torch_pipe, add_nested=False)
    repo.save()


    print("Torch model and all contained sub-objects saved under ./sentiment_models/")
