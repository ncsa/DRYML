import torch
import torch.nn as nn
import torch.nn.functional as F
from dryml.models.torch.generic import ModelWrapper

class SentimentModelTorch(nn.Module):
    def __init__(self, vocab_size=10000, max_length=250, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, embedding_dim))
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  
        x = F.relu(self.conv(x))
        x = torch.max(x, dim=2).values  
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

SentimentTorchModel = ModelWrapper(SentimentModelTorch, vocab_size=10000, max_length=250, embedding_dim=32)
