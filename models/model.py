import torch
import torch.nn as nn

class BinaryClassifieer(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # logits
        )

    def forward(self, x):
        return self.network(x)
