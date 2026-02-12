import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=30):
        super(MyModel, self).__init__()

        self.input_size = input_size

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)
