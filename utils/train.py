import torch
import torch.nn as nn
import torch.optim as optim
from models.model import MyModel

# Dummy dataset
X = torch.randn(100, 30)
y = torch.randint(0, 2, (100, 1)).float()

model = MyModel(input_size=30)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete")

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
