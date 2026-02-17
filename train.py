import torch
import torch.nn as nn
import torch.optim as optim
from models.model import MyModel

# Hyperparameters
INPUT_SIZE = 30
EPOCHS = 50
LR = 0.001

# Dummy dataset
X = torch.randn(200, INPUT_SIZE)
#y = torch.randint(0, 2, (200, 1)).float()
y = (X.sum(dim=1) > 0).float().unsqueeze(1)

model = MyModel(input_size=INPUT_SIZE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
