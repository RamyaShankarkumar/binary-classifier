import torch
import torch.nn as nn
import numpy as np

# 1️⃣ Define the SAME model architecture
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2️⃣ Load model
model = MyModel()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

print("Model loaded successfully!")

# 3️⃣ Create sample input
sample = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)

# 4️⃣ Run inference
with torch.no_grad():
    output = model(sample)
    prediction = torch.argmax(output, dim=1)

print("Raw output:", output)
print("Predicted class:", prediction.item())
