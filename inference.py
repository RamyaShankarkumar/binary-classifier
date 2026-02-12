import torch
import torch.nn as nn

# ------------------------------
# 1️⃣ Define the SAME model architecture
# ------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)  # adjust input size if different
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)  # adjust output size if different

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ------------------------------
# 2️⃣ Load trained model
# ------------------------------
model = MyModel()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
print("Model loaded successfully!")

# ------------------------------
# 3️⃣ Sample inputs for demo inference
# ------------------------------
sample_inputs = torch.tensor([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [5.0, 3.6, 1.4, 0.2]
], dtype=torch.float32)

# ------------------------------
# 4️⃣ Run inference
# ------------------------------
with torch.no_grad():
    outputs = model(sample_inputs)
    predictions = torch.argmax(outputs, dim=1)

print("Raw outputs:\n", outputs)
print("Predicted classes:", predictions.tolist())
