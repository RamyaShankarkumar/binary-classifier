import torch
import torch.nn as nn
import pandas as pd

# ------------------------------
# 1️⃣ Define the SAME model architecture
# ------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 16),  # input size = 4 (adjust if different)
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

    def forward(self, x):
        return self.network(x)

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

# ------------------------------
# 5️⃣ Save predictions as CSV (artifact)
# ------------------------------
df = pd.DataFrame({
    "sample_index": range(len(sample_inputs)),
    "predicted_class": predictions.tolist()
})
df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")
