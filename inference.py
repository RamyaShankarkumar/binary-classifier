import torch
import pandas as pd
from models.model import MyModel

INPUT_SIZE = 30

# Load model
model = MyModel(input_size=INPUT_SIZE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Sample inference input (5 samples)
#sample_inputs = torch.randn(5, INPUT_SIZE)

# Sum > 0 → class 1
# Sum < 0 → class 0
sample_inputs = torch.tensor([
    [1.0]*30,   # sum = 30 → class 1
    [-1.0]*30,  # sum = -30 → class 0
    [0.5]*30,   # sum = 15 → class 1
    [-0.5]*30,  # sum = -15 → class 0
    [0.0]*30    # sum = 0 → borderline
])


with torch.no_grad():
    outputs = model(sample_inputs)
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities > 0.5).int()

# Save results to CSV
df = pd.DataFrame({
    "Probability": probabilities.squeeze().numpy(),
    "Prediction": predictions.squeeze().numpy()
})

df.to_csv("predictions.csv", index=False)

print("Inference completed.")
print(df)
