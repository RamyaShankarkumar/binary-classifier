import torch
import pandas as pd
from models.model import MyModel

INPUT_SIZE = 30

# Load model
model = MyModel(input_size=INPUT_SIZE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Sample inference input (5 samples)
sample_inputs = torch.randn(5, INPUT_SIZE)

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
