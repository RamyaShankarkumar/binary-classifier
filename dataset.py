from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_datasets():
    data = load_breast_cancer()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    train_dataset = CancerDataset(X_train, y_train)
    val_dataset = CancerDataset(X_val, y_val)

    return train_dataset, val_dataset, X_train.shape[1]
