import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import get_datasets
from models.model import BinaryClassifier
from utils.train import train_one_epoch
from utils.eval import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, input_dim = get_datasets()

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    model = BinaryClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved as best_model.pth")


    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), Config.model_path)
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
