import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy
