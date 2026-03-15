"""
Task 1: The Dynamics of Generalization
train.py - Downloads data, builds and trains baseline and regularized models, saves weights.

GenAI Usage Statement:
Claude (Anthropic) was used in an assistive role to help structure the code and draft
the technical analysis. All code was reviewed, understood, and verified by the student.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import os


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def get_data_loaders(batch_size=128, val_split=0.1):
    """
    Download Fashion-MNIST and create train, validation, and test loaders.

    Args:
        batch_size (int): Batch size for data loaders.
        val_split (float): Fraction of training data used for validation.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download training and test sets
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Split training into train and validation
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_train = num_train - num_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 2. Model Definitions
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """
    High-capacity deep neural network with 6 hidden layers and NO regularization.
    Input: flattened 28x28 = 784 features.
    Output: 10 classes (Fashion-MNIST).
    """

    def __init__(self):
        super(BaselineModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 10).
        """
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class DropoutModel(nn.Module):
    """
    Deep neural network with 6 hidden layers regularised using ONLY dropout.
    Same architecture as BaselineModel but with dropout applied after each
    hidden layer's activation. No batch normalisation, no weight decay.

    Dropout randomly zeros a fraction of activations during training, which:
    - Prevents co-adaptation of neurons (forces redundant representations)
    - Acts as an approximate ensemble of exponentially many sub-networks
    - Can be interpreted as Bayesian model sampling (Gal & Ghahramani, 2016)

    Input: flattened 28x28 = 784 features.
    Output: 10 classes (Fashion-MNIST).
    """

    def __init__(self, dropout_rate=0.4):
        """
        Args:
            dropout_rate (float): Probability of dropping a unit during training.
                A rate of 0.4 provides strong regularisation while preserving
                enough capacity for learning. At inference, all units are active
                and outputs are scaled accordingly by PyTorch automatically.
        """
        super(DropoutModel, self).__init__()
        self.flatten = nn.Flatten()

        # Same layer dimensions as baseline for fair comparison
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass through the dropout-regularised network.
        Dropout is applied after each hidden layer's ReLU activation.
        No dropout on the output layer (fc6) to preserve class logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 10).
        """
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.fc6(x)
        return x


# ---------------------------------------------------------------------------
# 3. Training and Evaluation
# ---------------------------------------------------------------------------

def compute_accuracy(model, data_loader):
    """
    Compute classification accuracy on a dataset.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): Data loader to evaluate on.

    Returns:
        float: Accuracy as a fraction in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=30):
    """
    Train a model and record training/validation accuracy per epoch.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function.
        num_epochs (int): Number of training epochs.

    Returns:
        train_accs (list of float): Training accuracy per epoch.
        val_accs (list of float): Validation accuracy per epoch.
    """
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Compute accuracies at end of epoch
        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch [{epoch+1:2d}/{num_epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.4f}  "
              f"Val Acc: {val_acc:.4f}")

    return train_accs, val_accs


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    """Main function: load data, train both models, save weights and history."""
    torch.manual_seed(42)
    num_epochs = 30

    print("=" * 60)
    print("Task 1: The Dynamics of Generalization")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=128)
    print(f"  Train size: {len(train_loader.dataset)}, "
          f"Val size: {len(val_loader.dataset)}, "
          f"Test size: {len(test_loader.dataset)}")

    # --- Baseline Model (no regularization, plain SGD) ---
    print("\n[2/4] Training Baseline Model (no regularization)...")
    print("  Optimizer: SGD, lr=0.01, momentum=0.9, no weight decay")
    baseline = BaselineModel()
    baseline_optimizer = torch.optim.SGD(
        baseline.parameters(), lr=0.01, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    baseline_train_accs, baseline_val_accs = train_model(
        baseline, train_loader, val_loader, baseline_optimizer, criterion,
        num_epochs=num_epochs
    )

    # --- Dropout Model (dropout only, no other regularisation) ---
    print("\n[3/4] Training Dropout Model (dropout only, rate=0.4)...")
    print("  Optimizer: SGD, lr=0.01, momentum=0.9, NO weight decay")
    print("  Dropout rate: 0.4, No batch normalisation")
    regularized = DropoutModel(dropout_rate=0.4)
    reg_optimizer = torch.optim.SGD(
        regularized.parameters(), lr=0.01, momentum=0.9
    )

    reg_train_accs, reg_val_accs = train_model(
        regularized, train_loader, val_loader, reg_optimizer, criterion,
        num_epochs=num_epochs
    )

    # Save models
    print("\n[4/4] Saving models and training history...")
    torch.save(baseline.state_dict(), "baseline_model.pth")
    torch.save(regularized.state_dict(), "regularized_model.pth")

    # Save training history as JSON for task.py to load
    history = {
        "baseline_train_accs": baseline_train_accs,
        "baseline_val_accs": baseline_val_accs,
        "reg_train_accs": reg_train_accs,
        "reg_val_accs": reg_val_accs,
        "num_epochs": num_epochs,
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    print("  Saved: baseline_model.pth, regularized_model.pth, training_history.json")
    print("  Done!")


if __name__ == "__main__":
    main()