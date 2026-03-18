"""
GenAI Usage Statement:
Claude was used to check torch operations such as torch.distributions.Beta(),
to verify the numerically stable log-softmax formula, and for saving model and training history.
Specific mistake:
- thought best model weights were from early-stopped epoch, but are from early-stopped epoch minus 'patience',
  also used a fixed lamda for a whole mixed-up batch, changed this to a different lamda per pair
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")
TRAINING_HISTORY_PATH = os.path.join(SCRIPT_DIR, "training_history.json")


# Data Loading (same as Task 1)

def get_data_loaders(batch_size=128, val_split=0.1):
    """
    Download Fashion-MNIST and create train and validation loaders.

    Args:
        batch_size (int): Batch size for data loaders.
        val_split (float): Fraction of training data used for validation.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )

    # Split training set into train and validation
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

    return train_loader, val_loader


# Model (same as Task 1 Baseline)

class FashionMLP(nn.Module):
    """
    Deep funnel-shaped neural network with 6 hidden layers
    Input: flattened 28x28 = 784 features
    Output: 10 classes (Fashion-MNIST)
    """

    def __init__(self):
        super(FashionMLP, self).__init__()
        self.flatten = nn.Flatten()     # Flatten 2D image to 1D vector
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)
        self.relu = nn.ReLU()           # ReLU activation for non-linearity

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


# From-Scratch Components

def to_onehot(labels, num_classes=10):
    """
    Convert integer class labels to one-hot vectors.

    Args:
        labels (torch.Tensor): Integer labels of shape (batch_size,).
        num_classes (int): Number of classes K.

    Returns:
        torch.Tensor: One-hot tensor of shape (batch_size, K).
    """
    onehot = torch.zeros(labels.size(0), num_classes)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot


def mixup(x, y_onehot, alpha=0.4):
    """
    Apply MixUp augmentation to a batch using lambda ~ Beta(alpha, alpha).

    Generates mixed inputs and soft labels:
        x_mixed = lam * x[i] + (1-lam) * x[j]
        y_mixed = lam * y[i] + (1-lam) * y[j]
    where (i, j) pairs come from a random permutation of the batch.

    Args:
        x (torch.Tensor): Input images of shape (batch_size, C, H, W).
        y_onehot (torch.Tensor): One-hot labels of shape (batch_size, K).
        alpha (float): Beta distribution concentration parameter.

    Returns:
        mixed_x (torch.Tensor): Mixed images of shape (batch_size, C, H, W).
        mixed_y (torch.Tensor): Mixed soft labels of shape (batch_size, K).
        lam (torch.Tensor): Per-sample mixing coefficients of shape (batch_size,).
    """
    # Sample one lambda per pair ~ Beta(alpha, alpha)
    alpha_t = torch.tensor(float(alpha))
    lam = torch.distributions.Beta(alpha_t, alpha_t).sample((x.size(0),))

    # Random permutation for pairing
    indices = torch.randperm(x.size(0))

    # Both views are of the same lam tensor, so lam[i] is shared between image and label
    lam_img = lam.view(-1, 1, 1, 1)     # broadcast over (C, H, W)
    lam_lbl = lam.view(-1, 1)           # broadcast over K classes
    mixed_x = lam_img * x + (1.0 - lam_img) * x[indices]
    mixed_y = lam_lbl * y_onehot + (1.0 - lam_lbl) * y_onehot[indices]

    return mixed_x, mixed_y, lam


def label_smoothing_cross_entropy(logits, original_target_labels, smoothing=0.1):
    """
    Cross-entropy loss with label smoothing.

    Smooths the original labels by mixing with a uniform distribution:
        y_smooth = (1 - eps) * original_target_labels + eps / K
    Then computes cross-entropy loss:
        loss = -sum(y_smooth * log_softmax(logits)) averaged over batch

    Args:
        logits (torch.Tensor): Raw model outputs of shape (batch_size, K).
        original_target_labels (torch.Tensor): Original target labels of shape (batch_size, K),
            may be one-hot labels or blended labels from MixUp.
        smoothing (float): Label smoothing epsilon in [0, 1).

    Returns:
        torch.Tensor: Scalar mean cross-entropy loss.
    """
    K = logits.size(1)

    # Apply label smoothing by distributing epsilon uniformly
    smoothed_labels = (1.0 - smoothing) * original_target_labels + smoothing / K

    # Numerically stable log-softmax ( mathematically equivalent to log(softmax(logits)) )
    max_l = logits.max(dim=1, keepdim=True)[0]
    stable = logits - max_l     # subtract max so exp doesn't overflow
    log_softmax = stable - torch.log(torch.exp(stable).sum(dim=1, keepdim=True))

    # Negative log-likelihood under smoothed labels, mean over batch
    loss = -(smoothed_labels * log_softmax).sum(dim=1).mean()

    return loss


# Training and Evaluation

def compute_accuracy(model, data_loader):   # (same as Task 1)
    """
    Compute a model's classification accuracy on a dataset.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): Data loader for dataset to evaluate on.

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


# NOTE: used instead of compute_accuracy for early stopping, as loss is smoother and more sensitive to small improvements
def compute_val_loss(model, data_loader, smoothing=0.1):
    """
    Compute mean label-smoothing cross-entropy loss on a dataset (no MixUp).

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Data loader for the target dataset.
        smoothing (float): Label smoothing epsilon.

    Returns:
        float: Mean loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            y_soft = to_onehot(labels)
            loss = label_smoothing_cross_entropy(logits, y_soft, smoothing=smoothing)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
    return total_loss / total


def train_with_early_stopping(
    model, train_loader, val_loader, optimiser,
    alpha=0.4, smoothing=0.1, num_epochs=50, patience=7, min_delta=1e-4
):      # builds on Task 1's train_model()
    """
    Train a model with MixUp, label smoothing, and early stopping.
    - MixUp is applied to every training batch
    - Label smoothing is used for both training and validation loss computation
    - Validation is evaluated on non-mixed examples
    - Training ends when validation loss hasn't improved by more than 'min_delta' for 'patience' consecutive epochs.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimiser (torch.optim.Optimizer): Optimiser for parameter updates.
        alpha (float): MixUp Beta distribution concentration parameter.
        smoothing (float): Label smoothing epsilon.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait without improvement before stopping.
        min_delta (float): Minimum improvement in validation loss to reset patience.

    Returns:
        train_accs (list of float): Training accuracy per epoch.
        val_accs (list of float): Validation accuracy per epoch.
        stopped_epoch (int): The epoch at which training stopped.
    """
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    stopped_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Apply MixUp before forward pass
            y_onehot = to_onehot(labels)
            mixed_x, mixed_y, _ = mixup(images, y_onehot, alpha=alpha)

            optimiser.zero_grad()
            outputs = model(mixed_x)
            loss = label_smoothing_cross_entropy(outputs, mixed_y, smoothing=smoothing)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)

        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        val_loss = compute_val_loss(model, val_loader, smoothing=smoothing)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch [{epoch+1:2d}/{num_epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}")

        # Validation-based early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # No significant validation loss improvement
            patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                print(f"  Early stopping triggered at epoch {stopped_epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # Restore best weights from the epoch with lowest validation loss
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_accs, val_accs, stopped_epoch


# Main

def main():
    """Main function: load data, train model with MixUp + label smoothing + early stopping, save weights and training history."""
    
    torch.manual_seed(42)
    alpha = 0.4        # MixUp Beta concentration parameter
    smoothing = 0.1    # Label smoothing epsilon
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 50
    patience = 7
    min_delta = 1e-4    # early stopping improvement threshold

    # Load data
    print("\n[1/3] Loading Fashion-MNIST dataset...")
    train_loader, val_loader = get_data_loaders()
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Train model
    print(f"\n[2/3] Training with MixUp (alpha={alpha}) + "
          f"Label Smoothing (eps={smoothing})...")
    print(f"  Optimiser: SGD, lr={learning_rate}, momentum={momentum}, no weight decay")
    print(f"  Early Stopping: patience={patience}, min_delta={min_delta}, max epochs={num_epochs}")  # clarify early stopping parameters in printout
    model = FashionMLP()
    optimiser = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    train_accs, val_accs, stopped_epoch = train_with_early_stopping(
        model, train_loader, val_loader, optimiser,
        alpha=alpha, smoothing=smoothing,
        num_epochs=num_epochs, patience=patience, min_delta=min_delta
    )

    # Save model
    best_epoch = stopped_epoch - patience if stopped_epoch < num_epochs else num_epochs
    print(f"\n[3/3] Saving model (best weights from epoch {best_epoch}) and training history...")
    torch.save(model.state_dict(), MODEL_PATH)

    # Save training history as JSON for task.py to load
    training_history = {
        "train_accs": train_accs,
        "val_accs": val_accs,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "alpha": alpha,
        "smoothing": smoothing,
    }
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(training_history, f)

    print("  Saved: model.pth, training_history.json")
    print("  Done!")


if __name__ == "__main__":
    main()
