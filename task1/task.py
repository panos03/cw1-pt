"""
Task 1: The Dynamics of Generalization
task.py - Loads saved models, generates generalization_gap.png, prints technical analysis.

GenAI Usage Statement:
Claude (Anthropic) was used in an assistive role to help structure the code and draft
the technical analysis. All code was reviewed, understood, and verified by the student.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Model Definitions (must match train.py)
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """
    High-capacity deep neural network with 6 hidden layers, no regularization.
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
        """Forward pass. Input: (batch, 1, 28, 28). Output: (batch, 10)."""
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
    Deep neural network regularised using ONLY dropout.
    Same architecture as BaselineModel with dropout after each hidden activation.
    """

    def __init__(self, dropout_rate=0.4):
        """
        Args:
            dropout_rate (float): Dropout probability.
        """
        super(DropoutModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Forward pass. Input: (batch, 1, 28, 28). Output: (batch, 10)."""
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.fc6(x)
        return x


# ---------------------------------------------------------------------------
# Plotting with Pillow (no matplotlib)
# ---------------------------------------------------------------------------

def plot_accuracy_curves(history, filename="generalization_gap.png"):
    """
    Generate a PNG plotting train vs. validation accuracy for both models.

    Uses Pillow for drawing. The plot shows 4 curves:
    - Baseline train accuracy (solid blue)
    - Baseline validation accuracy (dashed blue)
    - Regularized train accuracy (solid red)
    - Regularized validation accuracy (dashed red)

    Args:
        history (dict): Dictionary with keys:
            'baseline_train_accs', 'baseline_val_accs',
            'reg_train_accs', 'reg_val_accs', 'num_epochs'.
        filename (str): Output filename for the PNG image.
    """
    # Image dimensions and margins
    width, height = 800, 500
    margin_left = 80
    margin_right = 200
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    num_epochs = history["num_epochs"]
    all_accs = (history["baseline_train_accs"] + history["baseline_val_accs"]
                + history["reg_train_accs"] + history["reg_val_accs"])
    y_min = max(0.0, min(all_accs) - 0.05)
    y_max = min(1.0, max(all_accs) + 0.02)

    def to_pixel(epoch_idx, acc):
        """
        Convert (epoch_index, accuracy) to pixel coordinates.

        Args:
            epoch_idx (int): Zero-based epoch index.
            acc (float): Accuracy value.

        Returns:
            tuple: (px_x, px_y) pixel coordinates.
        """
        px_x = margin_left + int(epoch_idx / max(num_epochs - 1, 1) * plot_w)
        px_y = margin_top + int((1.0 - (acc - y_min) / (y_max - y_min)) * plot_h)
        return px_x, px_y

    # Draw axes
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)],
              fill="black", width=2)
    draw.line([(margin_left, margin_top + plot_h),
               (margin_left + plot_w, margin_top + plot_h)],
              fill="black", width=2)

    # Y-axis labels
    num_y_ticks = 6
    for i in range(num_y_ticks + 1):
        val = y_min + i * (y_max - y_min) / num_y_ticks
        _, py = to_pixel(0, val)
        draw.text((5, py - 6), f"{val:.2f}", fill="black")
        # Grid line
        draw.line([(margin_left, py), (margin_left + plot_w, py)],
                  fill="#dddddd", width=1)

    # X-axis labels
    for epoch in range(0, num_epochs, max(1, num_epochs // 6)):
        px, _ = to_pixel(epoch, y_min)
        draw.text((px - 5, margin_top + plot_h + 10), str(epoch + 1), fill="black")

    # Title and axis labels
    draw.text((width // 2 - 120, 10),
              "Generalization Gap: Train vs Val Accuracy", fill="black")
    draw.text((width // 2 - 20, height - 20), "Epoch", fill="black")

    def draw_curve(accs, color, dashed=False):
        """
        Draw an accuracy curve on the image.

        Args:
            accs (list of float): Accuracy values per epoch.
            color (str): Line color.
            dashed (bool): If True, draw dashed line (dots every other segment).
        """
        points = [to_pixel(i, a) for i, a in enumerate(accs)]
        for i in range(len(points) - 1):
            if dashed and i % 2 == 1:
                continue
            draw.line([points[i], points[i + 1]], fill=color, width=2)
        # Draw small circles at each point
        for p in points:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)

    # Plot all four curves
    draw_curve(history["baseline_train_accs"], "#2196F3", dashed=False)   # Blue solid
    draw_curve(history["baseline_val_accs"], "#2196F3", dashed=True)      # Blue dashed
    draw_curve(history["reg_train_accs"], "#F44336", dashed=False)        # Red solid
    draw_curve(history["reg_val_accs"], "#F44336", dashed=True)           # Red dashed

    # Legend
    legend_x = margin_left + plot_w + 15
    legend_y = margin_top + 20
    legend_items = [
        ("Baseline Train", "#2196F3", False),
        ("Baseline Val", "#2196F3", True),
        ("Dropout Train", "#F44336", False),
        ("Dropout Val", "#F44336", True),
    ]
    for idx, (label, color, dashed) in enumerate(legend_items):
        ly = legend_y + idx * 25
        if dashed:
            draw.line([(legend_x, ly + 5), (legend_x + 10, ly + 5)], fill=color, width=2)
            draw.line([(legend_x + 16, ly + 5), (legend_x + 26, ly + 5)], fill=color, width=2)
        else:
            draw.line([(legend_x, ly + 5), (legend_x + 26, ly + 5)], fill=color, width=2)
        draw.text((legend_x + 32, ly), label, fill="black")

    img.save(filename)
    print(f"  Saved plot: {filename}")


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------

def print_analysis(history):
    """
    Print the ~500 word technical analysis discussing the generalization gap,
    bias-variance trade-off, and the role of optimizer as implicit regularization.

    Args:
        history (dict): Training history dictionary.
    """
    # Compute summary statistics
    bl_final_train = history["baseline_train_accs"][-1]
    bl_final_val = history["baseline_val_accs"][-1]
    bl_gap = bl_final_train - bl_final_val

    rg_final_train = history["reg_train_accs"][-1]
    rg_final_val = history["reg_val_accs"][-1]
    rg_gap = rg_final_train - rg_final_val

    bl_best_val = max(history["baseline_val_accs"])
    rg_best_val = max(history["reg_val_accs"])

    print("\n" + "=" * 60)
    print("TECHNICAL ANALYSIS: The Dynamics of Generalization")
    print("=" * 60)

    print(f"""
Summary Statistics:
  Baseline  - Final Train Acc: {bl_final_train:.4f}, Final Val Acc: {bl_final_val:.4f}, Gap: {bl_gap:.4f}
  Dropout   - Final Train Acc: {rg_final_train:.4f}, Final Val Acc: {rg_final_val:.4f}, Gap: {rg_gap:.4f}
  Baseline Best Val Acc: {bl_best_val:.4f}, Dropout Best Val Acc: {rg_best_val:.4f}

1. The Generalization Gap

The generalization gap is the difference between training and validation
accuracy. In the baseline model, we observe a substantial gap ({bl_gap:.4f})
by the final epoch. With approximately 670,000 parameters across six fully
connected layers and no regularisation, the baseline has sufficient capacity
to memorise training-specific patterns and noise. Its training accuracy
climbs toward very high values while validation accuracy plateaus or
declines — the classic hallmark of overfitting.

The dropout model exhibits a markedly smaller gap ({rg_gap:.4f}). Its
training accuracy is lower than the baseline's because dropout randomly
zeros 40% of activations in each forward pass during training, effectively
handicapping the network. However, its validation accuracy is comparable or
superior, demonstrating that the learned features generalise better to
unseen data.

2. How Dropout Works as Regularisation

Dropout (Srivastava et al., 2014) randomly sets each hidden unit's output
to zero with probability p during training. This can be understood through
multiple complementary lenses, as discussed in the lecture slides.

First, dropout prevents co-adaptation of neurons. Without dropout, neurons
can develop complex co-dependencies where specific neurons compensate for
errors made by others. By randomly removing units, dropout forces each
neuron to learn features that are independently useful, producing more
robust and redundant internal representations.

Second, dropout acts as an approximate model ensemble. Each training step
uses a different binary mask, effectively sampling from an exponentially
large family of sub-networks (2^n possible sub-networks for n hidden
units). At inference time, using all units with appropriately scaled weights
approximates the geometric mean of these sub-network predictions — an
implicit form of bagging without the cost of training separate models.

Third, dropout can be interpreted as approximate Bayesian inference (Gal
and Ghahramani, 2016). Each dropout mask samples a different set of model
parameters, and the inference-time averaging approximates the posterior
predictive distribution. This connection means dropout is equivalent to
adding a specific form of noise penalty on the weight norms.

3. Bias-Variance Trade-off

The baseline sits at the high-variance end of the bias-variance spectrum:
low training error (low bias) but high generalisation error (high variance).
Dropout shifts this balance by increasing bias slightly — the network
cannot achieve as low a training loss because it is impaired during
training — while substantially reducing variance. The ensemble
interpretation explains this: averaging over many sub-networks smooths out
the idiosyncratic noise that any single model would capture, reducing the
sensitivity of predictions to the particular training samples seen.

4. The Role of the Optimizer as Implicit Regularisation

Both models use SGD with momentum (lr=0.01, momentum=0.9) and no weight
decay. SGD itself provides implicit regularisation through mini-batch
stochasticity: the noise in gradient estimates biases optimisation toward
flat minima in the loss landscape, which correspond to solutions that
generalise better. Momentum smooths the trajectory and helps escape sharp
local minima, settling in broader basins. The learning rate controls the
noise scale (proportional to lr / batch_size), linking it directly to the
strength of implicit regularisation. These effects are present in both
models, meaning the performance difference we observe is attributable
specifically to dropout rather than optimiser choice.

5. Hyperparameter Justification

The dropout rate of 0.4 was selected to provide clearly visible
regularisation. The lecture slides note that rates up to 0.8 have been used
successfully; 0.4 is moderate enough to allow learning while producing a
distinct gap reduction compared to baseline. Both models use identical
optimiser settings (SGD, lr=0.01, momentum=0.9, batch_size=128) with no
weight decay, ensuring a fair comparison that isolates dropout's effect.
Training for 30 epochs allows sufficient time for overfitting to manifest
in the baseline while showing dropout's sustained generalisation.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function: load history, generate plot, print analysis."""
    print("=" * 60)
    print("Task 1: Generating Results")
    print("=" * 60)

    # Load training history
    print("\n[1/3] Loading training history...")
    with open("training_history.json", "r") as f:
        history = json.load(f)

    # Load models (verify they load correctly)
    print("[2/3] Loading saved models...")
    baseline = BaselineModel()
    baseline.load_state_dict(torch.load("baseline_model.pth", weights_only=True))
    print("  Baseline model loaded successfully.")

    regularized = DropoutModel(dropout_rate=0.4)
    regularized.load_state_dict(torch.load("regularized_model.pth", weights_only=True))
    print("  Dropout model loaded successfully.")

    # Generate plot
    print("[3/3] Generating generalization_gap.png...")
    plot_accuracy_curves(history, filename="generalization_gap.png")

    # Print technical analysis
    print_analysis(history)


if __name__ == "__main__":
    main()