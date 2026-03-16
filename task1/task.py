"""
GenAI Usage Statement:
Claude was used in an assistive role to help structure the code and draft
the technical analysis. All code was reviewed, understood, and verified by TODO
"""

import torch
import json
import os
from PIL import Image, ImageDraw
from train import BaselineModel, DropoutModel, get_data_loaders, compute_accuracy



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_MODEL_PATH = os.path.join(SCRIPT_DIR, "baseline_model.pth")
REGULARISED_MODEL_PATH = os.path.join(SCRIPT_DIR, "regularised_model.pth")
TRAINING_HISTORY_PATH = os.path.join(SCRIPT_DIR, "training_history.json")
PLOT_PATH = os.path.join(SCRIPT_DIR, "generalization_gap.png")


# Plotting with Pillow (no matplotlib)

def plot_accuracy_curves(training_history, filename="generalization_gap.png"):
    """
    Generate a PNG plotting train vs. validation accuracy for both models.

    Uses Pillow for drawing. The plot shows 4 curves:
    - Baseline train accuracy (dark blue)
    - Baseline validation accuracy (light blue)
    - Regularised train accuracy (dark red)
    - Regularised validation accuracy (light red)

    Args:
        training_history (dict): Dictionary with keys:
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

    num_epochs = training_history["num_epochs"]
    all_accs = (training_history["baseline_train_accs"] + training_history["baseline_val_accs"]
                + training_history["reg_train_accs"] + training_history["reg_val_accs"])
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
              "Generalisation Gap: Train vs Val Accuracy", fill="black")
    draw.text((width // 2 - 20, height - 20), "Epoch", fill="black")

    def draw_curve(accs, color):
        """
        Draw an accuracy curve on the image.

        Args:
            accs (list of float): Accuracy values per epoch.
            color (str): Line color.
        """
        points = [to_pixel(i, a) for i, a in enumerate(accs)]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)
        for p in points:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)

    # Plot all four curves: train = dark, val = light version of same colour
    draw_curve(training_history["baseline_train_accs"], "#1565C0")  # Dark blue
    draw_curve(training_history["baseline_val_accs"],   "#90CAF9")  # Light blue
    draw_curve(training_history["reg_train_accs"],      "#B71C1C")  # Dark red
    draw_curve(training_history["reg_val_accs"],        "#EF9A9A")  # Light red

    # Legend
    legend_x = margin_left + plot_w + 15
    legend_y = margin_top + 20
    legend_items = [
        ("Baseline Train", "#1565C0"),
        ("Baseline Val",   "#90CAF9"),
        ("Dropout Train",  "#B71C1C"),
        ("Dropout Val",    "#EF9A9A"),
    ]
    for idx, (label, color) in enumerate(legend_items):
        ly = legend_y + idx * 25
        draw.line([(legend_x, ly + 5), (legend_x + 26, ly + 5)], fill=color, width=2)
        draw.text((legend_x + 32, ly), label, fill="black")

    img.save(filename)
    print(f"  Saved plot: {filename}")


# Technical Analysis

def print_analysis(training_history):
    """
    Print the ~500 word technical analysis discussing the generalization gap,
    bias-variance trade-off, and the role of optimizer as implicit regularization.

    Args:
        training_history (dict): Training history dictionary.
    """
    # Compute summary statistics
    bl_final_train = training_history["baseline_train_accs"][-1]
    bl_final_val = training_history["baseline_val_accs"][-1]
    bl_gap = bl_final_train - bl_final_val

    rg_final_train = training_history["reg_train_accs"][-1]
    rg_final_val = training_history["reg_val_accs"][-1]
    rg_gap = rg_final_train - rg_final_val

    bl_best_val = max(training_history["baseline_val_accs"])
    rg_best_val = max(training_history["reg_val_accs"])

    # TODO
    print(f"""
Summary Statistics:
  Baseline  - Final Train Acc: {bl_final_train:.4f}, Final Val Acc: {bl_final_val:.4f}, Gap: {bl_gap:.4f}
  Dropout   - Final Train Acc: {rg_final_train:.4f}, Final Val Acc: {rg_final_val:.4f}, Gap: {rg_gap:.4f}
  Baseline Best Val Acc: {bl_best_val:.4f}, Dropout Best Val Acc: {rg_best_val:.4f}

1. The Generalisation Gap

The generalisation gap is the difference between training and validation
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

4. The Role of the Optimiser as Implicit Regularisation

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


# Main

def main():
    """Main function: load training_history, generate plot, print analysis."""

    # Load training history
    print("\n[1/5] Loading training history...")
    with open(TRAINING_HISTORY_PATH, "r") as f:
        training_history = json.load(f)

    # Load models (verify they load correctly)
    print("[2/5] Loading saved models...")
    baseline = BaselineModel()
    baseline.load_state_dict(torch.load(BASELINE_MODEL_PATH, weights_only=True))
    print("  Baseline model loaded successfully.")
    regularised = DropoutModel(dropout_rate=0.4)
    regularised.load_state_dict(torch.load(REGULARISED_MODEL_PATH, weights_only=True))
    print("  Dropout model loaded successfully.")

    # Recompute final accuracies to confirm consistency with training history
    print("[3/5] Recomputing final accuracies from saved models...")
    train_loader, val_loader = get_data_loaders(batch_size=128)
    bl_train_acc = compute_accuracy(baseline, train_loader)
    bl_val_acc = compute_accuracy(baseline, val_loader)
    reg_train_acc = compute_accuracy(regularised, train_loader)
    reg_val_acc = compute_accuracy(regularised, val_loader)
    print(f"  Baseline\n    Train: {bl_train_acc:.4f} (history: {training_history['baseline_train_accs'][-1]:.4f}), "
          f"Val: {bl_val_acc:.4f} (history: {training_history['baseline_val_accs'][-1]:.4f})")
    print(f"  Dropout\n    Train: {reg_train_acc:.4f} (history: {training_history['reg_train_accs'][-1]:.4f}), "
          f"Val: {reg_val_acc:.4f} (history: {training_history['reg_val_accs'][-1]:.4f})")

    # Generate plot
    print("[4/5] Generating generalization_gap.png...")
    plot_accuracy_curves(training_history, filename=PLOT_PATH)

    # Print technical analysis
    print("[5/5] Printing technical analysis...")
    print_analysis(training_history)


if __name__ == "__main__":
    main()
