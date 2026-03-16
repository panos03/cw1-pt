"""
GenAI Usage Statement:
Claude was used to generate the plot with Pillow
Specific mistake:
- dotted lines were used initially, which looked messy, so I changed to solid lines of different colour shade
"""

import torch
import json
import os
from PIL import Image, ImageDraw
# NOTE: these imports are not required, but included as the CW description mentions task.py should load saved models
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
    Print the technical analysis

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
  Baseline
        Final Train Acc: {bl_final_train:.4f}
        Final Val Acc: {bl_final_val:.4f}
        Gap: {bl_gap:.4f}
  Dropout
        Final Train Acc: {rg_final_train:.4f}
        Final Val Acc: {rg_final_val:.4f}
        Gap: {rg_gap:.4f}
  Baseline Best Val Acc: {bl_best_val:.4f}
  Dropout Best Val Acc: {rg_best_val:.4f}

1. Hyperparameter Justification

Six layers with decreasing widths were used, creating a funnel shape that 
progressively compresses low-level features into high-level features for 
good class separation, and providing enough capacity to overfit in the 
baseline. A dropout rate of 0.4 was used, as it is high enough to provide 
clear regularisation and low enough to ensure the model can still learn. 
For the optimiser, SGD was chosen with a learning rate of 0.01 was used, 
for gradual and visibly-smooth enough accuracy changes between epochs, and 
momentum of 0.9 to reduce noise produced by SGD. Both models have
identical architecture and optimiser settings to ensure a fair experiment.

2. Dropout as Regularisation

Dropout randomly zeros each hidden unit's output with probability p=0.4 
during training, but uses all units for inference. This simple change 
reduces overfitting.
   First, dropout prevents co-adaptation. By randomly dropping units, each 
unit is forced to learn useful features on its own, producing sparser 
and more robust representations.
   Second, dropout acts as implicit model ensembling. Each forward pass 
effectively samples a different sub-network by using a binary mask on the 
dropped-out units. At inference, using all units with scaled weights is 
like averaging these sub-network predictions. This is the same principle 
as bagging, where combining multiple low-bias models reduces variance.
   Third, dropout introduces Bernoulli noise into hidden units, which is 
equivalent to a penalty on weight norms. This noise prevents specific 
weights from growing too large, reducing overfitting on the training data.

3. Generalisation Gap Discussion

The generalisation gap is the difference between training and validation 
accuracy, where a larger gap indicates the model has overfit to the 
training data, minimising empirical risk (low bias), but generalises badly 
to the unseen validation data (high variance).
   The baseline model has a larger gap of {bl_gap:.4f} by the final epoch, 
showing higher variance. Its high capacity (from the 6-layer network 
complexity) allows it to fit to training-specific noise, pushing training 
accuracy above {bl_final_train:.4f} (low bias) while validation accuracy 
plateaus around {bl_final_val:.4f}.
   The dropout model intentionally introduces bias to tradeoff variance. 
It achieves a smaller gap of {rg_gap:.4f} (lower variance), showing the 
generalisation to unseen data is better, due to the implicit regularisation 
discussed above. Although its training accuracy of {rg_final_train:.4f} is lower  
(higher bias), as the impaired network cannot fit training data as tightly, 
its validation accuracy of {rg_final_val:.4f} is very close to the baseline.

4. GenAI Usage and Technical Errors

   In train.py, Claude was used to check torch operations such as torch.max(), 
and for how to download datasets and use dataloaders.
Specific mistake:
downloaded separate test set in get_data_loaders(), but we only need train 
to construct train/val, and included raw paths for dataset root, which were 
changed to relative paths using os.path.join()
   In task.py, Claude was used to generate the plot with Pillow.
Specific mistake:
dotted lines were used initially, which looked messy. These were changed to 
solid lines of different colour shade
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
    train_loader, val_loader = get_data_loaders()
    bl_train_acc = compute_accuracy(baseline, train_loader)
    bl_val_acc = compute_accuracy(baseline, val_loader)
    reg_train_acc = compute_accuracy(regularised, train_loader)
    reg_val_acc = compute_accuracy(regularised, val_loader)
    print(f"  Baseline\n    Train: {bl_train_acc:.4f} (history: {training_history['baseline_train_accs'][-1]:.4f})"
          f"\n    Val: {bl_val_acc:.4f} (history: {training_history['baseline_val_accs'][-1]:.4f})")
    print(f"  Dropout\n    Train: {reg_train_acc:.4f} (history: {training_history['reg_train_accs'][-1]:.4f})"
          f"\n    Val: {reg_val_acc:.4f} (history: {training_history['reg_val_accs'][-1]:.4f})")

    # Generate plot
    print("[4/5] Generating generalization_gap.png...")
    plot_accuracy_curves(training_history, filename=PLOT_PATH)

    # Print technical analysis
    print("[5/5] Printing technical analysis...")
    print_analysis(training_history)


if __name__ == "__main__":
    main()
