"""
GenAI Usage Statement:  TODO
Claude was used for Pillow operations and plotting
Specific mistake:
- claude tried to change the mixup function to return class indices,
  but I made it so that the class indices that were mixed up can be derived from the returned 'mixed_y' vector
"""

import torch
import torchvision
import torchvision.transforms as transforms
import json
import os
from PIL import Image, ImageDraw
from train import FashionMLP, mixup, to_onehot, compute_accuracy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")
TRAINING_HISTORY_PATH = os.path.join(SCRIPT_DIR, "training_history.json")
DEMO_PATH = os.path.join(SCRIPT_DIR, "robustness_demo.png")
GAP_PLOT_PATH = os.path.join(SCRIPT_DIR, "generalization_gap.png")


# Data Loading - for noisy test set evaluation

def get_test_loader(batch_size=128):
    """
    Download Fashion-MNIST and return a DataLoader for the test split.

    Args:
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test set data loader (train=False).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return test_loader


# Noisy Evaluation

def add_gaussian_noise(x, sigma=0.3):
    """
    Random gaussian noise is added in the normalised [-1, 1] space.
    The result is clamped to prevent values outside this range.

    Args:
        x (torch.Tensor): Input tensor with values in [-1, 1].
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy tensor, clamped to [-1, 1].
    """
    noisy_x = x + sigma * torch.randn_like(x)
    return torch.clamp(noisy_x, -1.0, 1.0)


def compute_accuracy_noisy(model, test_loader, sigma=0.3):      # similar to compute_accuracy (but with noise added to inputs)
    """
    Evaluate model accuracy on the test set with added Gaussian noise.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        float: Accuracy on the noisy test set as a fraction in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            noisy_images = add_gaussian_noise(images, sigma=sigma)
            outputs = model(noisy_images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Visualisation

def tensor_to_pil(t, scale=2):
    """
    Convert a single normalised (1, H, W) tensor to an upscaled RGB PIL image.

    Denormalises from [-1, 1] to [0, 255] uint8,
    then upscales resolution using nearest-neighbour interpolation.

    Args:
        t (torch.Tensor): Tensor of shape (1, H, W) with values in [-1, 1].
        scale (int): Integer upscaling factor.

    Returns:
        PIL.Image: RGB image of size (W*scale, H*scale).
    """
    arr = ((t.squeeze(0) * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype('uint8')
    h, w = arr.shape
    # NOTE: FashionMNIST is grayscale, so we create a 'L' mode image and convert to 'RGB' for the montage
    image = Image.fromarray(arr, mode='L').convert('RGB').resize(
        (w * scale, h * scale), Image.NEAREST
    )
    return image


def derive_mixed_classes(mixed_y_i, lam, class_names):
    """
    Identify the two source classes from a MixUp soft label vector.

    For mixed_y_i = lam * e_a + (1-lam) * e_b, where e_i are unit vecctors,
    the non-zero indices are the two source classes.
    The index whose value is closest to lam is class A
    and the other is class B.

    Args:
        mixed_y_i (torch.Tensor): Soft label vector of shape (K,).
        lam (float): MixUp lambda used to blend the pair.
        class_names (list of str): Class name strings.

    Returns:
        tuple: (cls_a, cls_b) names of the primary and secondary classes.
    """
    nz = (mixed_y_i > 0).nonzero(as_tuple=True)[0].tolist()
    if len(nz) == 1:    # two mixed samples were from same class
        cls_a = cls_b = class_names[nz[0]]
    else:
        idx0, idx1 = nz[0], nz[1]
        if abs(mixed_y_i[idx0].item() - lam) < abs(mixed_y_i[idx1].item() - lam):
            cls_a, cls_b = class_names[idx0], class_names[idx1]
        else:
            cls_a, cls_b = class_names[idx1], class_names[idx0]
    return cls_a, cls_b


def save_robustness_demo(test_loader, filename, alpha=0.4):
    """
    Save a montage PNG showing 16 images processed by the MixUp function.

    Takes 32 images from the test set, applies MixUp to form 16 mixed images
    (each mixed from a pair), and arranges them in a 4x4 grid. Each cell
    shows one MixUp output with a label like "0.4 Shirt + 0.6 Bag",
    derived from the mixed soft labels returned by MixUp.

    Args:
        test_loader (DataLoader): Test data loader providing source images.
        filename (str): Output PNG file path.
        alpha (float): Beta distribution concentration parameter for MixUp.
    """
    torch.manual_seed(0)
    images, labels = next(iter(test_loader))  # at least 128 images available
    class_names = test_loader.dataset.classes

    n = 16  # number of mixed images to show (4x4 grid)
    y_onehot = to_onehot(labels[:n * 2])

    # Apply MixUp to the first 2n images; take first n mixed results for display
    mixed_x, mixed_y, lam = mixup(images[:n * 2], y_onehot, alpha=alpha)
    mixed_x = mixed_x[:n]
    mixed_y = mixed_y[:n]
    lam = lam[:n]

    # Build 4x4 grid
    scale = 3
    cell = 28 * scale   # 84 px per image — wide enough for label text
    border = 3
    grid = 4
    header_h = 20
    label_h = 28        # two lines of text below each image
    canvas_w = grid * cell + (grid + 1) * border
    canvas_h = grid * (cell + label_h) + (grid + 1) * border + header_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"MixUp Montage  alpha={alpha}", fill=(40, 40, 40))

    for i in range(n):
        row = i // grid
        col = i % grid
        x_off = border + col * (cell + border)
        y_off = header_h + border + row * (cell + label_h + border)
        canvas.paste(tensor_to_pil(mixed_x[i], scale=scale), (x_off, y_off))

        li = lam[i].item()
        cls_a, cls_b = derive_mixed_classes(mixed_y[i], li, class_names)

        draw.text((x_off, y_off + cell + 2), f"{li:.2f} {cls_a}", fill=(40, 40, 40))
        draw.text((x_off, y_off + cell + 14), f"+ {1-li:.2f} {cls_b}", fill=(80, 80, 80))

    canvas.save(filename)
    print(f"  Saved: {filename}")


# Generalisation Gap Plot

# (similar to plot for Task 1)
def plot_accuracy_curves(history, filename):
    """
    Generate a PNG plotting train vs. validation accuracy over epochs.

    Uses Pillow for drawing. Shows two curves:
    - Train accuracy (dark blue)
    - Validation accuracy (light blue)

    Args:
        history (dict): Training history with keys 'train_accs' and 'val_accs'.
        filename (str): Output PNG file path.
    """
    train_accs = history["train_accs"]
    val_accs = history["val_accs"]
    num_epochs = len(train_accs)

    width, height = 800, 500
    margin_left = 80
    margin_right = 160
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    all_accs = train_accs + val_accs
    y_min = max(0.0, min(all_accs) - 0.05)
    y_max = min(1.0, max(all_accs) + 0.02)

    def to_pixel(epoch_idx, acc):
        px_x = margin_left + int(epoch_idx / max(num_epochs - 1, 1) * plot_w)
        px_y = margin_top + int((1.0 - (acc - y_min) / (y_max - y_min)) * plot_h)
        return px_x, px_y

    # Axes
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)],
              fill="black", width=2)
    draw.line([(margin_left, margin_top + plot_h),
               (margin_left + plot_w, margin_top + plot_h)],
              fill="black", width=2)

    # Y-axis ticks and grid
    num_y_ticks = 6
    for i in range(num_y_ticks + 1):
        val = y_min + i * (y_max - y_min) / num_y_ticks
        _, py = to_pixel(0, val)
        draw.text((5, py - 6), f"{val:.2f}", fill="black")
        draw.line([(margin_left, py), (margin_left + plot_w, py)],
                  fill="#dddddd", width=1)

    # X-axis ticks
    for epoch in range(0, num_epochs, max(1, num_epochs // 6)):
        px, _ = to_pixel(epoch, y_min)
        draw.text((px - 5, margin_top + plot_h + 10), str(epoch + 1), fill="black")

    # Title and axis labels
    draw.text((width // 2 - 140, 10),
              "Generalisation Gap: Train vs Val Accuracy", fill="black")
    draw.text((width // 2 - 20, height - 20), "Epoch", fill="black")

    def draw_curve(accs, color):
        points = [to_pixel(i, a) for i, a in enumerate(accs)]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)
        for p in points:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)

    draw_curve(train_accs, "#1565C0")  # Dark blue
    draw_curve(val_accs,   "#90CAF9")  # Light blue

    # Legend
    legend_x = margin_left + plot_w + 15
    legend_y = margin_top + 20
    for label, color in [("Train", "#1565C0"), ("Val", "#90CAF9")]:
        draw.line([(legend_x, legend_y + 5), (legend_x + 26, legend_y + 5)],
                  fill=color, width=2)
        draw.text((legend_x + 32, legend_y), label, fill="black")
        legend_y += 25

    img.save(filename)
    print(f"  Saved plot: {filename}")


# Technical Analysis

def print_report(history, clean_acc, noisy_acc, sigma):
    """
    Print a ~500-word technical report on MixUp and Label Smoothing.

    Covers: (1) why MixUp prevents memorisation of training samples,
    (2) how label smoothing prevents overshooting during optimisation,
    (3) early stopping justification, supported by quantitative results.

    Args:
        history (dict): Training history with keys 'alpha', 'smoothing',
            'stopped_epoch', 'train_accs', 'val_accs'.
        clean_acc (float): Model accuracy on the clean test set in [0, 1].
        noisy_acc (float): Model accuracy on the noisy test set in [0, 1].
        sigma (float): Gaussian noise standard deviation used.
    """
    alpha = history.get("alpha", 0.4)
    eps = history.get("smoothing", 0.1)
    stopped = history.get("stopped_epoch", "N/A")
    patience = 7
    K = 10
    final_train = history["train_accs"][-1]
    final_val = history["val_accs"][-1]
    correct_target = (1.0 - eps) + eps / K
    wrong_target = eps / K

    # TODO
    print(f"""
=== Task 2: Results Summary ===
  MixUp alpha:              {alpha}
  Label Smoothing epsilon:  {eps}
  Early stopping patience:  {patience}
  Training stopped at epoch {stopped}
  Final Train Acc:          {final_train:.4f}
  Final Val Acc:            {final_val:.4f}
  Clean Test Acc:           {clean_acc:.4f}
  Noisy Test Acc (sigma={sigma}): {noisy_acc:.4f}
  Robustness Drop:          {clean_acc - noisy_acc:.4f}

=== Technical Analysis (~500 words) ===

1. Why MixUp Prevents Memorisation

Standard empirical risk minimisation (ERM) trains on fixed (x, y) pairs,
which pushes the network to memorise sharp, sample-specific input-output
mappings. MixUp breaks this by replacing every training sample with a
convex interpolation between two randomly selected examples:

    x_mixed = lam * x_i + (1-lam) * x_j
    y_mixed = lam * y_i + (1-lam) * y_j,  lam ~ Beta({alpha}, {alpha})

Every mixed sample lies strictly off the training manifold — it is a
point on the line segment between two real examples in input space. The
model never observes an actual training point during optimisation and
therefore cannot memorise individual samples in the way ERM allows.

MixUp also constrains the function the network can learn. Perfect
memorisation requires a highly non-linear function with sharp, isolated
response peaks around each training point. Forcing the loss to be small
at convex combinations implicitly enforces a Lipschitz-like smoothness:
the model's output must vary nearly linearly with lam. This is equivalent
to vicinal risk minimisation, where risk is minimised in a neighbourhood
around each pair of training points rather than at discrete points alone.
The decision boundaries are pushed away from real examples, reducing
reliance on brittle, sample-specific features.

This robustness is confirmed quantitatively. Clean test accuracy is
{clean_acc:.4f}. Under Gaussian pixel noise (sigma={sigma}) it drops by only
{clean_acc - noisy_acc:.4f} to {noisy_acc:.4f}. Because the model learned smooth,
distributed representations rather than sharp, localised ones, small
perturbations to pixel values change the class logits only slightly.

2. How Label Smoothing Prevents Overshooting

Hard one-hot targets instruct the network to drive the correct-class
logit to +inf and all others to -inf. The cross-entropy gradient with
respect to the correct-class logit is (p_k - 1), where p_k approaches
1 but never reaches it: the gradient never vanishes, so the optimiser
keeps nudging weights to ever-larger values. This overshooting leads to
over-confident, poorly calibrated predictions and large effective
learning-rate steps near convergence.

Label smoothing replaces hard targets with a softer distribution:

    y_smooth_k = (1 - eps) * y_k + eps / K,  eps={eps}, K={K}

The correct-class target becomes {correct_target:.4f} instead of 1.0,
and each wrong-class target becomes {wrong_target:.4f} instead of 0.0.
The loss is now minimised at finite logit differences, because the
target distribution has full support over all K classes — no finite
logit gap achieves zero loss. The residual eps/K mass on wrong classes
acts as a continuous restoring force that prevents weights from drifting
to extreme values, directly limiting overshooting.

Geometrically, label smoothing changes the loss landscape so the global
minimum sits at a finite, bounded weight configuration rather than at
the limit of an infinite norm ball. This is analogous to L2
regularisation applied in output probability space: instead of penalising
large weight norms directly, it penalises over-confident predictions,
which indirectly limits activation magnitudes through the loss gradient.

Combined with MixUp, the effect is compounded. MixUp already produces
soft, multi-class targets (the lam-blend of two one-hot vectors); label
smoothing then prevents over-confidence in those mixed targets. Together
they produce well-calibrated uncertainty in the model, especially
valuable under distributional shift such as the Gaussian noise above.

3. Early Stopping

Validation loss (label-smoothing cross-entropy on clean examples) was
tracked after every epoch. Training halted at epoch {stopped} when no
improvement greater than 1e-4 was observed for {patience} consecutive
epochs, preventing late-stage overfitting beyond the regime where MixUp
and label smoothing remain effective. The model weights achieving the
lowest validation loss were restored before saving.
""")


# Main

def main():
    """Load model, evaluate on clean/noisy test sets, save demo, print report."""

    print("\n[1/5] Loading model and training history...")
    model = FashionMLP()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with open(TRAINING_HISTORY_PATH) as f:
        history = json.load(f)
    print("  Loaded successfully.")

    print("[2/5] Evaluating on clean and noisy test sets...")
    test_loader = get_test_loader()
    clean_acc = compute_accuracy(model, test_loader)
    sigma = 0.3
    noisy_acc = compute_accuracy_noisy(model, test_loader, sigma=sigma)
    print(f"  Clean Test Acc:           {clean_acc:.4f}")
    print(f"  Noisy Test Acc (sigma={sigma}): {noisy_acc:.4f}")

    print("[3/5] Saving robustness_demo.png...")
    save_robustness_demo(test_loader, DEMO_PATH, alpha=history.get("alpha", 0.4))

    print("[4/5] Saving generalization_gap.png...")
    plot_accuracy_curves(history, GAP_PLOT_PATH)

    print("[5/5] Printing technical report...")
    print_report(history, clean_acc, noisy_acc, sigma)


if __name__ == "__main__":
    main()
