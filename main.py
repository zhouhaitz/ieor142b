"""
Movie Poster Genre Classification
==================================
Multi-label genre classification from IMDB movie posters using
transfer learning (ResNet-50) with Focal Loss for class imbalance.

Dataset columns: imdbId, Imdb Link, Title, IMDB Score, Genre, Poster
Posters (jpg) are stored locally in: ./SampleMoviePosters/
"""

import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
class Config:
    CSV_PATH        = "MovieGenre.csv"          # path to your CSV
    POSTER_DIR      = "SampleMoviePosters"      # folder with .jpg files
    IMG_SIZE        = 224                        # ResNet input size
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 30
    LR              = 1e-4
    WEIGHT_DECAY    = 1e-4                       # L2 regularization
    DROPOUT         = 0.5
    PATIENCE        = 5                          # early stopping patience
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    SEED            = 42
    # Focal Loss params
    FOCAL_ALPHA     = 0.25
    FOCAL_GAMMA     = 2.0

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
print(f"Using device: {Config.DEVICE}")


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
class MoviePosterDataset(Dataset):
    """
    Loads movie poster images from disk and returns
    (image_tensor, multi-hot genre label vector).

    The CSV 'Genre' column is pipe-separated: e.g. "Animation|Adventure|Comedy"
    Each poster is matched by imdbId.jpg inside POSTER_DIR.
    """
    def __init__(self, df: pd.DataFrame, mlb: MultiLabelBinarizer,
                 poster_dir: str, transform=None):
        self.df         = df.reset_index(drop=True)
        self.mlb        = mlb
        self.poster_dir = poster_dir
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        imdb_id  = str(int(row["imdbId"]))
        img_path = os.path.join(self.poster_dir, f"{imdb_id}.jpg")

        # Load image — fall back to blank image if file missing
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        # Multi-hot label vector
        genres = row["Genre"].split("|")
        label  = torch.tensor(
            self.mlb.transform([genres])[0], dtype=torch.float32
        )
        return img, label


# ─────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    """ImageNet-normalised transforms. Training set uses augmentation."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# 3. FOCAL LOSS  ← Advanced Method
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-label Binary Focal Loss (Lin et al., 2017 — RetinaNet / ICCV).
    Addresses class imbalance by down-weighting easy (well-classified)
    examples and focusing training on hard, misclassified ones.

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Compared to vanilla BCEWithLogitsLoss, the (1 - p_t)^γ modulating
    factor reduces the loss for confident correct predictions, so rare
    genres (Documentary, Short, etc.) contribute more meaningfully.

    Args:
        alpha (float): Weighting factor for the rare/positive class [0,1].
        gamma (float): Focusing parameter. 0 → reduces to BCE. Typical: 2.
        reduction (str): 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable sigmoid + BCE in one step
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t = probability of the TRUE class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulating factor
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss   = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
class PosterGenreClassifier(nn.Module):
    """
    ResNet-50 backbone (pretrained on ImageNet) with a custom
    multi-label classification head.

    Architecture:
        ResNet-50 feature extractor (frozen early layers)
        → AdaptiveAvgPool (built into ResNet)
        → Dropout(p)
        → Linear(2048 → 512)  + ReLU
        → Dropout(p)
        → Linear(512 → num_genres)
        → [Sigmoid applied at inference; BCEWithLogitsLoss handles it in training]
    """
    def __init__(self, num_genres: int, dropout: float = Config.DROPOUT):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all backbone layers initially; we fine-tune layer4 + head
        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Replace the final FC layer
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()            # remove original classifier
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_genres),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 2048)
        logits   = self.head(features)  # (B, num_genres)
        return logits

    def unfreeze_backbone(self):
        """Call after a few warm-up epochs to fine-tune the full network."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """
    Computes sample-averaged multi-label metrics.
    Returns: precision, recall, F1, exact-match accuracy.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()

    # Per-sample intersection / union
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall    = (tp / (tp + fn + 1e-8)).mean().item()
    f1        = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
    exact     = (preds == targets).all(dim=1).float().mean().item()

    return {"precision": precision, "recall": recall,
            "f1": f1, "exact_match": exact}


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(
        torch.cat(all_logits), torch.cat(all_targets)
    )
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits     = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(
        torch.cat(all_logits), torch.cat(all_targets)
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────
def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Focal Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_f1"], label="Train")
    axes[1].plot(epochs, history["val_f1"],   label="Val")
    axes[1].set_title("F1 Score"); axes[1].legend()

    axes[2].plot(epochs, history["val_precision"], label="Precision")
    axes[2].plot(epochs, history["val_recall"],    label="Recall")
    axes[2].set_title("Val Precision / Recall"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")


def plot_genre_distribution(df: pd.DataFrame, save_path: str = "genre_dist.png"):
    """Bar chart of genre frequencies to visualise class imbalance."""
    all_genres = df["Genre"].str.split("|").explode()
    counts = all_genres.value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Genre Distribution in Dataset")
    ax.set_xlabel("Genre"); ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Genre distribution saved → {save_path}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    # ── Load & explore data ──────────────────
    df = pd.read_csv(Config.CSV_PATH, encoding="latin-1")
    print(f"Dataset size: {len(df)} movies")
    print(df.head(3))
    print("\nGenre value counts (top 15):")
    print(df["Genre"].str.split("|").explode().value_counts().head(15))
    plot_genre_distribution(df)

    # Drop rows with missing genres or poster paths
    df = df.dropna(subset=["Genre"]).reset_index(drop=True)

    # ── Encode multi-labels ──────────────────
    genre_lists = df["Genre"].str.split("|").tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)
    num_genres = len(mlb.classes_)
    print(f"\nGenres ({num_genres}): {list(mlb.classes_)}")

    # ── Train / Val / Test split ─────────────
    train_df, temp_df = train_test_split(df, test_size=0.2,
                                         random_state=Config.SEED)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5,
                                          random_state=Config.SEED)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Datasets & Loaders ───────────────────
    train_tf, val_tf = get_transforms()
    train_ds = MoviePosterDataset(train_df, mlb, Config.POSTER_DIR, train_tf)
    val_ds   = MoviePosterDataset(val_df,   mlb, Config.POSTER_DIR, val_tf)
    test_ds  = MoviePosterDataset(test_df,  mlb, Config.POSTER_DIR, val_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model, Loss, Optimizer ───────────────
    model     = PosterGenreClassifier(num_genres).to(Config.DEVICE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY   # L2 regularization
    )
    # Cosine annealing LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )

    # ── Training with early stopping ─────────
    history = {k: [] for k in
               ["train_loss", "val_loss", "train_f1", "val_f1",
                "val_precision", "val_recall"]}
    best_val_f1  = 0.0
    patience_ctr = 0

    print("\n── Training ──────────────────────────────")
    for epoch in range(1, Config.NUM_EPOCHS + 1):

        # Unfreeze full backbone after 5 warm-up epochs
        if epoch == 6:
            model.unfreeze_backbone()
            print("  [Unfroze full backbone for fine-tuning]")
            # Re-create optimizer to include newly unfrozen params
            optimizer = optim.Adam(model.parameters(),
                                   lr=Config.LR / 5,
                                   weight_decay=Config.WEIGHT_DECAY)

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_m = evaluate(
            model, val_loader, criterion, Config.DEVICE
        )
        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])

        print(f"Epoch {epoch:3d}/{Config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_m['f1']:.4f} | Val Exact: {val_m['exact_match']:.4f}")

        # Early stopping
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(model.state_dict(), "best_model.pt")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= Config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Final Evaluation on Test Set ─────────
    print("\n── Test Set Evaluation ───────────────────")
    model.load_state_dict(torch.load("best_model.pt",
                                     map_location=Config.DEVICE))
    _, test_m = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"Test F1:        {test_m['f1']:.4f}")
    print(f"Test Precision: {test_m['precision']:.4f}")
    print(f"Test Recall:    {test_m['recall']:.4f}")
    print(f"Test Exact Match: {test_m['exact_match']:.4f}")

    plot_training_curves(history)
    print("\nDone.")


if __name__ == "__main__":
    main()