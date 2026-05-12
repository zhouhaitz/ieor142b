from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


DEFAULT_CHECKPOINT_NAME = "hannah_scratch_cnn.pt"
DEFAULT_THRESHOLD = 0.40
DEFAULT_IMG_SIZE = 224
TOP_K = 8


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root in local or Colab-style layouts."""
    candidates: list[Path] = []
    if start is None:
        start = Path.cwd().resolve()

    candidates.extend([start, *start.parents])
    candidates.extend([
        Path("/content/ieor142b_runtime"),
        Path("/content/drive/MyDrive/ieor142b"),
    ])

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if (candidate / "cleaned" / "MovieGenre_clean_with_images_full.csv").is_file():
            return candidate

    raise RuntimeError(
        "Could not find repo root containing cleaned/MovieGenre_clean_with_images_full.csv. "
        "Run this from the ieor142b repo root or pass absolute paths."
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HannahScratchCNN(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.10),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.20),
            ConvBlock(128, 256, dropout=0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def resolve_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return root / path


def choose_random_poster(root: Path, seed: int) -> Path:
    poster_dir = root / "cleaned" / "downloaded_posters"
    posters = sorted(poster_dir.glob("*.jpg"))
    if not posters:
        raise FileNotFoundError(f"No JPG posters found in {poster_dir}")
    rng = random.Random(seed)
    return rng.choice(posters)


def build_eval_transform(img_size: int) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Create it by running the full-training cell in experimentation/hannah.ipynb, "
            "or pass --checkpoint path/to/hannah_scratch_cnn.pt."
        )
    return torch.load(path, map_location=device)


def infer(model: nn.Module, image_path: Path, transform, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
    return torch.sigmoid(logits).squeeze(0).detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hannah CNN inference on one movie poster.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to hannah_scratch_cnn.pt.")
    parser.add_argument("--image", type=Path, default=None, help="Poster image path. Defaults to a random local poster.")
    parser.add_argument("--threshold", type=float, default=None, help="Prediction threshold. Defaults to checkpoint best_threshold or 0.40.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of top probabilities to print.")
    parser.add_argument("--seed", type=int, default=142, help="Seed used when selecting a random poster.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repo_root()
    here = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint or (here / DEFAULT_CHECKPOINT_NAME)
    checkpoint_path = resolve_path(checkpoint_path, root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)

    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint is missing the 'classes' list needed for label names.")

    img_size = int(checkpoint.get("img_size", DEFAULT_IMG_SIZE))
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("best_threshold", DEFAULT_THRESHOLD))

    model = HannahScratchCNN(num_labels=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image_path = resolve_path(args.image, root) if args.image else choose_random_poster(root, args.seed)
    if not image_path.is_file():
        raise FileNotFoundError(f"Poster image not found: {image_path}")

    probs = infer(model, image_path, build_eval_transform(img_size), device)
    prob_df = pd.DataFrame({"genre": classes, "prob": probs.numpy()})
    prob_df = prob_df.sort_values("prob", ascending=False).reset_index(drop=True)

    pred_labels = prob_df.loc[prob_df["prob"] >= threshold, "genre"].tolist()

    print("Repo root:", root)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Poster:", image_path)
    print("Image size:", img_size)
    print(f"Threshold: {threshold:.2f}")
    print("\nPredicted labels:")
    print(pred_labels if pred_labels else "(none above threshold)")
    print(f"\nTop {args.top_k} probabilities:")
    for row in prob_df.head(args.top_k).itertuples(index=False):
        print(f"{row.genre:>12s}: {row.prob:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
