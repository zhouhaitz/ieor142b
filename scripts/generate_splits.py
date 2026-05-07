#!/usr/bin/env python3
"""
Regenerate fixed train/val/test splits (imdbId lists) to match main.py logic.

Run from anywhere:
  python scripts/generate_splits.py

Or from ieor142b/:
  python scripts/generate_splits.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "MovieGenre.csv"
OUT_DIR = ROOT / "splits"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed train/val/test splits from a source CSV."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=CSV_PATH,
        help="Input CSV path (defaults to MovieGenre.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for split CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for splitting.",
    )
    parser.add_argument(
        "--no-dropna-genre",
        action="store_true",
        help="Do not drop rows where Genre is missing.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.out_dir
    seed = args.seed

    if not csv_path.exists():
        print(f"Missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="latin-1")
    if not args.no_dropna_genre:
        df = df.dropna(subset=["Genre"]).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_rows.csv", index=False)
    val_df.to_csv(out_dir / "val_rows.csv", index=False)
    test_df.to_csv(out_dir / "test_rows.csv", index=False)

    print(f"Wrote splits to {out_dir}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Columns: {list(train_df.columns)}")


if __name__ == "__main__":
    main()
