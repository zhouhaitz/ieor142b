#!/usr/bin/env python3
"""
Regenerate fixed train/val/test splits (imdbId lists) to match main.py logic.

Run from anywhere:
  python scripts/generate_splits.py

Or from ieor142b/:
  python scripts/generate_splits.py
"""
from __future__ import annotations

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
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df = df.dropna(subset=["Genre"]).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use row indices into the cleaned dataframe so splits stay 1:1 even if imdbId repeats.
    pd.DataFrame({"row_id": train_df.index}).to_csv(OUT_DIR / "train_rows.csv", index=False)
    pd.DataFrame({"row_id": val_df.index}).to_csv(OUT_DIR / "val_rows.csv", index=False)
    pd.DataFrame({"row_id": test_df.index}).to_csv(OUT_DIR / "test_rows.csv", index=False)

    print(f"Wrote splits to {OUT_DIR}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()
