"""
Shared helpers for team experiments.

Keeps data cleaning, split loading, label encoding, and result JSON shape consistent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer

# Project root: ieor142b/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
CSV_PATH = PROJECT_ROOT / "MovieGenre.csv"
POSTER_DIR = PROJECT_ROOT / "SampleMoviePosters"

BASELINE: Dict[str, Any] = {
    "csv_path": "MovieGenre.csv",
    "poster_dir": "SampleMoviePosters",
    "img_size": 224,
    "seed": 42,
    "dropna_columns": ["Genre"],
    "train_fraction": 0.8,
    "val_fraction_of_holdout": 0.5,  # val/test each 10% of full data
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "metric_threshold": 0.5,
}

REQUIRED_METRICS_KEYS: List[str] = [
    "model_name",
    "seed",
    "img_size",
    "train_size",
    "val_size",
    "test_size",
    "best_val_f1",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_exact_match",
    "num_epochs_run",
    "notes",
]


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_cleaned_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    return df.dropna(subset=BASELINE["dropna_columns"]).reset_index(drop=True)


def fit_mlb_on_full_cleaned(df: pd.DataFrame) -> MultiLabelBinarizer:
    genre_lists = df["Genre"].str.split("|").tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)
    return mlb


def load_fixed_split_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
    """
    Return train_df, val_df, test_df, mlb with mlb fit on full cleaned data.

    Each splits CSV contains all original columns from MovieGenre.csv (after cleaning),
    written directly by generate_splits.py. mlb is fit on the full cleaned dataframe
    so genre class order is stable regardless of which split is loaded first.
    """
    df = load_cleaned_dataframe()
    mlb = fit_mlb_on_full_cleaned(df)

    train_df = pd.read_csv(SPLITS_DIR / "train_rows.csv", encoding="latin-1")
    val_df = pd.read_csv(SPLITS_DIR / "val_rows.csv", encoding="latin-1")
    test_df = pd.read_csv(SPLITS_DIR / "test_rows.csv", encoding="latin-1")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), mlb


def validate_metrics_payload(payload: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_METRICS_KEYS if k not in payload]
    if missing:
        raise ValueError(f"metrics JSON missing keys: {missing}")


def save_metrics_json(path: Path | str, payload: Dict[str, Any]) -> None:
    validate_metrics_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_metrics_json(path: Path | str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
