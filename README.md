# ieor142b

Multi-label movie genre classification from poster images.

Start here after pull: see `TEAM_DATA_SETUP.md`.

## Baseline reference (do not edit casually)

- **`main.py`** — canonical script implementation; treat as the reference for data cleaning, splits logic, metrics, and training loop.
- **`main.ipynb`** — notebook mirror of the baseline; **only change by team agreement** (e.g. bugfixes or shared baseline updates).

Experiments (ViT, scratch CNN, ResNet fine-tune) live under **`experimentation/`**. Each person should edit **only their own notebook** to avoid merge conflicts.

## Shared team contract

- **Data:** `MovieGenre.csv`, posters in `SampleMoviePosters/`.
- **Splits:** Fixed row indices in `splits/` (`train_rows.csv`, `val_rows.csv`, `test_rows.csv`), generated with seed **42** and the same cleaning as `main.py` (drop rows with missing `Genre`). See `splits/README.md`.
- **Labels:** Fit `MultiLabelBinarizer` on the **full cleaned** dataframe before subsetting by split (same as baseline).
- **Constants:** Image size **224**, eval threshold **0.5** on `sigmoid(logits)`, shared metrics and **FocalLoss** defaults as in `main.py` unless you document a deliberate deviation.

Helpers and JSON schema for saved metrics: see **`experimentation/shared.py`** and **`results/README.md`**.

**After experiments:** open **`final_comparison.ipynb`** to load all `results/*_metrics.json` files and compare models.

## Shared cleaned dataset (recommended for experiments)

If your model should only use rows with confirmed image files, use:

- **Canonical strict CSV:** `cleaned/MovieGenre_clean_with_images_full.csv` — **only** rows where the image was resolved via **`imdbId`** (highest confidence).
- **Manifest:** `cleaned/MovieGenre_clean_with_images_full_manifest.json`
- **Splits:** `splits_cleaned/` — generated from the strict CSV.

These artifacts are produced by `scripts/data_cleaning.ipynb`. You can regenerate splits with `scripts/generate_splits.py` (same `random_state=42` split logic as `main.py`).

After `git pull`, teammates should run:

```bash
python scripts/fetch_cleaned_images.py
```

## Regenerating splits (only if `MovieGenre.csv` changes)

From repo root **`ieor142b/`**:

```bash
python scripts/generate_splits.py
```

Commit the updated CSVs in `splits/` so everyone stays aligned.

To regenerate splits for the cleaned strict dataset:

```bash
python scripts/generate_splits.py \
  --csv-path cleaned/MovieGenre_clean_with_images_full.csv \
  --out-dir splits_cleaned \
  --seed 42
```

`splits/` remains the **legacy** baseline splits for the full raw `MovieGenre.csv` (see `splits/README.md`). Use `splits_cleaned/` for image-filtered experiments.
