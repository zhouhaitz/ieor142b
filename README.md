# ieor142b

Multi-label movie genre classification from poster images.

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

## Regenerating splits (only if `MovieGenre.csv` changes)

From repo root **`ieor142b/`**:

```bash
python scripts/generate_splits.py
```

Commit the updated CSVs in `splits/` so everyone stays aligned.
