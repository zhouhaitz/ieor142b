# Team Data Setup

Use this once after `git pull` so your local environment has poster files for the shared strict dataset.

## 1) Enter project root

```bash
cd ieor142b
```

## 2) Install dependencies (if needed)

Use your existing course environment. At minimum this script uses:

- `python`
- `pandas`

## 3) Optional quick smoke test

```bash
python scripts/fetch_cleaned_images.py --smoke-test
python scripts/fetch_cleaned_images.py --verify
```

## 4) Full setup command

```bash
python scripts/fetch_cleaned_images.py
```

This downloads (or reuses cached) posters into `cleaned/downloaded_posters/`.

## 5) Files to use for modeling

- Dataset CSV: `cleaned/MovieGenre_clean_with_images_full.csv`
- Splits: `splits_cleaned/train_rows.csv`, `splits_cleaned/val_rows.csv`, `splits_cleaned/test_rows.csv`
- Images: `cleaned/downloaded_posters/` (paths in CSV are relative and point here)

## 6) Legacy baseline note

- `MovieGenre.csv` and `splits/` are legacy baseline artifacts for `main.py` comparisons.
- For the cleaned strict workflow, use `cleaned/*` + `splits_cleaned/*`.
