# Team Data Setup

Use this once after `git pull` so your local environment has poster files for the shared strict dataset.

`scripts/fetch_cleaned_images.py` is the one setup script for teammates. It:
- reads `cleaned/MovieGenre_clean_with_images_full.csv`
- uses each row's `Poster` URL and `imdbId`
- downloads images to `cleaned/downloaded_posters/<imdbId>.jpg`
- skips already downloaded non-empty files (resume/cached behavior)

Why everyone must run it after pull:
- the CSV/splits are committed, but poster JPG binaries are local cache and not committed.
- running the script populates local image files so model training can load images from the CSV `image_path` column.

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

- `--smoke-test`: short run on a few rows to validate setup quickly.
- `--verify`: reports how many CSV rows currently have cached local JPGs.

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
