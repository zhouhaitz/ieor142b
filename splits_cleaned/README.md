Fixed train / validation / test splits for the **strict** cleaned dataset (`imdbId` image match only).

- Source CSV: `cleaned/MovieGenre_clean_with_images_full.csv`
- Seed: `42`
- Files: `train_rows.csv`, `val_rows.csv`, `test_rows.csv`

Before training, fetch poster files locally:

```bash
python scripts/fetch_cleaned_images.py \
  --csv-path cleaned/MovieGenre_clean_with_images_full.csv \
  --out-dir cleaned/downloaded_posters \
  --timeout 6 \
  --retries 0
```

Regenerate from repo root:

```bash
python scripts/generate_splits.py \
  --csv-path cleaned/MovieGenre_clean_with_images_full.csv \
  --out-dir splits_cleaned \
  --seed 42
```
