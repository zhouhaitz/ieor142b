## Cleaned Dataset Artifacts

Canonical cleaned datasets for team experiments:

- `MovieGenre_clean_with_images_full.csv` (strict `imdbId` matches; default)
- `MovieGenre_clean_with_images_full_manifest.json`

Local cache directory:

- `downloaded_posters/` (local cache populated by `scripts/fetch_cleaned_images.py`, not committed)

Use `MovieGenre_clean_with_images_full.csv` as the shared source when training on rows with confirmed image files.
