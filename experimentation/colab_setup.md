# Colab GPU Setup for Tim ViT + LoRA (Drive-Synced Workflow)

This runbook matches `experimentation/tim.ipynb` using Option 3: Google Drive sync.

## Why this workflow

- Colab GPU runs on a remote machine, so it cannot read your Mac local files directly.
- Keeping `ieor142b` in Google Drive (synced from your laptop) lets you iterate locally and use latest files in Colab.
- Notebook bootstrap copies Drive repo to `/content/ieor142b_runtime` for faster training I/O.

## 1) Connect Colab GPU kernel

1. Open `experimentation/tim.ipynb`.
2. Select a Colab kernel in VS Code.
3. Prefer `L4`/`A100`; use `T4` if needed.

## 2) Set your Drive repo path in notebook cell 1

In cell 1, set:

- `DRIVE_REPO = Path("<your actual Drive path>/ieor142b")`

Example:

- `/content/drive/MyDrive/Berkeley/IEOR_142B/proj/ieor142b`

Then run cell 1 to:

- mount Drive
- copy repo to `/content/ieor142b_runtime`
- install dependencies
- set `IEOR142B_ROOT=/content/ieor142b_runtime`

## 3) Verify strict data files

After cell 1, these checks should be `True`:

- `experimentation/shared.py`
- `cleaned/MovieGenre_clean_with_images_full.csv`
- `splits_cleaned/train_rows.csv`

If any are `False`, your Drive path is wrong or missing files.

## 4) Poster images check

If `cleaned/downloaded_posters/*.jpg` is missing in runtime copy:

```python
!python scripts/fetch_cleaned_images.py --verify
!python scripts/fetch_cleaned_images.py
```

## 5) Runtime-budget policy (no pilot)

- Smoke run:
  - `run_tier="smoke"`
  - 2-3 epochs
  - wallclock cap ~35 minutes
- Full run:
  - `run_tier="full"`
  - 12-16 epochs max + early stopping
  - wallclock cap ~330 minutes

Keep one full run candidate unless smoke reveals a bug.

## 6) Keep Colab synced with local edits

Each new Colab session:

1. wait for local edits to sync to Drive,
2. rerun notebook cell 1 to refresh `/content/ieor142b_runtime`,
3. run training cells.

## 7) Expected outputs

- `results/tim/tim_vit_metrics.json`
- `results/tim/tim_vit_history.json`
- `checkpoints/tim_vit_lora_best.pt`
- `checkpoints/tim_vit_lora_last.pt`

`final_comparison.ipynb` reads `results/tim/tim_vit_metrics.json`.
# Colab GPU Setup for Tim ViT + LoRA (Drive-Synced Workflow)

This runbook matches `experimentation/tim.ipynb` using Option 3: Google Drive sync.

## Why this workflow

- Colab GPU runs on a remote machine, so it cannot read your Mac local files directly.
- Keeping `ieor142b` in Google Drive (synced from your laptop) lets you iterate locally and reuse the latest files in Colab.
- Notebook bootstrap copies Drive repo to `/content/ieor142b_runtime` for faster training I/O.

## 1) Connect Colab GPU kernel

1. Open `experimentation/tim.ipynb`.
2. Select a Colab kernel in VS Code.
3. Prefer `L4`/`A100`; use `T4` if needed.

## 2) Ensure your repo is in Drive

Put your project at a stable Drive path, for example:

- `/content/drive/MyDrive/Berkeley/IEOR_142B/proj/ieor142b`

In cell 1 of `tim.ipynb`, set:

- `DRIVE_REPO = Path("<your actual Drive path>")`

Run cell 1 to:

- mount Drive
- copy repo to `/content/ieor142b_runtime`
- install dependencies
- export `IEOR142B_ROOT=/content/ieor142b_runtime`

## 3) Verify strict data files

After running cell 1, all should print `True`:

- `experimentation/shared.py`
- `cleaned/MovieGenre_clean_with_images_full.csv`
- `splits_cleaned/train_rows.csv`

If any are `False`, your Drive path is wrong or files are missing from synced folder.

## 4) Poster images check

If `cleaned/downloaded_posters/*.jpg` is missing in runtime copy, run:

```python
!python scripts/fetch_cleaned_images.py --verify
!python scripts/fetch_cleaned_images.py
```

## 5) Runtime-budget policy (no pilot)

- Smoke run:
  - `run_tier="smoke"`
  - 2-3 epochs
  - wallclock cap ~35 minutes
- Full run:
  - `run_tier="full"`
  - 12-16 epochs max + early stopping
  - wallclock cap ~330 minutes

Keep one full run candidate unless smoke reveals a bug.

## 6) Keep Colab in sync with local edits

Each new Colab session:

1. sync latest local changes into Drive (wait for Drive sync to finish),
2. rerun cell 1 to refresh `/content/ieor142b_runtime`,
3. run training cells.

## 7) Expected outputs

- `results/tim/tim_vit_metrics.json`
- `results/tim/tim_vit_history.json`
- `checkpoints/tim_vit_lora_best.pt`
- `checkpoints/tim_vit_lora_last.pt`

`final_comparison.ipynb` consumes `results/tim/tim_vit_metrics.json`.
# Colab GPU Setup for Tim ViT + LoRA

Use this runbook for `experimentation/tim.ipynb` with strict cleaned data and bounded runtime.

## 1) Connect a Colab GPU kernel from VS Code

1. Open `experimentation/tim.ipynb`.
2. Click kernel selector -> `Select Another Kernel` -> `Colab`.
3. Create/select a Colab server with GPU runtime.
4. Prefer:
   - `L4` or `A100` if available (fastest),
   - `T4` otherwise.

Use your Berkeley account (`zhouhaitz@berkeley.edu`) for Pro access.

## 2) Verify runtime before training

Run this in a notebook cell:

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

Optional shell check:

```python
!nvidia-smi
```

If CUDA is not available, stop and reconnect to a GPU runtime.

## 3) Install notebook dependencies

Run once per fresh Colab session:

```python
!pip -q install timm peft scikit-learn pandas pillow matplotlib
```

## 4) Confirm project root and strict data assets

`tim.ipynb` expects:

- `cleaned/MovieGenre_clean_with_images_full.csv`
- `splits_cleaned/train_rows.csv`
- `splits_cleaned/val_rows.csv`
- `splits_cleaned/test_rows.csv`
- local images referenced by `image_path` (typically `cleaned/downloaded_posters/*.jpg`)

Quick check cell:

```python
from pathlib import Path
root = Path.cwd().resolve()
print(root)
print((root / "cleaned" / "MovieGenre_clean_with_images_full.csv").is_file())
print((root / "splits_cleaned" / "train_rows.csv").is_file())
```

If `False`, switch to correct project working directory before running training.

## 5) Runtime-budget strategy (no pilot)

- **Smoke tier**:
  - `run_tier="smoke"`
  - 2-3 epochs
  - hard wallclock cap ~35 min
  - objective: verify full pipeline + metrics JSON write
- **Full tier**:
  - `run_tier="full"`
  - 12-16 epochs max + early stopping
  - hard wallclock cap ~330 min
  - objective: final metrics for comparison notebook

Keep exactly one full run candidate unless smoke reveals a clear bug.

## 6) Credit protection rules

- Always enforce both:
  - epoch cap
  - wallclock cap
- Save checkpoints every epoch.
- Stop immediately after best test report is produced.
- Disconnect Colab server when done.

## 7) Expected outputs

After a successful run:

- `results/tim/tim_vit_metrics.json`
- `results/tim/tim_vit_history.json`
- `checkpoints/tim_vit_lora_best.pt`
- `checkpoints/tim_vit_lora_last.pt`

`final_comparison.ipynb` reads the metrics JSON from `results/`.
