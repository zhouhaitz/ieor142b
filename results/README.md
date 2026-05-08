# Experiment metrics (JSON)

Each experiment notebook should write one file here after training, e.g.:

| File | Owner / model |
|------|----------------|
| `tim/tim_vit_metrics.json` | Tim — ViT |
| `hannah_scratch_cnn_metrics.json` | Hannah — CNN from scratch |
| `xenia_resnet_metrics.json` | Xenia — ResNet fine-tune |

Use `experimentation.shared.save_metrics_json()` so required fields stay consistent.
Subfolders under `results/` are allowed for owner-specific organization (for example `results/tim/`).

```json
{
  "model_name": "vit",
  "seed": 42,
  "img_size": 224,
  "train_size": 0,
  "val_size": 0,
  "test_size": 0,
  "best_val_f1": 0.0,
  "test_f1": 0.0,
  "test_precision": 0.0,
  "test_recall": 0.0,
  "test_exact_match": 0.0,
  "num_epochs_run": 0,
  "notes": ""
}
```

Large artifacts (`*.pt`, plots) should stay local or be ignored by git unless the course requires them.
