# Hannah CNN Mini Tests

This folder contains a small inference test for Hannah's scratch CNN from
`experimentation/hannah.ipynb`.

The test does not retrain the model. It loads the saved checkpoint produced by
the notebook and runs predictions on one poster image.

## Files

- `checkpoints/hannah_test.py`: command-line mini test for one poster.
- `checkpoints/hannah_scratch_cnn.pt`: expected trained checkpoint path. This
  file is created by the full-training cell in `experimentation/hannah.ipynb`.

## Run The Mini Test

From the repo root:

```bash
python3 experimentation/hannah/checkpoints/hannah_test.py
```

By default, the script:

- looks for `experimentation/hannah/checkpoints/hannah_scratch_cnn.pt`
- chooses one deterministic random poster from `cleaned/downloaded_posters`
- uses the notebook's evaluation preprocessing
- prints top genre probabilities and labels above the checkpoint threshold

Run against a specific poster:

```bash
python3 experimentation/hannah/checkpoints/hannah_test.py --image cleaned/downloaded_posters/10040.jpg
```

Run against a different checkpoint:

```bash
python3 experimentation/hannah/checkpoints/hannah_test.py --checkpoint path/to/hannah_scratch_cnn.pt
```

Adjust the label threshold:

```bash
python3 experimentation/hannah/checkpoints/hannah_test.py --threshold 0.40
```

## Expected Checkpoint

The full-training cell in `experimentation/hannah.ipynb` saves a checkpoint with:

- `model_state_dict`
- `classes`
- `num_genres`
- `img_size`
- `best_threshold`

If the checkpoint is missing, rerun the Hannah notebook's full-training cell or
copy the saved checkpoint into:

```text
experimentation/hannah/checkpoints/hannah_scratch_cnn.pt
```

## Interpreting Output

The model is multi-label, so it can predict more than one genre for a poster.

- `Predicted labels`: every genre whose probability is at or above the threshold.
- `Top probabilities`: the highest-scoring genres, even if they are below the threshold.
- If no labels pass the threshold, inspect the top probabilities or try the
  notebook's calibrated threshold, usually stored in the checkpoint.

