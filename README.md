# Image Captioning: 50 experiments on two laptops

`FinalAML.ipynb` contains all preparation, model, training, evaluation and transfer code. There is no separate Python training module. The final submission is this one notebook with the combined results displayed.

The experiment grid is **5 frozen CNN encoders x 2 decoders x 5 configurations = 50 runs**. Training uses saved image vectors, so the second laptop does not need the original photographs or another CNN extraction.

| Configuration | Embedding | Hidden size | Layers | External dropout | Image LayerNorm |
|---|---:|---:|---:|---:|---|
| `baseline` | 512 | 512 | 1 | 0.3 | No |
| `small_hidden` | 512 | 256 | 1 | 0.3 | No |
| `stacked` | 512 | 512 | 2 | 0.3 | No |
| `high_dropout` | 512 | 512 | 1 | 0.5 | No |
| `image_layernorm` | 512 | 512 | 1 | 0.3 | After image projection |

Encoders: ResNet50, ResNet18, VGG19, MobileNetV2 and EfficientNetV2-S. Decoders: RNN and LSTM. Recurrent-block dropout remains zero in every configuration; external dropout is applied after word embeddings and before the output layer.

## Equal division

Laptop A and laptop B each receive **25 runs**, including five per CNN and five per configuration. A has 13 LSTM / 12 RNN runs; B has 12 LSTM / 13 RNN runs. Section B6 prints the exact assignment and saves `experiment_plan.json`. Only `LAPTOP_ID` changes between laptops; both use the same Git commit.

Both laptops train on the **same full split**: 113,287 training images and 5,000 validation images. The separate 5,000-image labeled test stays unused during selection. We divide experiments, not photographs. Equal run counts do not guarantee equal runtime on different GPUs.

## Set up laptop B

1. Clone the shared branch:

   ```text
   git clone --branch codex/50-experiments https://github.com/AskatQwerty/AMLNN.git
   ```

2. Receive `caption_training_data.zip` separately through USB or cloud storage. It is intentionally not in Git. Extract its contents beside `FinalAML.ipynb`, producing:

   ```text
   AMLNN/
     FinalAML.ipynb
     bundle_manifest.json
     cache/
     data/coco2017/annotations/
   ```

3. Use a Python environment with CUDA-enabled PyTorch, matching torchvision and the notebook's supporting packages. See `environment-reference.json`; supporting packages are pinned in `requirements-captioning.txt`. Keep `pycocoevalcap==1.2`. Install CUDA-compatible torch/torchvision separately if needed; do not replace a working CUDA build with a CPU build. Open Jupyter in the directory containing the notebook so `Path.cwd()` points there.

4. In Settings, set `LAPTOP_ID="B"`, keep `TRAINING_ONLY=True` and `RUN_FINAL_TEST=False`. Initially set `RUN_TRAINING=False` and `RUN_PREFLIGHT=True`, then Run All. This verifies the transferred data and performs batch/backward/generation/small-set checks for B's actual configurations. The plan must show exactly 25 assigned runs.

5. After the checks pass, set `RUN_PREFLIGHT=False` and `RUN_TRAINING=True`, then Run All. Each run also checks its configuration before training. Leave model settings, batch size, seed and split unchanged. If memory does not fit, coordinate a shared configuration change rather than silently making B's protocol different.

The first cache verification checks file hashes. Later runs reuse it while file sizes and modification times match. Training-only mode skips image preparation and CNN extraction.

## Results, timing and interruptions

Results go to `experiments/all_encoders_v4_50/`, one folder per uniquely named run. Each completed run includes `best.pt`, `last.pt`, plain `best_weights.pt` / `last_weights.pt`, configuration, vocabulary, history, summary and predictions.

Use the full `last.pt` checkpoint to resume training: it also contains optimizer, AMP scaler, RNG and early-stopping state. After an interruption, rerun with the same settings; completed runs are reused. Do not train the same run on both laptops at once.

The notebook prints epoch time, validation time, active total time and the current session's wall time. Active run time includes training, validation and checkpoint writing; pauses and preflight checks are excluded. GPU names are saved. Do not interpret timing differences across GPUs as architecture effects alone.

Check free disk space before training. A 25-run half can require roughly **10 GiB plus prediction files and archive copies**; retaining both halves requires over 20 GiB for results in addition to the prepared data. `RESULTS_ROOT` and `TRANSFER_DIR` can point to another drive. The training bundle expands to about 4.4 GiB.

## Return B's work and submit one notebook

1. After B finishes, set `RUN_TRAINING=False`. In section C, set `EXPORT_RESULTS=True` and run that cell. It creates `transfer/caption_results_B.zip`. Confirm the printed count is **25/25**. Use a new archive filename for another export. Incomplete runs and temporary files are excluded.
2. Send that archive to A. **Weights alone are insufficient**: the complete folders preserve metrics, curves, predictions, configuration and resume state. Do not commit archives, caches or checkpoint files to GitHub.
3. On A, initialize the same notebook with training off. Set `RECEIVED_RESULTS_ARCHIVE` to the received ZIP and `IMPORT_RESULTS=True` in section C, then run that cell. The importer checks versions and hashes and refuses conflicting results.
4. Rerun B8 to display the combined 50-row table and B9 for selected learning curves. Scores remain blank for runs that have not produced results. Select finalists using validation only; then use B10 for the held-out test. Keep test evaluation off until that decision is fixed.
5. Save A's `FinalAML.ipynb` with the combined outputs. This is the single notebook to submit. Keep the result folders alongside it for reproducibility; no weight averaging or notebook merging is required.

`verification_report.json` records checks actually performed for this implementation. Section B11 also displays a local verification report when one is present. These diagnostics do not represent completion of the 50 full training runs.

For the second laptop's coding assistant, use [TEAMMATE_CODEX_PROMPT.md](TEAMMATE_CODEX_PROMPT.md).
