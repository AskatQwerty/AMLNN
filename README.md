# Image Captioning: 75 experiments in one notebook

`FinalAML.ipynb` contains the preparation, models, training, evaluation, timers, checkpoints and transfer helpers. The final submission is this one notebook with the combined results displayed.

The project now has **5 CNNs x 3 decoder families x 5 configurations = 75 experiments**. The encoders are ResNet50, ResNet18, VGG19, MobileNetV2 and EfficientNetV2-S. Every model uses the existing cached CNN vectors; no new image extraction is needed.

## Who runs what

| Laptop | Existing RNN/LSTM experiments | New Transformer experiments | Total assigned |
|---|---:|---:|---:|
| A | 25 | 25 | 50 |
| B | 25 | 0 | 25 |

**B should continue its original 25 runs.** The Transformer addition does not require B to update its notebook, interrupt training or start those runs again. Existing checkpoints and result archives from the shared 100-epoch plan remain compatible. Completed runs are reused.

The new notebook defaults to `LAPTOP_ID="A"`, `TRAIN_DECODERS=["transformer"]` and `RUN_TRAINING=False`. This selects only A's 25 new Transformer runs. If B later uses this version, set `LAPTOP_ID="B"` and `TRAIN_DECODERS=["rnn", "lstm"]`; the plan must select B's original 25 runs.

Both laptops use the same full training split of 113,287 images, validation split of 5,000 images, vocabulary and references. The separate 5,000-image labeled test stays unused during model selection. We assign whole experiments to laptops; we do not divide the training images.

## The five configurations

| Configuration | RNN/LSTM embedding / hidden | Transformer embedding / d_model | Layers | Dropout p | Extra image LayerNorm |
|---|---|---|---:|---:|---|
| `baseline` | 512 / 512 | 512 / 512 | 1 | 0.3 | No |
| `small_hidden` | 512 / 256 | 256 / 256 | 1 | 0.3 | No |
| `stacked` | 512 / 512 | 512 / 512 | 2 | 0.3 | No |
| `high_dropout` | 512 / 512 | 512 / 512 | 1 | 0.5 | No |
| `image_layernorm` | 512 / 512 | 512 / 512 | 1 | 0.3 | After image projection |

All Transformer configurations use **8 attention heads** and **feedforward width 2048**. The smaller Transformer changes its model width, including its token embedding; the recurrent small-hidden variant keeps its embedding at 512. Parameter counts are recorded in the comparison.

The Transformer is an **autoregressive decoder with an image prefix**. A learned projection converts the cached global image vector into the first token. Caption tokens follow it and receive learned position embeddings. A causal attention mask blocks future words, and a padding mask excludes padding keys. Generation starts with the image and `<start>`, then uses previous predictions until `<end>` or the length limit.

The PyTorch self-attention blocks use a causal mask to act as a decoder-only Transformer. There is no separate cross-attention module or spatial image attention: the CNN cache contains one global vector per photograph. Transformer blocks already contain LayerNorm; `image_layernorm` adds normalization specifically after the image projection. Transformer dropout applies inside the blocks and at the embedding/output stages. RNN/LSTM retain their existing external dropout and zero built-in recurrent dropout.

## Start A's Transformer runs

1. Open Jupyter in the folder containing `FinalAML.ipynb`, using the CUDA environment. Keep `TRAINING_ONLY=True` so preparation loads the existing cache.
2. Check `LAPTOP_ID="A"`, `TRAIN_DECODERS=["transformer"]`, `RUN_TRAINING=False` and `RUN_FINAL_TEST=False`. Run All and confirm that the plan selects **25 Transformer runs**.
3. Set `RUN_TRAINING=True` and Run All when ready. Each configuration performs a batch/backward/generation and small-set check before full training. The main model is created again after that check.

All families use Adam at 0.001, batch size 64, **MAX_EPOCHS=100** and **PATIENCE=3** on validation CIDEr-D. The limit is a maximum; early stopping can finish much sooner. The caption-selection policy, split, vocabulary and metric implementation stay shared. Keep `pycocoevalcap==1.2`; see `requirements-captioning.txt` and `environment-reference.json` for the environment.

Results stay in `experiments/all_encoders_v5_100epochs/`, with unique names such as `resnet50_transformer_baseline_seed42`. Full `best.pt` / `last.pt` checkpoints contain optimizer, AMP scaler, RNG and early-stopping state. `best_weights.pt` / `last_weights.pt` contain plain model weights. Resume with the same configuration; strict compatibility checks remain enabled.

Timers report epoch time, validation time, active total time and the current training session. Active run time includes training, validation and checkpoint writing, while pauses and preflight checks are excluded. Device information is saved; timing from different GPUs is not an architecture-only comparison. Check free space for new weights and archives, and use `RESULTS_ROOT` or `TRANSFER_DIR` on another drive when needed.

## Data and laptop B

The shared branch remains:

```text
https://github.com/AskatQwerty/AMLNN/tree/codex/50-experiments
```

For a new checkout:

```text
git clone --branch codex/50-experiments https://github.com/AskatQwerty/AMLNN.git
```

The existing `caption_training_data.zip` remains valid. Transfer it separately through USB or cloud storage and extract it beside the notebook. The project then contains `cache/`, `data/coco2017/annotations/` and `bundle_manifest.json`. It expands to about 4.4 GiB. Raw photographs and another CNN extraction are unnecessary for training. The first Run All verifies transferred file hashes before loading the prepared captions.

If B is already training with its earlier 100-epoch notebook, let that run continue. If using the new notebook, select `LAPTOP_ID="B"` and `TRAIN_DECODERS=["rnn", "lstm"]`, keep test evaluation off, and use the existing result directory. B still completes exactly 25 recurrent experiments.

## Collect results and submit one notebook

1. B exports its completed experiment folders with the existing notebook. The original `caption_results_B.zip` format is still accepted by A. If B uses this updated notebook with selected recurrent decoders, the default name is `caption_results_B_rnn_lstm.zip`. Confirm that all **25/25** B runs are included.
2. Send the complete archive, not just weights: it includes configurations, vocabulary, histories, metrics, predictions and resume checkpoints. Keep large caches, weights and archives out of GitHub.
3. On A, initialize the updated notebook with training off. In section C, set `RECEIVED_RESULTS_ARCHIVE` to the actual ZIP path and `IMPORT_RESULTS=True`. The importer checks identities and hashes and refuses conflicting folders.
4. Rerun the comparison and learning-curve cells. The full table now covers **75 runs**; pending scores stay blank. Select finalists using validation, then evaluate the labeled test after the selection is fixed.
5. Save A's `FinalAML.ipynb` with the combined outputs as the single submission notebook. Retain the result folders for reproducibility; no weight averaging or notebook merging is needed.

In the new export controls, `EXPORT_SCOPE="selected"` exports the currently selected decoder families; `EXPORT_SCOPE="all_assigned"` exports this laptop's complete assigned set. For A, these are 25 Transformers or all 50 assigned runs. Export includes only completed runs, so check the printed count. Use a fresh archive filename for a later export.

`verification_report.json` records the recurrent-model checks, and `transformer_verification.json` records the Transformer checks actually performed. These development diagnostics are separate from full experiment results. For B's coding assistant, use [TEAMMATE_CODEX_PROMPT.md](TEAMMATE_CODEX_PROMPT.md).
