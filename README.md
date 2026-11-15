# Image Captioning: 79 experiments in one notebook

`FinalAML.ipynb` contains all preparation, models, training, evaluation, timers, checkpoint and transfer code. The original **75 completed experiments are preserved**, including B's 25 imported recurrent runs. The plan now adds **four ResNet50 depth experiments**: LSTM with 3 or 4 layers and the causal image-prefix Transformer with 3 or 4 layers. These four new runs require training; their scores stay blank until results are saved.

The original grid is 5 CNNs x 3 decoder families x 5 variants. Its configurations, names, assignments and compatibility IDs are unchanged. The CNNs are ResNet50, ResNet18, VGG19, MobileNetV2 and EfficientNetV2-S. The depth study reuses the saved ResNet50 global vectors; no new image extraction is needed.

## Run the four new experiments

Open Jupyter beside `FinalAML.ipynb` using the working CUDA environment. The notebook defaults to:

```python
LAPTOP_ID = "A"
TRAIN_DECODERS = ["lstm", "transformer"]
TRAIN_VARIANTS = ["depth_3", "depth_4"]
TRAINING_ONLY = True
RUN_PREFLIGHT = False
RUN_TRAINING = False
RUN_FINAL_TEST = False
```

Run All and confirm that B6 selects **exactly four new ResNet50 runs**. Then change the existing `RUN_TRAINING` flag to `True` and Run All when ready to train. Keep the separate `TRAINING_ONLY=True` line: it loads prepared data and cached CNN vectors. The default filters exclude all 75 completed configurations from training.

Every new run performs a batch/backward/greedy check and fixed tiny-set memorization with dropout disabled, then creates a fresh main model. Transformer also checks its masks and image conditioning. Training uses Adam 0.001, batch size 64, at most **100 epochs**, early-stopping **patience 3** on validation CIDEr-D, gradient clipping 5.0 and seed 42.

The temporary small-set check allows up to **2,000 steps for the depth variants**, stopping early when loss is at most 0.1 and greedy captions match the fixed targets exactly. The original 75 configurations retain their 200-step diagnostic limit. This separate check does not change the main experiment's training budget or initialize its model.

## Matched depth comparison

Both decoder families use the following ResNet50 configurations:

| Variant | Layers | Width and embedding | Dropout | Status before new training |
|---|---:|---:|---:|---|
| `baseline` | 1 | 512 | 0.3 | Existing result |
| `stacked` | 2 | 512 | 0.3 | Existing result |
| `depth_3` | 3 | 512 | 0.3 | New experiment |
| `depth_4` | 4 | 512 | 0.3 | New experiment |

Only layer count changes within each decoder's depth comparison. LSTM keeps built-in recurrent dropout at zero and uses separate dropout layers. Transformer keeps 8 heads, feed-forward width 2048, pre-normalization, learned positions, and its global image-prefix token. Its causal mask blocks future words; padding keys are masked. There is no separate spatial-image or cross-attention module.

All runs share 113,287 training images, 5,000 validation images, the saved vocabulary and human references. Training uses one random caption per image per epoch. Validation loss uses a fixed caption; metrics use all original references. Keep `pycocoevalcap==1.2` and the saved tokenizer. BLEU-1/BLEU-4 use the 0–1 scale; CIDEr-D uses the implementation's native scale, including its factor 10. The separate 5,000-image labeled test remains unused during model selection.

## Results, timers and resume

Results stay in `experiments/all_encoders_v5_100epochs/`. New names include `resnet50_lstm_depth_3_seed42` and `resnet50_transformer_depth_4_seed42`. Each run saves `best.pt` and `last.pt` with model, optimizer, AMP scaler, RNG and early-stopping state. `best_weights.pt` and `last_weights.pt` contain plain weights. Configuration, vocabulary, history, summary and validation predictions are also retained. Resume interrupted runs with the same settings; completed runs are reused.

Timers show epoch time, validation time, active total time and session time. Active run time includes training, validation and checkpoint writing, while pauses and preflight are excluded. GPU/device details are recorded. Do not attribute timing differences across devices solely to architecture. Keep space for checkpoints and archives; `RESULTS_ROOT` and `TRANSFER_DIR` can point to another drive.

B8 shows the full **79-run** table and an eight-row ResNet50 depth comparison. The family summary and heatmap use only completed runs. After the new runs finish, ResNet50 LSTM/Transformer have seven candidate variants; other CNN/decoder pairs retain five. These displayed winners are maxima, not averages across variants or seeds. A small validation difference from one seed does not establish a reliable advantage. B9 plots the best completed model in each decoder family.

## Assignments and transfer

A owns its original 50 runs plus the four depth additions: **54**. B retains its original **25** recurrent runs, already collected on A. No new B training is required. To review or reproduce B's original selection with this notebook, use:

```python
LAPTOP_ID = "B"
TRAIN_DECODERS = ["rnn", "lstm"]
TRAIN_VARIANTS = None
```

`TRAIN_VARIANTS=None` selects all assigned variants for the chosen decoders. Omitting that change on B would leave the default depth selection empty. Both laptops use the same full data splits; experiments, not images, are assigned between them.

The original data archive and `caption_results_B.zip` remain compatible. Large caches, weights and archives are transferred separately and stay out of GitHub. A new machine can extract `caption_training_data.zip` beside the notebook, producing `cache/`, `data/coco2017/annotations/` and `bundle_manifest.json`. The notebook verifies the transferred data. Raw photographs are unnecessary for cached-feature training.

Section C defaults to no transfer actions. `EXPORT_SCOPE="selected"` exports completed runs selected by both decoder and variant filters; A's default archive is `caption_results_A_lstm_transformer_depth_3_depth_4.zip`. B's selection above produces `caption_results_B_rnn_lstm_all_variants.zip`. `EXPORT_SCOPE="all_assigned"` uses all 54 A configurations or all 25 B configurations, exporting only completed runs. Check the printed count and use a new filename for a later export. Set `RECEIVED_RESULTS_ARCHIVE` to the actual incoming path; the importer checks identities and refuses conflicting folders.

The branch remains [codex/50-experiments](https://github.com/AskatQwerty/AMLNN/tree/codex/50-experiments). Environment references are in `requirements-captioning.txt` and `environment-reference.json`. B11 displays available verification reports; these diagnostics are separate from completed experiments, and a report covers only its recorded configurations.

Submit one `FinalAML.ipynb` with the combined outputs. Retain the result folders for later analysis; no weight averaging or notebook merging is needed. Select finalists using validation, then use the labeled test after that choice is fixed. See [TEAMMATE_CODEX_PROMPT.md](TEAMMATE_CODEX_PROMPT.md) for B's preservation and reproduction instructions.
