# Prompt for Codex on laptop B

Continue the PyTorch Image Captioning project on laptop B. Repository: `https://github.com/AskatQwerty/AMLNN.git`, branch `codex/50-experiments`.

**B's original 25 recurrent experiments have already been collected on A. No new B training is required for this extension.** A is adding four ResNet50 experiments: LSTM with 3 or 4 layers and causal image-prefix Transformer with 3 or 4 layers. The full plan has 79 runs: A owns 54 and B owns its unchanged 25.

## Preserve the existing work

Inspect the notebook, result folders and any active process first. Preserve completed results and valid resume checkpoints. Do not interrupt ongoing work merely to update the notebook, and do not start Transformer or depth experiments on B. Do not repeat completed runs.

All code remains in `FinalAML.ipynb`. Original configurations, run names, data, vocabulary, metric settings and compatibility IDs remain unchanged. The group is `all_encoders_v5_100epochs`; older 15-epoch experiments belong to a different plan. Keep strict identity checks enabled rather than editing checkpoints to bypass a mismatch.

If reviewing or reproducing B's original selection with the updated notebook, set:

```python
LAPTOP_ID = "B"
TRAIN_DECODERS = ["rnn", "lstm"]
TRAIN_VARIANTS = None
TRAINING_ONLY = True
RUN_PREFLIGHT = False
RUN_TRAINING = False
RUN_FINAL_TEST = False
MAX_EPOCHS = 100
PATIENCE = 3
```

The default notebook selects A's four depth runs. `TRAIN_VARIANTS=None` is required to select B's original variants; the printed B selection must contain exactly 25 runs. Run All with training off for review. Resume a genuinely incomplete original run only when that work is requested or already authorized; use the existing `RUN_TRAINING` switch and preserve the separate `TRAINING_ONLY` line. No additional training is needed if all B runs are finished.

## Keep the experiment protocol

Use the existing working CUDA environment and Jupyter kernel. Check `requirements-captioning.txt` and `environment-reference.json`; keep `pycocoevalcap==1.2`. Do not replace a working CUDA PyTorch installation with a CPU build. Report actual GPU/environment details and any unresolved mismatch.

Reuse the existing cache and `caption_training_data.zip`. If setting up a new project folder, extract the archive beside the notebook so `cache/`, `data/coco2017/annotations/` and `bundle_manifest.json` are present. Start Jupyter there and use its hash verification. Do not download COCO again or re-extract CNN vectors.

Both laptops use all 113,287 training images and 5,000 validation images. The separate 5,000 labeled test images remain held out. Do not change B's five original variants, seed, batch size, captions, vocabulary, split or metrics. Original training settings are Adam 0.001, batch size 64, maximum 100 epochs and patience 3 on validation CIDEr-D. The notebook includes preflight checks, timers, full checkpoints, plain weights and resume.

## Return or preserve results

If another export is requested, keep `RUN_TRAINING=False` and export through section C. The original `caption_results_B.zip` remains valid. The updated settings above with `EXPORT_SCOPE="selected"` produce `caption_results_B_rnn_lstm_all_variants.zip`. Check the printed completed count is 25/25 and use a new filename if one already exists.

Send complete experiment folders, including best/last checkpoints, plain weights, configuration, vocabulary, history, summary and predictions. Weights alone do not preserve the full report. Do not push caches, photographs, checkpoints or archives to GitHub. `TRANSFER_DIR` can point to another drive if archive space is limited.

A imports compatible results and displays the 79-run comparison, preserving the original 75 completed experiments. We submit one `FinalAML.ipynb`; do not average model weights or merge notebook outputs manually. State the actual notebook/Git version, GPU, completed run count, measured timings, result/archive paths and unresolved issues. Do not claim checks or new training that were not performed.
