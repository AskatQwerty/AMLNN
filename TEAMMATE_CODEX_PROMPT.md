# Prompt for Codex on laptop B

Continue our PyTorch Image Captioning project on **laptop B**. Repository: `https://github.com/AskatQwerty/AMLNN.git`, branch `codex/50-experiments`.

**Your task is still to finish B's original 25 RNN/LSTM experiments and return their complete results.** Laptop A now runs 25 additional Transformer experiments itself. Do not start Transformers on B or repeat completed recurrent runs.

## Preserve the current run

First inspect the existing notebook, result folders, configuration and running training process. If B is already training with the agreed 100-epoch notebook, let it continue. This extension does **not** require a Git update, kernel restart or retraining on B. Do not interrupt working training just to install the Transformer extension.

Existing recurrent configurations, checkpoints and result archives from `all_encoders_v5_100epochs` remain compatible with A's updated notebook. Keep exact configuration and version checks enabled; do not modify checkpoint identities to force a mismatch through. Older 15-epoch experiments belong to a different plan.

If you use the updated notebook after the current run ends, read `README.md`, the environment files and the actual notebook code. Set:

```python
LAPTOP_ID = "B"
TRAIN_DECODERS = ["rnn", "lstm"]
TRAINING_ONLY = True
MAX_EPOCHS = 100
PATIENCE = 3
RUN_FINAL_TEST = False
```

The updated notebook defaults to A and Transformer-only selection, so these B settings matter. The printed plan must select B's original **25** recurrent runs. The complete project has 75 runs: A's original 25, B's original 25 and A's 25 Transformers.

## Environment and data

- Use the existing working CUDA environment. Check actual GPU, CUDA availability and free disk space. The reference environment and supporting packages are recorded in `environment-reference.json` and `requirements-captioning.txt`; keep `pycocoevalcap==1.2`.
- If setup is still needed, use a compatible CUDA-enabled PyTorch/torchvision pair and the correct Jupyter kernel. Consult official PyTorch installation instructions if necessary; do not replace a working GPU installation with a CPU build.
- The original `caption_training_data.zip` remains valid. If it is missing, ask for its local path while continuing independent environment checks. Do not download another COCO dataset or extract CNN vectors again.
- Extract the archive beside `FinalAML.ipynb`, producing `cache/`, `data/coco2017/annotations/` and `bundle_manifest.json`. Start Jupyter in this project folder. Use the notebook's hash verification; investigate a mismatch instead of bypassing it.
- Both laptops use all 113,287 training images and all 5,000 validation images. The separate 5,000-image labeled test remains held out. The training images are not divided between laptops.
- Allow space for result folders, predictions and the return archive. `RESULTS_ROOT` and `TRANSFER_DIR` can point to another drive. Do not delete unrelated user data to create space.

## Finish B's experiments

For a fresh start or an updated notebook, initially set `RUN_TRAINING=False` and `RUN_PREFLIGHT=True`, then Run All. Verify the data and the selected 25 configurations. After these checks pass, set `RUN_PREFLIGHT=False` and `RUN_TRAINING=True`. The user already authorized these 25 runs; no additional start confirmation is needed.

Keep the original RNN/LSTM configuration names: `baseline`, `small_hidden`, `stacked`, `high_dropout`, `image_layernorm`, across ResNet50, ResNet18, VGG19, MobileNetV2 and EfficientNetV2-S. Do not change assignments, architecture definitions, seed, batch size, captions, vocabulary, split or metrics. If a real environment or memory limitation prevents the agreed setup, report it and coordinate a shared fix rather than silently changing the protocol.

Use validation CIDEr-D for early stopping with patience 3 and a maximum of 100 epochs. BLEU-1 and BLEU-4 are additional metrics. Keep test evaluation off during training and selection. The notebook already includes timers, full checkpoints, plain weights and resume. A diagnostic small-set model must be discarded before the real experiment.

After an interruption, preserve the latest committed `last.pt` and resume with the same configuration. Completed runs are reused. Continue the authorized work until all 25 finish unless the user explicitly pauses it. Never invent metrics, timings, epoch counts or successful checks.

## Return the complete results

After all B runs finish, set `RUN_TRAINING=False` and export through section C. The earlier notebook creates `caption_results_B.zip`; this archive is still valid for A. The updated notebook with `TRAIN_DECODERS=["rnn", "lstm"]` and `EXPORT_SCOPE="selected"` creates `caption_results_B_rnn_lstm.zip`. Use a new filename if one already exists and verify the printed count is **25/25**.

Send the complete archive. It includes `best.pt`, `last.pt`, plain best/last weights, configuration, vocabulary, history, summary, metrics and predictions. Weights alone do not preserve enough information for the combined report. Do not push caches, checkpoint files, photographs or archives to GitHub.

A will import this archive into the updated notebook, add A's Transformer results and display the full 75-run comparison. We submit one `FinalAML.ipynb`; do not combine model weights or merge notebook outputs manually.

Your final report should state the actual Git commit/notebook version used, GPU and environment, which 25 runs completed, measured times, any unresolved failures, the results directory and the exact archive path to send to A. If fewer than 25 finish, report the true count and the saved resume state.
