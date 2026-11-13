# Prompt for Codex on laptop B

Continue our PyTorch Image Captioning project on **laptop B**. The repository is `https://github.com/AskatQwerty/AMLNN.git`, branch `codex/50-experiments`. Use the same commit as laptop A. Read `README.md`, `environment-reference.json`, `requirements-captioning.txt`, `verification_report.json` and `FinalAML.ipynb` first.

Your task is to **train exactly the 25 experiments assigned to B**, save their complete results and prepare the return archive for A. This request authorizes that training after the environment, data and preflight checks pass. Proceed without asking again whether to start the already authorized 25 runs.

All project code must remain in **one notebook, `FinalAML.ipynb`**. Keep the code and explanatory text in English, with simple student-readable functions and useful blank lines. Do not introduce a separate training module or redesign the experiment.

## Inspect and prepare

- Check the GPU, CUDA availability, Python/PyTorch/torchvision versions and free disk space. The reference environment uses Python 3.11.18, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, NumPy 2.4.6, Pillow 12.3.0 and Matplotlib 3.11.2. Use the repository's environment files as the reference. Keep `pycocoevalcap==1.2`. If a CUDA-compatible installation is needed, use official PyTorch instructions and a compatible torch/torchvision pair; preserve an already working environment.
- Locate `caption_training_data.zip`, which A sends separately. If it is absent, ask for its local path and continue independent environment checks. Do not invent files, download another COCO copy or extract the CNN vectors again.
- Extract the archive contents beside the notebook so `cache/`, `data/coco2017/annotations/` and `bundle_manifest.json` are directly inside the project folder. Start Jupyter with this folder as its working directory. Run the notebook's manifest verification before using the cache; investigate mismatches rather than bypassing assertions.
- Allow sufficient space for expanded data, approximately 10 GiB or more of result folders for this half, predictions and an additional export archive. Use `RESULTS_ROOT` / `TRANSFER_DIR` on another drive if appropriate. Do not delete existing user data to make room.

## Run the assigned experiments

Set `LAPTOP_ID="B"`, `TRAINING_ONLY=True`, `RUN_TRAINING=False`, `RUN_PREFLIGHT=True` and `RUN_FINAL_TEST=False`. Run All and inspect the actual outputs. B6 must print exactly 25 B assignments. Both laptops use all 113,287 training images and all 5,000 validation images; the dataset itself is not split between workers.

The shared grid has five encoders (ResNet50, ResNet18, VGG19, MobileNetV2, EfficientNetV2-S), RNN/LSTM decoders and these configurations: `baseline`, `small_hidden`, `stacked`, `high_dropout`, `image_layernorm`. Do not change their definitions, seeds, batch size, captions, vocabulary, split, metrics or assignment. If an environment or memory limitation prevents the defined protocol, report the concrete issue and coordinate a shared fix; do not quietly create incomparable runs.

After the data, metric and model checks pass, set `RUN_PREFLIGHT=False` and `RUN_TRAINING=True` and run B's 25 experiments. The notebook already provides timers, best/last weights, full checkpoints, early stopping and resume. Preserve these. A short diagnostic model is discarded before the actual experiment; do not use it as a trained result.

Keep `RUN_FINAL_TEST=False`. Use validation CIDEr-D for early stopping and selection, with BLEU-1/BLEU-4 as additional metrics. Do not inspect test scores to choose configurations. Report actual failures and results; never fabricate scores, epoch counts, timings or successful checks.

If training is interrupted, retain the latest committed checkpoint and resume using the same configuration. Do not overwrite another run or start A's assigned half. Track progress until the authorized runs finish, unless the user explicitly pauses. Record the GPU/device and actual timings; different GPUs do not give directly comparable architecture runtimes.

## Return complete results

Once all 25 B runs are finished, set `RUN_TRAINING=False`. In section C, set `EXPORT_RESULTS=True` and run it to create `caption_results_B.zip`. Verify that the printed completed count is 25/25. Use a fresh archive filename if an earlier export exists.

Return the complete archive, not just weights. It must retain the finished run folders with `best.pt`, `last.pt`, plain weights, configuration, vocabulary, history, metrics and predictions. The notebook's exporter excludes temporary files and unfinished runs and creates a checksum manifest.

Do not push caches, images, weights or archives to GitHub. Do not merge or average independent models' weights. A will import this archive through the same notebook, display all 50 results and save one notebook for submission.

Your final report should state the Git commit used, GPU and environment, which 25 runs completed, any unresolved failures, measured timings, the results directory and the exact archive path to send to A. Clearly distinguish diagnostics from full training. If fewer than 25 finish, report the real count and resume state rather than claiming completion.
