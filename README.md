# PicToText: image captioning with frozen CNNs and a Transformer decoder

Final project by **Askat Aubakirov and Bakdaulet Dagystanov, BDA2403**, Astana IT University.

The final system combines frozen ResNet50 spatial features and frozen Faster R-CNN region features. A trained three-layer image-prefix Transformer supplies the starting decoder. New visual projections, normalization, position/source embeddings and gated cross-attention are trained with the decoder; both CNNs remain frozen.

## Start here

- [Complete submission PDF](docs/AML_Final_Project_Complete.pdf): presentation, full executed notebook, then research report (322 pages).
- [Executed submission notebook](AML_Final_Submission.ipynb): the complete project and saved outputs.
- [Presentation](docs/Presentation.pdf), [research report](docs/Research_Report.pdf), and [speaker notes](docs/Speaker_Notes.txt).
- [Final fusion notebook](FinalFusion.ipynb): extraction, training/resume, decoding search and inference helpers.
- [Portable image-captioning CLI](inference/README.md): run the selected model on a new photograph without old numeric feature caches.
- [Earlier global experiments](FinalAML.ipynb) and [spatial attention](SpatialAttention.ipynb).
- [Results](results/): configurations, histories, audits, comparisons and final selected predictions. The original stage-5 handoff remains in [stage5_handoff/](stage5_handoff/).

Opening the notebooks shows their saved outputs. No new training was run when preparing this archive. Training and extraction need the external dataset and weights; the Git checkout alone is not a complete local training environment.

## Results

All scores below use all 5,000 validation images and all human references. CIDEr-D is the native `pycocoevalcap==1.2` score, without an extra factor of 100; BLEU is in 0–1. We use the saved project regex tokenizer, not official COCO leaderboard PTB tokenization.

| Experiment | Validation CIDEr-D | Note |
|---|---:|---|
| Best original 75 RNN |0.6480|ResNet50, image LayerNorm|
| Best original 75 LSTM |0.8047|ResNet50, two layers|
| Best original 75 Transformer-prefix |0.8085|ResNet50, two layers|
| Global LSTM with three layers |0.8081|ResNet50|
| Global Transformer-prefix with three layers |0.8578|ResNet50|
| Spatial attention LSTM |0.7802|49 ResNet50 positions; did not improve the global LSTM|
| Region attention LSTM, no visual LayerNorm |0.0514|Collapsed to one repeated caption|
| Region attention LSTM, visual LayerNorm |0.8185|Best epoch 25; stopped at 28|
| Fusion, greedy |0.9380|Epoch 11, the same weights used in the next row|
| **Fusion, beam 5, length penalty 0** |**0.9801**|**Selected final system, epoch 11**|

The original search contains **75 runs: five CNNs × three decoder families × five configurations**. Four additional ResNet50 depth experiments bring the global total to 79. ResNet50 led all three decoder families in the original search. Three layers helped the strongest global models; four did not. Higher dropout (0.3→0.5) did not help any of the 15 matched encoder–decoder pairs.

The final selected model gives validation BLEU-1 **0.7413**, BLEU-4 **0.3162**, and CIDEr-D **0.9801**. Its local test scores are BLEU-1 **0.7492**, BLEU-4 **0.3262**, and CIDEr-D **0.9984**. The primary setting was selected on validation. Epoch 13 achieved the best greedy validation score, 0.9415, but epoch 11 with beam5 was the selected decoding configuration. The fusion run completed 13 epochs and was manually stopped during epoch 14; it did not reach early stopping.

## Data and interpretation limits

The project uses 113,287 training images, 5,000 validation images and 5,000 caption-held-out test images. Image IDs and exact image hashes do not overlap across splits. Near-duplicate absence was not established. The vocabulary has 10,000 training words plus four special tokens; training samples one human caption per image per epoch, while metrics retain all original references. Official unlabeled `test2017` is excluded from local caption scoring.

**The local test comes from COCO train2017, which is also the documented source of the detector's pretraining data.** Exact historical detector-training image membership was not reconstructed. These are caption-training holdout results, not proof of fully unseen-image generalization for the entire visual system.

The final fusion system achieved the strongest validation result, but its gain combines architecture changes, different visual pretraining and additional training. There is no matched control that continues only the prefix model for the same extra budget. Results use one seed, with no confidence intervals. Spatial versus regional LSTM comparisons also change normalization and pretraining. A spatial Transformer run was incomplete, CNN fine-tuning was not completed, and the planned formal 100-image human assessment was not performed. Curated caption examples illustrate improvements and failures; they are not error-rate estimates.

## Reproduction and disk cleanup

Use [requirements-captioning.txt](requirements-captioning.txt) with the compatible PyTorch/torchvision versions in [environment-reference.json](environment-reference.json). The recorded environment was Python 3.11.18, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126 and `pycocoevalcap==1.2`. Install a CUDA build suitable for the target machine; do not replace a working GPU driver just to match this record.

1. Open the desired notebook in Jupyter. Set its project and COCO dataset paths for your machine. Original path strings are retained in historical metadata for traceability.
2. Keep the exact saved split IDs, vocabulary and text settings under [metadata/preparation/](metadata/preparation/). Do not regenerate a different split or vocabulary when comparing with these results.
3. Obtain COCO train2017/val2017 JPEGs and caption annotations separately. They are not uploaded to this repository. Restore the original prepared-caption files or regenerate them using the notebook's preparation section and preserved metadata before training.
4. The requested disk cleanup retains the dataset, results and final model, but does **not** retain numeric feature caches or superseded experimental weights. Re-extract features before retraining or resuming cached-feature training. Older deleted checkpoints cannot be resumed without another saved copy.
5. For the final model, use the selected epoch 11 vocabulary/configuration and beam 5 with length penalty 0. New-image inference runs the two frozen encoders directly and does not require the full-dataset feature cache.

To rerun the report cells, restore the full result archive into the project directory so its original `experiments/...` paths exist. The lightweight `results/experiments/` tree in Git preserves the summaries for inspection; it does not contain every per-epoch prediction file expected by all report cells. The executed notebook can be read without rerunning any cell.

The notebooks are the main experimental record. For a new image, use [inference/caption_image.py](inference/caption_image.py), which resolves model and encoder paths from command-line arguments. A CUDA smoke check on COCO validation image 119911 reproduced its saved final caption without reading or writing feature arrays; CPU execution is implemented but was not tested in that check. See [verification](inference/verification.json).

After the planned inference release archive is available, extract it under `local_artifacts/final_model/` and run from the repository root:

```powershell
python inference/caption_image.py --image "photo.jpg" --model-dir "local_artifacts/final_model/selected_model" --torch-cache "local_artifacts/final_model/encoder_cache" --device auto --output "caption.json"
```

[src/](src/) preserves the development model, data, metrics, feature extraction and decoding sources. Some historical CLI files retain local paths and scheduling defaults; adapt them before execution. The older `src/final_fusion/inference.py` is a notebook fragment, whereas `inference/caption_image.py` is the portable standalone entry point.

## Final weights and full result archive

Publication of GitHub Release **`final-model-2026-11-16` is pending verification**. The planned [release destination](https://github.com/AskatQwerty/AMLNN/releases/tag/final-model-2026-11-16) will contain:

- `final_model_inference.zip`: selected epoch 11 weights/config/vocabulary, portable CLI and both frozen CNN weight files for offline inference.
- `final_model_resume.zip`: retained training checkpoints, warm-start compatibility dependency and run metadata. Re-extract the numeric features before resuming cached-feature training.
- `full_results_without_weights.zip`: all retained results, including per-epoch predictions, without obsolete weights, original COCO JPEGs or numeric feature arrays.

These uploads have not yet been verified; this README does not claim that the release assets are already available. Large weight files, datasets and transfer archives stay outside normal Git history.

The selected decoder metadata is in [metadata/final_model/](metadata/final_model/). The selected trained weight file has SHA-256 `f97a57d7adbbbe6b3a6d77589ab2a785520e9a2c7b8a99c573dd9d13c44f7b6f`. A full training checkpoint and an inference-only weight file have different hashes and contents; do not interchange their loaders.

The frozen encoder sources are the official torchvision weights used during extraction:

| Encoder | Official weight download | SHA-256 |
|---|---|---|
|ResNet50, `IMAGENET1K_V2`|[resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)|`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`|
|Faster R-CNN ResNet50-FPN, `COCO_V1`|[fasterrcnn_resnet50_fpn_coco-258fb6c6.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)|`258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`|

ResNet50 uses proportional resize and padding to 224×224 with ImageNet normalization. The detector uses its pretrained aspect-preserving transform. The final memory contains 49 grid locations and up to 36 valid region vectors, projected to 512 dimensions; padded regions are masked. Detector class names are not decoder inputs.

## Archive contents

The Git version contains summaries, histories, configurations, audit reports, comparison tables, selected final validation/test predictions and final documents. Large per-epoch prediction sets remain in the full results archive rather than being repeated in Git. No source experiment folders or user dataset files were changed while assembling the repository package. `package_manifest.json` records packaged-file hashes; it excludes itself and the `.git` directory.
