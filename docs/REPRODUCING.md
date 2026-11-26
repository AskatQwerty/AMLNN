# Reproducing the project

This guide separates three tasks: captioning one image, reading archived results, and training. Commands below assume the repository root is the working directory.

For a small first example, open [Data_Visual_Examples.ipynb](../Data_Visual_Examples.ipynb). It uses a bundled COCO image, demonstrates resizing on CPU, and displays recorded detector boxes and captions. See the [visual guide](VISUAL_EXAMPLES.md). It does not download weights or run a new detector evaluation.

## Environment

Use Python 3.11. The recorded project environment used Python 3.11.18, PyTorch 2.13.0+cu126 and torchvision 0.28.0+cu126 on an RTX 4060 Laptop GPU. See [environment-reference.json](../environment-reference.json).

Install the matching PyTorch and torchvision builds for your machine before the remaining requirements. The region extractor checks their versions as part of its saved feature contract. A working CUDA driver is required for GPU use; do not replace an existing working driver merely to match this historical record.

```powershell
python -m pip install -r requirements-captioning.txt
python -m ipykernel install --user --name aml-captioning --display-name "Python (AML Captioning)"
python -m jupyterlab
```

Select the `Python (AML Captioning)` kernel. The requirements include pandas and JupyterLab for the new notebooks. They do not install PyTorch. For CLI-only use, [inference/requirements-inference.txt](../inference/requirements-inference.txt) lists the smaller dependency set; choose the CPU/CUDA builds deliberately.

These instructions describe the project environment and file layout. A completely fresh environment has not been validated end to end. The published [inference verification](../inference/verification.json) records a CUDA smoke check that reproduced the saved caption for COCO image 119911 without reading or writing feature arrays. A [November 26 CPU check](../metadata/cpu_photo_smoke_20261126.json) also reproduced the recorded caption for the bundled image 113403. The [publication checks](../metadata/publication_verification_20261126.json) record notebook syntax checks, data/checkpoint compatibility, and exact metric reproduction from saved predictions.

## Caption a photograph

Download `final_model_inference.zip` from the [original release](https://github.com/AskatQwerty/AMLNN/releases/tag/final-model-2026-11-16), and extract it under `local_artifacts/final_model/`. Check that it contains `selected_model/weights.pt` and `encoder_cache/checkpoints/` at those paths.

```powershell
python inference/caption_image.py --image "photo.jpg" --model-dir "local_artifacts/final_model/selected_model" --torch-cache "local_artifacts/final_model/encoder_cache" --device auto --output "caption.json"
```

Alternatively, open [Final_Model_Photo.ipynb](../Final_Model_Photo.ipynb), change the image path, and run the cells. Loading the models and generating the caption are timed separately. No training dataset or prepared-caption archive is needed.

The released model uses epoch 11, beam size 5, length penalty 0, and at most 33 generated tokens. The trained weight SHA-256 is:

```text
f97a57d7adbbbe6b3a6d77589ab2a785520e9a2c7b8a99c573dd9d13c44f7b6f
```

A full training checkpoint has different contents and a different hash. Do not pass `last.pt` to the inference-only `weights.pt` loader.

## Restore data for validation or training

Obtain the original COCO 2017 training and validation JPEGs and caption annotations separately. Set the dataset path in the notebook. A typical layout is:

```text
data/coco2017/
  train2017/
  val2017/
  annotations/
    captions_train2017.json
    captions_val2017.json
```

The project uses 113,287 training images, 5,000 validation images and 5,000 held-out caption test images selected from `train2017`. It does not use unlabeled official `test2017` for local metrics.

Download `prepared_caption_data.zip` from the [November 26 release](https://github.com/AskatQwerty/AMLNN/releases/tag/followup-experiments-2026-11-26) and extract it **into the repository root**, preserving its `cache/` folder. It restores `prepared.pkl`, `splits.json`, `vocab.json`, `test_captions.json`, `training_config.json`, `text_report.json`, `image_audit.json` and `holdout_validation.json` inside `cache/`. This is text, IDs and preparation metadata; it does not contain COCO JPEGs or numeric image-feature caches.

The committed [metadata/preparation/](../metadata/preparation/) directory preserves split IDs, vocabulary, preparation settings and audits. Its `training_config.json` records the original runtime paths under `cache/`; the metadata directory alone is not a replacement for the prepared-data archive. Keep the exact split and vocabulary when comparing existing checkpoints. Do not create a new random split or rebuild a different vocabulary.

The new notebooks resolve the project relative to the repository and expose configurable model/data paths. Review their setup cells before running. The historical notebooks preserve some original Windows paths; edit their setup paths for your machine.

## Repeat the resolution experiment

Open [FinalFusion_Resolution.ipynb](../FinalFusion_Resolution.ipynb) after restoring the inference model, prepared captions and COCO images. Its expensive validation switch is off by default. Review the evaluation size before enabling it: all 5,000 validation images are needed to reproduce the reported full comparison.

The four modes compare ResNet inputs of 224 or 448 pixels with detector short sides of 800 or 1200 pixels. No weights are updated. At 448 pixels, the larger ResNet feature map is pooled back to the decoder's original 7×7 grid. The decoder still receives 49 spatial tokens.

The [saved results](../results/followup_20261126/resolution/) include the full comparison, an earlier 100-image pilot, and the three-bear diagnosis. The pilot is preliminary historical evidence; the full comparison determines the conclusion.

## Continue the unfinished joint fine-tuning run

Open [FinalFusion_FineTune.ipynb](../FinalFusion_FineTune.ipynb). Training is off by default. Restore the original inference package and the prepared captions as above.

To resume the existing partial run, download `partial_finetune_resume.zip` from the [November 26 release](https://github.com/AskatQwerty/AMLNN/releases/tag/followup-experiments-2026-11-26) and extract it **into the repository root**. Its checkpoint belongs at:

```text
experiments/final_joint_finetune_v1/resnet50_fasterrcnn_transformer_layer4_seed42/last.pt
```

Keep the support files beside the checkpoint, including `best.pt`, configuration, vocabulary, summary, history and epoch-0 predictions. Here `best.pt` preserves the epoch-0 initialization as the current incumbent; it is not evidence of successful fine-tuning. Use the notebook's resume path; the joint `last.pt` contains updated encoder and decoder state, and is not interchangeable with the frozen-model inference package.

The saved state is **partway through epoch 1: 22,688 training images and 709 successful optimizer steps**. It is not a completed fine-tuned model. The run's best evaluated state is still the initialization at epoch 0. The separate 100-image assessment of `last.pt` is a diagnostic; it does not replace full-validation checkpoint selection.

Fine-tuning updates the caption decoder, ResNet50's final block, and selected detector components: its final backbone block, FPN and RoI box head. Early backbone blocks, BatchNorm state, RPN and detector predictor stay fixed. The microbatch is 4 images with accumulation over 8 steps, giving an effective batch of 32; decoder learning rate is 5e-5 and CNN learning rate is 1e-5. The notebook divides an epoch into five sequential training cells that save progress.

Image-based fine-tuning computes features while training. It does **not** require the old full-dataset numeric caches. It does require the original images and more GPU compute than frozen cached-feature training.

## Read or rerun the historical submission

The [complete PDF](AML_Final_Project_Complete.pdf) and [executed submission notebook](../AML_Final_Submission.ipynb) can be read without executing code.

To rerun their report cells, download `full_results_without_weights.tar.xz` from the [original release](https://github.com/AskatQwerty/AMLNN/releases/tag/final-model-2026-11-16):

```powershell
tar -xf full_results_without_weights.tar.xz
```

Copy the contents of `project_results/` into the repository root to restore `experiments/`, `saved_results/` and `submission_assets/`. The `recovered_archives/` folder preserves teammate evidence separately. The lightweight [results/experiments/](../results/experiments/) directory in Git does not contain every per-epoch prediction file needed by historical report cells.

The original `final_model_resume.zip` is for **cached-feature fusion training**, not the new joint fine-tuning run. Its README uses an earlier ZIP filename for the results archive; the published results are TAR.XZ. Numeric feature caches and superseded experimental weights were removed during the earlier disk cleanup. Re-extract features before resuming original cached-feature training. Deleted checkpoints require another saved copy.

## Repository layout and provenance

| Location | Purpose |
|---|---|
| Root notebooks | Executed project and user-facing experiment entry points |
| `inference/` | Portable single-image inference implementation |
| `src/` | Development models, data, metrics, extraction and decoding sources |
| `results/` | Lightweight histories, predictions, audits and comparison evidence |
| `metadata/preparation/` | Exact split, vocabulary and preprocessing metadata |
| `metadata/final_model/` | Selected frozen-model configuration and vocabulary |
| `metadata/release/` | Release asset records and checksums |
| `docs/` | Submission documents and usage guides |
| `stage5_handoff/` | Original teammate experiment protocol |

Some historical `src/` scripts retain local paths and scheduling defaults. In particular, `src/final_fusion/inference.py` is a notebook fragment; use `inference/caption_image.py` for standalone inference.

The original frozen encoder sources are:

| Encoder | Official download | SHA-256 |
|---|---|---|
| ResNet50, ImageNet1K V2 | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| Faster R-CNN ResNet50-FPN, COCO V1 | [fasterrcnn_resnet50_fpn_coco-258fb6c6.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |

Large model files, datasets and archives remain outside normal Git history. Use the release metadata to check downloaded files. Historical absolute paths in provenance records describe the original machine; they are not suggested installation paths.
