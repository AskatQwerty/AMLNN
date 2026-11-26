# Caption one image with the released model

This CLI uses the selected frozen fusion model: epoch 11, beam size 5, length penalty 0, maximum 33 generated tokens. It extracts features for one original image in memory. It does not need COCO annotations, prepared captions, the warm-start model or old numeric feature arrays. It does not train or write feature caches.

The unfinished November 26 fine-tuning checkpoint is a separate experiment. The CLI below loads the original released model.

## Setup

Use Python 3.11 with PyTorch 2.13.0, torchvision 0.28.0 and Pillow. The region extractor checks the PyTorch/torchvision versions because they are part of the saved feature contract. Choose appropriate CPU or CUDA builds for your machine. The project used CUDA 12.6 on an RTX 4060 Laptop GPU.

[inference requirements](requirements-inference.txt) list the CLI dependency versions; [the reproduction guide](../docs/REPRODUCING.md) covers notebook dependencies and the recorded environment. No clean-environment installation is claimed to have been tested.

Download `final_model_inference.zip` from the [original release](https://github.com/AskatQwerty/AMLNN/releases/tag/final-model-2026-11-16). Extract it under `local_artifacts/final_model/` in the repository:

```text
local_artifacts/final_model/
  selected_model/
    weights.pt
    inference_config.json
    source_run_config.json
    vocabulary.json
  encoder_cache/checkpoints/
    resnet50-11ad3fa6.pth
    fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

Historical absolute paths in JSON metadata are provenance; the loader uses the model directory supplied below. Keep `caption_image.py`, `models.py`, `image_features.py`, `region_features.py` and `beam_search.py` together in `inference/`.

## Run from the repository root

```powershell
python inference/caption_image.py --image "photo.jpg" --model-dir "local_artifacts/final_model/selected_model" --torch-cache "local_artifacts/final_model/encoder_cache" --device auto --output "caption.json"
```

Replace `photo.jpg` with your image path. `--device auto` selects CUDA when available; `--device cpu` forces CPU and `--device cuda` requires CUDA. The output includes the caption, model checksum, decoding settings, device, and separate model-loading and inference times.

For a displayed photo, caption and timer, use [Final_Model_Photo.ipynb](../Final_Model_Photo.ipynb).

## Encoder cache and downloads

The two frozen encoder files are checked against the saved SHA-256 values. With `--torch-cache <cache-root>`, they must be in `<cache-root>/checkpoints/`. Without that option, the loader uses the normal PyTorch hub cache, typically `~/.cache/torch/hub/checkpoints/`.

Downloads are disabled unless `--allow-download` is supplied. That flag restores only the pretrained CNN files; it cannot download the trained caption decoder. The inference release includes both CNN files for offline use.

## What stays identical

ResNet50 uses EXIF-corrected RGB, centered 224×224 padding, the saved padding color and ImageNet normalization. The detector uses its own 800/1333 transform in FP32. Up to 36 selected final boxes are re-pooled to region vectors. The spatial and regional features reproduce feature-cache float16 rounding; the global vector is pooled before rounding.

The decoder receives the global prefix, 49 spatial positions and up to 36 region vectors with padding masks. Detector class names are not decoder inputs. Both CNNs and the decoder run in evaluation mode with gradients disabled.

A [CUDA smoke check](verification.json) reproduced the saved caption for COCO image 119911 without feature-cache access. A later [CPU check](../metadata/cpu_photo_smoke_20261126.json) reproduced the recorded caption for the bundled three-bear image, 113403, in about 6.3 seconds plus 1.7 seconds for loading. These are individual checks, not a benchmark or a fresh-environment installation test. Single-image or CPU output may differ slightly from original batched GPU evaluation because of numerical kernels. This CLI generates captions; it does not calculate validation or test metrics.

Original cached-feature training needs numeric arrays extracted again. The new image-based joint fine-tuning notebook computes features during training and needs original images instead. Its full joint `last.pt` checkpoint is incompatible with this inference-only `weights.pt` loader.
