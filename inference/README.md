# Final image-captioning inference

This small CLI uses the final selected fusion model: epoch 11, beam size 5, length penalty 0, and a maximum of 33 generated tokens. It loads one original image and extracts its features in memory. It does not need the old feature arrays, experiment checkpoints, warm-start model or COCO annotations. It does not train or write feature caches.

Keep these five Python files together. The model directory must contain:

- `weights.pt`
- `inference_config.json`
- `source_run_config.json`
- `vocabulary.json`

The script resolves `weights.pt` locally. Historical absolute paths in the JSON metadata are not used to load checkpoints.

## Environment

Use the project's PyTorch environment: Python 3.11, PyTorch 2.13.0, torchvision 0.28.0 and Pillow. The region extractor checks the PyTorch/torchvision versions because they are part of the saved feature contract. Install the appropriate CPU or CUDA builds for your machine. The project used CUDA 12.6 and an RTX 4060 Laptop GPU.

The two frozen pretrained CNN files are required:

- `resnet50-11ad3fa6.pth` (ResNet50 ImageNet1K V2)
- `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` (Faster R-CNN COCO V1)

By default they are read from PyTorch's existing hub cache, normally `~/.cache/torch/hub/checkpoints/`. Both files are checked against the training-time SHA256 values. To use a portable local cache, put them in `<cache-root>/checkpoints/` and pass `--torch-cache <cache-root>`. Downloads are disabled unless `--allow-download` is supplied. Downloading restores only pretrained CNN weights, not the trained caption model.

## Run

```powershell
python caption_image.py --image "photo.jpg" --model-dir "selected_model" --device auto
```

```powershell
python caption_image.py --image "photo.jpg" --model-dir "selected_model" --device cuda --torch-cache "encoder_cache" --output "caption.json"
```

`--device cpu` provides a CPU fallback. `auto` selects CUDA when available. The JSON output includes the caption, model checksum, decoding settings, device and separate loading/inference times.

## What remains identical

The model, beam-search implementation and region extractor are copied from the verified project code. ResNet50 uses EXIF-corrected RGB, centered 224x224 padding, the saved padding color and ImageNet normalization. The detector uses its own 800/1333 transform in FP32. Up to 36 selected final boxes are re-pooled to region vectors. The spatial grid and region vectors reproduce the feature-cache float16 rounding, while the global vector is pooled before rounding. Only one image's tensors exist in memory.

The final decoder uses 49 spatial positions, up to 36 region vectors and the global image prefix. CNNs and the caption model are all in evaluation mode with gradients disabled. Single-image inference or CPU inference can differ slightly from the original batched GPU evaluation because of numerical kernels. This command is for generating captions; it does not calculate validation/test metrics.

For future training, feature arrays must be extracted again. The full resumable training checkpoint is separate from this inference package.
