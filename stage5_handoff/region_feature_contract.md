## Frozen Faster R-CNN region-feature contract

The primitive is `extract_region_batch(detector,images,max_regions=36,score_threshold=0.2,nms_iou=0.5)`. It returns one CPU dictionary per image. `build_frozen_region_detector(device,allow_download=False)` is the only constructor; nothing is downloaded or instantiated when the file is read. `load_region_image(path)` provides the original-size, EXIF-transposed RGB tensor in `[0,1]`.

The pinned model is `torchvision.models.detection.fasterrcnn_resnet50_fpn`, with `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. The code checks `torch==2.13.0` and `torchvision==0.28.0` (CUDA wheel suffixes are allowed). The default box head is `TwoMLPHead`: ROIAlign output passes through two fully connected/ReLU stages, ending at **1,024 features**. These details were inspected in the installed source and the [official model implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py).

The detector owns ImageNet normalization, aspect-preserving resize to min side 800/max side 1,333, and size-divisible-by-32 batch padding. There is no 224-pixel classification preprocessing. Inputs are float RGB tensors in `[0,1]`, as specified by the [official model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

Inference runs with frozen parameters, eval mode, no gradients, and autocast disabled. The RPN and native ROI head run once. Their final detections are still in transformed-image coordinates. The selection keeps score≥0.2, applies an additional class-agnostic NMS at IoU 0.5, then keeps up to 36 highest scores. ROIAlign and `box_head` are **run again on these selected final boxes on the same FPN tensors**. Proposal-feature row indices are never reused as final-detection indices. Original-image boxes are produced by one coordinate rescaling after feature extraction.

If selection produces zero regions, the full transformed image is pooled as one valid region. This includes the image only, excluding batch padding. Its fallback flag is True, score is zero and label is -1. `mask=True` always means valid; padding has zero features/boxes/scores and label -1. The decoder receives only region features and this mask. Predicted class IDs, category words and detection scores are retained only for inspection.

Recommended source cache files:

| File | Shape | Storage |
|---|---|---|
| `train2017_features.npy` / `val2017_features.npy` | `[N,36,1024]` | float16 |
| `*_mask.npy` | `[N,36]` | bool; True=valid |
| `*_num_regions.npy` | `[N]` | uint8 |
| `*_boxes.npy` | `[N,36,4]` | float32, original RGB xyxy |
| `*_scores.npy` | `[N,36]` | float32 |
| `*_labels.npy` | `[N,36]` | int64 |
| `*_fallback.npy` | `[N]` | bool |
| `*_image_ids.json` | ordered N IDs | JSON |

Extraction returns float32 visual tensors and checks that converting them to float16 stays finite. The future writer performs the storage conversion and records source fingerprints, ordered IDs, the exact contract, the full weight-file SHA256, library versions and completion/progress state. Begin with an extraction batch of **1**. Vectors alone for original train2017+val2017 (123,287 images) require approximately **8.47 GiB** in float16, or 16.93 GiB in float32; metadata arrays and temporary/resume files require additional space. The project train/validation/test still select rows through the existing saved split IDs.

### Pretraining exposure must be disclosed

The official COCO weights link to a [COCO detection training recipe](https://github.com/pytorch/vision/blob/main/references/detection/README.md), whose current [dataset loader selects train2017 for training](https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py). The project's labeled caption test was drawn from train2017. Therefore detector pretraining is expected to overlap with held-out caption images. The exact historical weight-training ID manifest has not been independently verified here, so do not claim that every held-out image was used or that the encoder never saw these images. This is an encoder-pretraining exposure limitation even though caption training/test IDs remain disjoint and caption labels are not passed into extraction. Report this detector baseline separately with its pretraining provenance; preserve the current split rather than silently relabeling validation/test.

### Checks actually performed

`test_region_features.py` uses a lightweight CPU stub detector and the installed torchvision CPU NMS operator. It passed final-box/feature row alignment, reuse of the same FPN, additional NMS, inclusive score filtering, top-K truncation, low-score and empty-detection fallback, one-time coordinate scaling, masks/padding, EXIF/RGB input handling, and the download-disabled guard. It also verifies that a later cache writer can convert the returned feature tensors to float16 without overflow.

No pretrained detector was constructed or run, no detector checkpoint was downloaded, no GPU was used, and no full region cache or decoder training was performed. A future real pretrained smoke test and extraction remain necessary.

Imports needed before this notebook fragment: `Path`, `hashlib`, `torch`, `torchvision`, `transforms`, `Image`, `ImageOps`, `fasterrcnn_resnet50_fpn`, `FasterRCNN_ResNet50_FPN_Weights`, and `nms` from `torchvision.ops`.

### Resumable cache writer

`region_features.py` and `region_cache.py` are standalone importable modules. The notebook embeds their definitions and collects imports at its beginning. `extract_region_cache(data,detector,cache_dir,raw_image_root,batch_size=1,producer_code_id=...)` creates only the original train2017/val2017 feature arrays. It returns the path dictionaries expected by the masked region dataset. `load_region_cache(...)` reads completed headers and metadata without loading full arrays; by default it requires the agreed 36-region, score-0.2, NMS-0.5 contract. Pass the protocol code fingerprint during both extraction and loading.

The extraction identity binds the full detector weight SHA256, source/annotation/ordered-ID hashes, all selection and transform settings, producer code fingerprint, library versions, extraction batch size and extraction device details. The builder computes the code fingerprint from the ASTs of both modules. If their source changes, regenerate the protocol fingerprint; an incomplete cache cannot silently mix producer versions, batches or devices. A completed cache can be consumed on a different training device through `load_region_cache`, which does not compare the consumer's GPU to the extraction GPU.

Checkpoint JSON advances only after all seven arrays are flushed. An incomplete batch is rewritten on resume. A ready checkpoint supports recovery if a process stops between publishing individual array files and writing final metadata. Completion checks every stored row for finite features/coordinates/scores, count/mask agreement, valid padding and fallbacks. Each finished array receives a streaming SHA256 hash. Reports include count minimum/mean/maximum, fallback rate, device and active time; full validation and hashing are included in that time.

Raw photos are checked against original annotation height/width, allowing the EXIF-related swap of dimensions. This rejects resized previews, but **does not re-hash or pixel-verify every raw JPEG**. Cache provenance therefore relies on the supplied prepared source ID and annotation identity plus this dimension check. Native NMS can choose different boxes for tied scores on different devices; cross-device bitwise feature/selection equality is not promised.

`region_cache_verification.json` records the actual CPU-stub tests: pause/resume preserves the committed prefix and overwrites an invalid tail; masks, counts, IDs and all array hashes round-trip; mixed final/partial ready-state recovery works; completed loading needs no photos; incompatible code/batch/selection is rejected; and resized previews are rejected before detector execution. These tests do not validate a real pretrained detector or a real COCO region cache.
