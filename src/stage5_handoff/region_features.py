from pathlib import Path
import hashlib

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from PIL import Image, ImageOps


REGION_TORCH_PIN = "2.13.0"
REGION_TORCHVISION_PIN = "0.28.0"
REGION_WEIGHT_NAME = "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
REGION_CACHE_VERSION = "fasterrcnn_resnet50_fpn_coco_v1_regions_v1"
REGION_DIM = 1024


def region_feature_contract(max_regions=36, score_threshold=0.2, nms_iou=0.5):
    return {
        "version": REGION_CACHE_VERSION,
        "detector": "torchvision.models.detection.fasterrcnn_resnet50_fpn",
        "weights": REGION_WEIGHT_NAME,
        "torch": REGION_TORCH_PIN,
        "torchvision": REGION_TORCHVISION_PIN,
        "input": "EXIF-transposed RGB; ToTensor float32 in [0,1]; original aspect ratio and size",
        "detector_transform": {
            "min_size": 800, "max_size": 1333, "size_divisible": 32,
            "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
        },
        "compute_dtype": "float32; autocast disabled",
        "cnn_frozen": True,
        "detector_eval": True,
        "detector_score_threshold": 0.05,
        "detector_class_nms_iou": 0.5,
        "detector_max_detections": 100,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
        "rpn_nms_iou": 0.7,
        "selection": "Final class-NMS detections, score filter, class-agnostic NMS, descending score, top K",
        "score_threshold": float(score_threshold),
        "class_agnostic_nms_iou": float(nms_iou),
        "max_regions": int(max_regions),
        "feature_dimension": REGION_DIM,
        "feature_layer": "box_roi_pool + TwoMLPHead.fc6/ReLU/fc7/ReLU, rerun on selected FINAL boxes",
        "feature_storage_dtype": "float16 after finite/overflow validation; compute and returned features remain float32",
        "boxes": "xyxy in original EXIF-transposed RGB pixel coordinates; zero padding",
        "boxes_normalized": "xyxy divided by original [width,height,width,height]; reporting only",
        "mask": "bool; True means a real selected region, including the full-image fallback",
        "padding": "zero features/boxes/scores, label=-1, mask=False",
        "fallback": "If no boxes remain, ROI-pool the full transformed image; valid count=1, score=0, label=-1, fallback=True",
        "decoder_inputs": "visual features and valid-region mask only; no predicted class labels, scores or object words",
        "test_exposure_note": "COCO detector pretraining may include train2017 images used by the captioning holdout; do not claim an unseen-image encoder test.",
    }


def build_frozen_region_detector(device="cpu", allow_download=False):
    """No model is constructed or downloaded until this function is explicitly called."""
    if torch.__version__.split("+")[0] != REGION_TORCH_PIN or torchvision.__version__.split("+")[0] != REGION_TORCHVISION_PIN:
        raise RuntimeError(f"Use torch=={REGION_TORCH_PIN} and torchvision=={REGION_TORCHVISION_PIN} for this pinned region contract.")
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    filename = weights.url.rsplit("/", 1)[-1]
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file() and not allow_download:
        raise FileNotFoundError(f"Missing {checkpoint}. Downloading is disabled; set allow_download=True only when ready to obtain the pretrained detector.")
    if checkpoint.is_file():
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest.startswith("258fb6c6"), "Cached detector weights do not match COCO_V1."
    detector = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, progress=True,
        min_size=800, max_size=1333,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7,
    )
    detector = detector.to(device=torch.device(device), dtype=torch.float32).requires_grad_(False).eval()
    assert detector.roi_heads.box_head.fc7.out_features == REGION_DIM
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest.startswith("258fb6c6")
    detector.region_feature_metadata = {**region_feature_contract(), "weights_sha256": digest,
                                        "device": str(torch.device(device))}
    return detector


def load_region_image(path):
    """Do not apply the 224x224 classification resize, padding or normalization here."""
    with Image.open(path) as image:
        return transforms.ToTensor()(ImageOps.exif_transpose(image).convert("RGB"))


def _select_region_boxes(detection, transformed_size, max_regions, score_threshold, nms_iou):
    boxes, scores, labels = detection["boxes"], detection["scores"], detection["labels"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.shape == labels.shape == (len(boxes),)
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert torch.all((scores >= 0) & (scores <= 1))
    if len(boxes):
        assert torch.all(boxes[:, 2:] > boxes[:, :2]) and torch.all(labels > 0)
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(boxes):
        order = torch.argsort(scores, descending=True, stable=True)
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        keep = nms(boxes, scores, nms_iou)[:max_regions]
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep], "fallback": False}
    height, width = transformed_size
    return {
        "boxes": boxes.new_tensor([[0, 0, width, height]]),
        "scores": scores.new_zeros(1),
        "labels": torch.full((1,), -1, dtype=torch.long, device=boxes.device),
        "fallback": True,
    }


@torch.no_grad()
def extract_region_batch(detector, images, max_regions=36, score_threshold=0.2, nms_iou=0.5):
    """Return one CPU record per image; no files, caches or downloads are written."""
    if not isinstance(max_regions, int) or isinstance(max_regions, bool) or max_regions < 1:
        raise ValueError("max_regions must be a positive integer.")
    if not 0 <= score_threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("score_threshold and nms_iou must be in [0,1].")
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError("Pass a nonempty list of original RGB image tensors.")
    if detector.training or any(parameter.requires_grad for parameter in detector.parameters()):
        raise ValueError("The detector must be frozen and in eval() mode.")
    device = next(detector.parameters()).device
    original_sizes = []
    prepared_images = []
    for image in images:
        if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) < 1:
            raise ValueError("Each image must have shape [3,H,W].")
        if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Use unnormalized RGB float tensors in [0,1].")
        original_sizes.append(tuple(image.shape[-2:]))
        prepared_images.append(image.to(device=device, dtype=torch.float32))

    # All detector/ROI computation stays float32, even inside a caller's AMP context.
    with torch.autocast(device_type=device.type, enabled=False):
        transformed, _ = detector.transform(prepared_images, None)
        fpn_features = detector.backbone(transformed.tensors)
        if isinstance(fpn_features, torch.Tensor):
            fpn_features = {"0": fpn_features}
        proposals, _ = detector.rpn(transformed, fpn_features, None)
        detections, _ = detector.roi_heads(fpn_features, proposals, transformed.image_sizes, None)
        assert len(detections) == len(images)
        selected = [_select_region_boxes(detection, size, max_regions, score_threshold, nms_iou)
                    for detection, size in zip(detections, transformed.image_sizes)]
        selected_boxes = [item["boxes"] for item in selected]

        # NMS/regression changes the proposal rows. Re-pool FINAL selected boxes;
        # never index the earlier proposal feature matrix using final detection IDs.
        roi_maps = detector.roi_heads.box_roi_pool(fpn_features, selected_boxes, transformed.image_sizes)
        roi_vectors = detector.roi_heads.box_head(roi_maps)
        counts = [len(boxes) for boxes in selected_boxes]
        assert roi_vectors.shape == (sum(counts), REGION_DIM) and roi_vectors.dtype == torch.float32
        assert torch.isfinite(roi_vectors).all()
        grouped_vectors = roi_vectors.split(counts, dim=0)

    records = []
    for vectors, selection, original_size, transformed_size in zip(grouped_vectors, selected, original_sizes, transformed.image_sizes):
        count = len(vectors)
        original_height, original_width = original_size
        transformed_height, transformed_width = transformed_size
        scale = selection["boxes"].new_tensor([original_width / transformed_width, original_height / transformed_height,
                                              original_width / transformed_width, original_height / transformed_height])
        boxes = selection["boxes"] * scale  # Exactly one transformed -> original rescaling.
        boxes[:, 0::2].clamp_(min=0, max=original_width)
        boxes[:, 1::2].clamp_(min=0, max=original_height)
        normalized = boxes / boxes.new_tensor([original_width, original_height, original_width, original_height])
        record = {
            "features": torch.zeros(max_regions, REGION_DIM, dtype=torch.float32),
            "boxes": torch.zeros(max_regions, 4, dtype=torch.float32),
            "boxes_normalized": torch.zeros(max_regions, 4, dtype=torch.float32),
            "scores": torch.zeros(max_regions, dtype=torch.float32),
            "labels": torch.full((max_regions,), -1, dtype=torch.long),
            "mask": torch.zeros(max_regions, dtype=torch.bool),
            "num_regions": count, "fallback": selection["fallback"],
            "original_size": original_size, "transformed_size": tuple(transformed_size),
        }
        record["features"][:count] = vectors.cpu()
        record["boxes"][:count] = boxes.cpu()
        record["boxes_normalized"][:count] = normalized.cpu()
        record["scores"][:count] = selection["scores"].cpu()
        record["labels"][:count] = selection["labels"].cpu()
        record["mask"][:count] = True
        assert 1 <= count <= max_regions and int(record["mask"].sum()) == count
        assert torch.isfinite(record["features"].to(torch.float16)).all(), "ROI features overflow float16 storage."
        records.append(record)
    return records
