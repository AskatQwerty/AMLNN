"""Optional new-image inference; inline after the notebook's region helper definitions.

This fragment calls the already included build_frozen_region_detector,
extract_region_batch and region_feature_contract functions. Defining these
functions does not load weights, download anything or generate captions.
"""
import hashlib
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageOps


FUSION_RESNET50_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
FUSION_DETECTOR_SHA256 = "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384"


def _inference_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8 * 1024 * 1024),b""):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_fusion_encoders(device, project_dir=None, allow_download=False):
    """Load the two fixed encoders only when explicitly called."""
    device = torch.device(device)
    project_dir = Path.cwd() if project_dir is None else Path(project_dir)
    weights = ResNet50_Weights.IMAGENET1K_V2
    filename = weights.url.rsplit("/",1)[-1]
    candidates = [
        project_dir / "cache" / "models" / "checkpoints" / filename,
        Path(torch.hub.get_dir()) / "checkpoints" / filename,
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
    ]
    weight_path = next((path for path in candidates if path.is_file()),None)
    if weight_path is None:
        if not allow_download:
            raise FileNotFoundError(
                f"Cached {filename} is missing. Supply the project's cached weights; "
                "downloads are disabled unless allow_download=True is explicitly set."
            )
        weights.get_state_dict(progress=True,check_hash=True)
        weight_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if _inference_file_hash(weight_path) != FUSION_RESNET50_SHA256:
        raise ValueError("ResNet50 weights differ from the training feature cache.")

    backbone = resnet50(weights=None)
    backbone.load_state_dict(torch.load(weight_path,map_location="cpu",weights_only=True),strict=True)
    grid_encoder = nn.Sequential(*list(backbone.children())[:-2]).requires_grad_(False).to(device).eval()
    # Use the original pooling module on its feature map, before cache rounding.
    global_pool = backbone.avgpool.requires_grad_(False).to(device).eval()
    del backbone
    detector = build_frozen_region_detector(device,allow_download=allow_download)
    if detector.region_feature_metadata["weights_sha256"] != FUSION_DETECTOR_SHA256:
        raise ValueError("Faster R-CNN weights differ from the training region cache.")
    assert not any(parameter.requires_grad for parameter in grid_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in detector.parameters())
    actual_device = next(grid_encoder.parameters()).device
    metadata = {
        "resnet50_weights":"ResNet50_Weights.IMAGENET1K_V2",
        "resnet50_weights_sha256":FUSION_RESNET50_SHA256,
        "resnet50_image":"EXIF transpose, RGB, ImageOps.pad224x224 BILINEAR, centered",
        "padding_color":[124,116,104],
        "normalization":{"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        "resnet50_compute":"CUDA autocast float16" if actual_device.type == "cuda" else "CPU float32 fallback",
        "grid":"layer4 NCHW -> flatten spatial dimensions -> [B,49,2048]; float16 round then float32",
        "global":"AdaptiveAvgPool2d on the feature map inside the same autocast forward, before grid storage rounding; float32 result",
        "detector":dict(detector.region_feature_metadata),
        "regions":"Original RGB float32[0,1]; detector FP32; selected final-box ROI vectors float16 round then float32",
        "region_settings":region_feature_contract(36,0.2,0.5),
        "cnn_frozen":True,"device":str(actual_device),
        "precision_note":"The preprocessing and storage rounding match training. CPU fallback and different GPU batch/kernel paths can produce small numerical differences.",
    }
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return {"grid_encoder":grid_encoder,"global_pool":global_pool,"detector":detector,
            "resnet_transform":transform,"device":actual_device,"metadata":metadata}


@torch.no_grad()
def extract_fusion_image_features(image_path, encoders):
    """Produce the same three cached feature inputs for one original JPEG/image."""
    image_path = Path(image_path)
    device = torch.device(encoders["device"])
    for name in ["grid_encoder","detector"]:
        encoder = encoders[name]
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Both visual encoders must stay frozen and in eval mode.")
    with Image.open(image_path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        original_size = rgb.size
        original_tensor = transforms.ToTensor()(rgb)
        padded = ImageOps.pad(rgb,(224,224),method=Image.Resampling.BILINEAR,
                              color=(124,116,104),centering=(0.5,0.5))
        resnet_input = encoders["resnet_transform"](padded).unsqueeze(0).to(device)

    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type == "cuda"):
        feature_map = encoders["grid_encoder"](resnet_input)
        # Do not reconstruct this from an already rounded/float32-averaged grid.
        global_features = encoders["global_pool"](feature_map).flatten(1)
    if feature_map.shape != (1,2048,7,7) or global_features.shape != (1,2048):
        raise ValueError("Unexpected ResNet50 feature shape.")
    grid = feature_map.flatten(2).transpose(1,2).contiguous().to(torch.float16).float()
    global_features = global_features.float()
    # The helper disables autocast internally and re-pools the final selected boxes.
    record = extract_region_batch(encoders["detector"],[original_tensor],
                                  max_regions=36,score_threshold=0.2,nms_iou=0.5)[0]
    regions = record["features"].to(torch.float16).float().unsqueeze(0).to(device)
    mask = record["mask"].unsqueeze(0).to(device)
    boxes = record["boxes_normalized"].float().unsqueeze(0).to(device)
    assert torch.isfinite(grid).all() and torch.isfinite(global_features).all()
    assert torch.isfinite(regions).all() and torch.isfinite(boxes).all() and mask.any()
    count = int(record["num_regions"])
    features = {"grid":grid,"global_features":global_features,"regions":regions,
                "region_mask":mask,"region_boxes":boxes}
    audit = {"image_path":str(image_path.resolve()),"image_sha256":_inference_file_hash(image_path),
             "original_rgb_size_width_height":list(original_size),"valid_regions":count,
             "region_fallback":bool(record["fallback"]),
             "region_boxes_xyxy":record["boxes"][:count].tolist(),
             "region_boxes_normalized":record["boxes_normalized"][:count].tolist(),
             "region_scores":record["scores"][:count].tolist(),
             "region_labels":record["labels"][:count].tolist(),
             "preprocessing":encoders["metadata"]}
    return features,audit


@torch.no_grad()
def caption_new_image(image_path, caption_model, vocabulary, encoders, max_steps=33):
    """Greedy caption; load the chosen caption best_weights separately beforehand."""
    if not 1 <= max_steps <= 33:
        raise ValueError("Use the fixed maximum of33 generated steps, or a smaller display limit.")
    token_to_id = vocabulary["token_to_id"]
    id_to_token = vocabulary["id_to_token"]
    ids = {name:token_to_id[token] for name,token in [
        ("pad","<pad>"),("unk","<unk>"),("start","<start>"),("end","<end>")
    ]}
    if caption_model.pad_id != ids["pad"] or caption_model.output.out_features != len(id_to_token):
        raise ValueError("The vocabulary does not match this caption model.")
    device = next(caption_model.parameters()).device
    if device != torch.device(encoders["device"]):
        raise ValueError("Caption decoder and visual encoders must use the same device.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    features,audit = extract_fusion_image_features(image_path,encoders)
    was_training = caption_model.training
    caption_model.eval()
    try:
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type == "cuda"):
            tokens = caption_model.greedy(
                features["grid"],features["regions"],ids["start"],ids["end"],max_steps=max_steps,
                region_mask=features["region_mask"],region_boxes=features["region_boxes"],
                global_features=features["global_features"],
            )[0]
    finally:
        caption_model.train(was_training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert ids["pad"] not in tokens and ids["start"] not in tokens and ids["end"] not in tokens
    return {"caption":" ".join(id_to_token[index] for index in tokens),"token_ids":tokens,
            "generation":{"method":"greedy","max_steps":max_steps,"uses_human_references":False},
            "inference_seconds":time.perf_counter()-started,**audit}
