"""Frozen detector cache: complete validation first, then a timeboxed train prefix."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch
import torchvision

from region_features import region_feature_contract, load_region_image, extract_region_batch


FUSION_REGION_VERSION = "fasterrcnn_fusion_partial_v1"


def _fusion_utc():
    return datetime.now(timezone.utc)


def _fusion_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def _fusion_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fusion_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fusion_specs(count):
    return {
        "features": ((count, 36, 1024), np.dtype("float16")),
        "mask": ((count, 36), np.dtype("bool")),
        "normalized_boxes": ((count, 36, 4), np.dtype("float32")),
        "boxes": ((count, 36, 4), np.dtype("float32")),
        "scores": ((count, 36), np.dtype("float32")),
        "labels": ((count, 36), np.dtype("int64")),
        "num_regions": ((count,), np.dtype("uint8")),
        "fallback": ((count,), np.dtype("bool")),
    }


def _fusion_paths(directory, source):
    directory = Path(directory)
    return {
        "partial": {name: directory / f"{source}_{name}.partial.npy" for name in _fusion_specs(1)},
        "final": {name: directory / f"{source}_{name}.npy" for name in _fusion_specs(1)},
        "ids": directory / f"{source}_image_ids.json",
        "progress": directory / f"{source}_progress.json",
        "meta": directory / f"{source}_meta.json",
    }


def _fusion_close(arrays):
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _fusion_validate_chunk(arrays):
    count = arrays["num_regions"]
    mask = arrays["mask"]
    assert np.all((count >= 1) & (count <= 36))
    assert np.array_equal(mask, np.arange(36)[None, :] < count[:, None])
    for name in ["features", "boxes", "normalized_boxes", "scores"]:
        assert np.isfinite(arrays[name]).all(), f"Nonfinite {name}."
        assert np.all(arrays[name][~mask] == 0), f"Invalid {name} padding."
    assert np.all(arrays["labels"][~mask] == -1)
    normalized = arrays["normalized_boxes"][mask]
    boxes = arrays["boxes"][mask]
    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    assert np.all(normalized[:, 2:] > normalized[:, :2])
    assert np.all(boxes >= 0) and np.all(boxes[:, 2:] > boxes[:, :2])
    assert np.all((arrays["scores"][mask] >= 0) & (arrays["scores"][mask] <= 1))
    fallback = arrays["fallback"]
    assert np.all(count[fallback] == 1)
    assert np.all(arrays["labels"][fallback, 0] == -1)
    assert np.all(arrays["scores"][fallback, 0] == 0)
    normal = mask & ~fallback[:, None]
    assert np.all(arrays["labels"][normal] > 0)
    assert np.all(arrays["scores"][normal] >= 0.2)


def _fusion_pack(records):
    arrays = {}
    for name, (shape, dtype) in _fusion_specs(len(records)).items():
        key = "boxes_normalized" if name == "normalized_boxes" else name
        values = [row[key] for row in records]
        if isinstance(values[0], torch.Tensor):
            values = torch.stack(values).detach().cpu().numpy()
        arrays[name] = np.asarray(values, dtype=dtype)
        assert arrays[name].shape == shape
    _fusion_validate_chunk(arrays)
    return arrays


def _fusion_annotation_records(data, source, ordered_ids, raw_root):
    path = Path(data["data_root"]) / "annotations" / f"captions_{source}.json"
    assert _fusion_file_sha(path) == data["provenance"]["annotations_sha256"][source]
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in payload["images"]}
    assert set(ordered_ids) <= set(images)
    directory = Path(raw_root) / source
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing original JPEG folder: {directory}")
    with os.scandir(directory) as entries:
        available = {entry.name for entry in entries if entry.is_file()}
    rows = [images[image_id] for image_id in ordered_ids]
    missing = []
    for row in rows:
        assert Path(row["file_name"]).name == row["file_name"]
        if row["file_name"] not in available:
            missing.append(row["file_name"])
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} original {source} images, e.g. {missing[:3]}")
    return rows


def _fusion_finish_source(directory, source, ordered_ids, progress, config):
    paths = _fusion_paths(directory, source)
    count = progress["processed"]
    assert count > 0 and count <= len(ordered_ids)
    selected_ids = ordered_ids[:count]
    start = time.perf_counter()
    source_arrays, final_arrays = {}, {}
    try:
        for name, (shape, dtype) in _fusion_specs(len(ordered_ids)).items():
            source_arrays[name] = np.load(paths["partial"][name], mmap_mode="r", allow_pickle=False)
            assert source_arrays[name].shape == shape and source_arrays[name].dtype == dtype
        for name, (shape, dtype) in _fusion_specs(count).items():
            # Repeating a failed compaction is safe: the complete metadata is written last.
            final_arrays[name] = np.lib.format.open_memmap(paths["final"][name], mode="w+", dtype=dtype, shape=shape)
        for first in range(0, count, 256):
            last = min(first + 256, count)
            chunk = {name: values[first:last] for name, values in source_arrays.items()}
            _fusion_validate_chunk(chunk)
            for name, values in final_arrays.items():
                values[first:last] = chunk[name]
        for values in final_arrays.values():
            values.flush()
        counts = np.asarray(source_arrays["num_regions"][:count])
        statistics = {
            "images": count, "regions_min": int(counts.min()), "regions_mean": float(counts.mean()),
            "regions_max": int(counts.max()), "valid_regions": int(counts.sum(dtype=np.int64)),
            "fallback_images": int(source_arrays["fallback"][:count].sum(dtype=np.int64)),
            "fallback_rate": float(source_arrays["fallback"][:count].mean()),
            "all_rows_finite_and_masks_valid": True,
        }
    finally:
        _fusion_close(final_arrays)
        _fusion_close(source_arrays)
    hashes = {name: _fusion_file_sha(path) for name, path in paths["final"].items()}
    _fusion_json(paths["ids"], selected_ids)
    compact_seconds = time.perf_counter() - start
    metadata = {
        **progress, "status": "complete", "complete": True, "config": config,
        "cache_key": _fusion_hash(config), "source": source,
        "completed_utc": _fusion_utc().isoformat(), "shape": [count, 36, 1024],
        "actual_ids_sha256": _fusion_hash(selected_ids), "statistics": statistics,
        "compaction_and_hash_seconds": compact_seconds,
        "elapsed_seconds": progress["elapsed_seconds"] + compact_seconds,
        "arrays": {name: {"shape": list(shape), "dtype": str(dtype), "sha256": hashes[name],
                          "bytes": paths["final"][name].stat().st_size}
                   for name, (shape, dtype) in _fusion_specs(count).items()},
        "temporary_files": "Capacity-sized .partial.npy files retained for recovery; final exact-size .npy files are used for training.",
    }
    _fusion_json(paths["meta"], metadata)
    return metadata


def _fusion_extract_source(data, detector, directory, raw_root, source, ordered_ids, deadline, batch_size, min_train_images, config):
    paths = _fusion_paths(directory, source)
    key = _fusion_hash(config)
    if paths["meta"].exists():
        metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
        assert metadata["complete"] and metadata["cache_key"] == key
        return metadata
    prior = {}
    if paths["progress"].exists():
        prior = json.loads(paths["progress"].read_text(encoding="utf-8"))
        assert prior["cache_key"] == key and prior["source"] == source
        assert prior["planned_ids_sha256"] == _fusion_hash(ordered_ids)
    else:
        if any(path.exists() for group in [paths["partial"], paths["final"]] for path in group.values()):
            raise ValueError("Arrays exist without matching progress metadata; use a new cache directory.")
    offset = int(prior.get("processed", 0))
    assert 0 <= offset <= len(ordered_ids)
    if prior.get("status") == "ready":
        return _fusion_finish_source(directory, source, ordered_ids, prior, config)
    records = _fusion_annotation_records(data, source, ordered_ids, raw_root)
    device = next(detector.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    previous_seconds = float(prior.get("elapsed_seconds", 0))
    common = {"source": source, "cache_key": key, "planned_count": len(ordered_ids),
              "planned_ids_sha256": _fusion_hash(ordered_ids), "config": config,
              "batch_size": batch_size, "resume_from_row": offset,
              "timing": "Active allocation, JPEG reads, detector computation, writes, final validation, compaction and hashes; pauses excluded.",
              "requested_deadline_utc": deadline.isoformat()}
    arrays = {}
    _fusion_json(paths["progress"], {**common, "status": "initializing", "processed": offset,
                                    "elapsed_seconds": previous_seconds, "complete": False})
    try:
        for name, (shape, dtype) in _fusion_specs(len(ordered_ids)).items():
            path = paths["partial"][name]
            if path.exists():
                arrays[name] = np.load(path, mmap_mode="r+", allow_pickle=False)
                assert arrays[name].shape == shape and arrays[name].dtype == dtype
            else:
                assert offset == 0, f"Missing a committed partial array: {name}"
                arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

        def checkpoint(status, stop_reason=None):
            for array in arrays.values():
                array.flush()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            progress = {**common, "status": status, "complete": False, "processed": offset,
                        "elapsed_seconds": previous_seconds + time.perf_counter() - start,
                        "updated_utc": _fusion_utc().isoformat(), "stop_reason": stop_reason}
            _fusion_json(paths["progress"], progress)
            return progress

        try:
            last_print = time.perf_counter()
            reason = "planned_source_complete"
            for first in range(offset, len(ordered_ids), batch_size):
                if _fusion_utc() >= deadline:
                    if source == "val2017" or offset < min_train_images:
                        checkpoint("paused", "deadline_before_required_minimum")
                        raise TimeoutError(f"Deadline reached: {source} has {offset}/{len(ordered_ids)} images. Extend the extraction deadline explicitly to finish required validation/minimum train.")
                    reason = "deadline_train_prefix"
                    break
                last = min(first + batch_size, len(ordered_ids))
                images = []
                for row in records[first:last]:
                    image = load_region_image(Path(raw_root) / source / row["file_name"])
                    assert sorted(image.shape[-2:]) == sorted([row["height"], row["width"]]), "Use original JPEG dimensions, not resized previews."
                    images.append(image)
                output = extract_region_batch(detector, images, max_regions=36, score_threshold=0.2, nms_iou=0.5)
                assert len(output) == last - first
                chunk = _fusion_pack(output)
                for name, array in arrays.items():
                    array[first:last] = chunk[name]
                offset = last
                if time.perf_counter() - last_print >= 20 or offset == len(ordered_ids):
                    progress = checkpoint("running")
                    speed = (offset - common["resume_from_row"]) / max(time.perf_counter() - start, 1e-9)
                    remaining = (len(ordered_ids) - offset) / max(speed, 1e-9)
                    print(f"ROI {source}: {offset:,}/{len(ordered_ids):,} | {speed:.2f} img/s | active {progress['elapsed_seconds']/60:.1f} min | full-source ETA {remaining/60:.1f} min", flush=True)
                    last_print = time.perf_counter()
            assert offset > 0
            if source == "val2017":
                assert offset == len(ordered_ids), "Validation must be complete."
            else:
                assert offset >= min(min_train_images, len(ordered_ids))
            progress = checkpoint("ready", reason)
        except BaseException:
            checkpoint("paused", "interrupted_or_failed")
            raise
    finally:
        _fusion_close(arrays)
    return _fusion_finish_source(directory, source, ordered_ids, progress, config)


def load_fusion_region_cache(data, cache_dir, expected_cache_key=None, producer_code_id=None):
    directory = Path(cache_dir)
    plan_path = directory / "cache_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError("Missing fusion-region cache plan.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = plan["config"]
    key = _fusion_hash(config)
    assert key == plan["cache_key"]
    assert config["source_id"] == data["provenance"]["source_id"]
    assert config["split_id"] == data["provenance"]["split_id"]
    assert config["annotations_sha256"] == data["provenance"]["annotations_sha256"]
    if expected_cache_key is not None:
        assert key == expected_cache_key
    if producer_code_id is not None:
        assert config["producer_code_id"] == producer_code_id
    for name, value in region_feature_contract().items():
        assert config["detector_contract"][name] == value
    files = {}
    for source in ["val2017", "train2017"]:
        paths = _fusion_paths(directory, source)
        if not paths["meta"].is_file():
            raise FileNotFoundError(f"Incomplete fusion cache: {source}")
        metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
        assert metadata["complete"] and metadata["cache_key"] == key
        ids = json.loads(paths["ids"].read_text(encoding="utf-8"))
        assert ids and len(ids) == len(set(ids)) and _fusion_hash(ids) == metadata["actual_ids_sha256"]
        planned = plan["planned_ids"][source]
        assert ids == planned[:len(ids)]
        if source == "val2017":
            assert ids == data["split_ids"]["val"]
        else:
            assert set(ids) <= set(data["split_ids"]["train"])
            assert not set(ids) & set(data["split_ids"]["test"])
        for name, (shape, dtype) in _fusion_specs(len(ids)).items():
            array = np.load(paths["final"][name], mmap_mode="r", allow_pickle=False)
            try:
                assert array.shape == shape and array.dtype == dtype
                assert paths["final"][name].stat().st_size == metadata["arrays"][name]["bytes"]
            finally:
                array._mmap.close()
        files[source] = {**{f"{name}_path": str(path) for name, path in paths["final"].items()},
                         "image_ids_path": str(paths["ids"]), "image_ids": ids,
                         "boxes_normalized": True, "shape": [len(ids), 36, 1024],
                         "cache_key": key, "meta": metadata}
    return files


def extract_fusion_region_cache(data, detector, cache_dir, raw_root, train_order, deadline_utc,
                               batch_size=1, max_train_images=None, min_train_images=2048,
                               producer_code_id=None):
    """Run validation first, then finalize a completed random train prefix at the deadline."""
    assert isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0
    assert isinstance(min_train_images, int) and min_train_images > 0
    train_order = [int(value) for value in train_order]
    assert len(train_order) == len(set(train_order)) and train_order
    assert set(train_order) <= set(data["split_ids"]["train"])
    assert not set(train_order) & (set(data["split_ids"]["val"]) | set(data["split_ids"]["test"]))
    limit = len(train_order) if max_train_images is None else int(max_train_images)
    assert 0 < limit <= len(train_order)
    train_order = train_order[:limit]
    if isinstance(deadline_utc, str):
        deadline_utc = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    assert deadline_utc.tzinfo is not None
    deadline = deadline_utc.astimezone(timezone.utc)
    assert not detector.training and not any(parameter.requires_grad for parameter in detector.parameters())
    identity = detector.region_feature_metadata
    weight_hash = identity["weights_sha256"]
    assert len(weight_hash) == 64 and weight_hash.startswith("258fb6c6")
    if producer_code_id is None:
        producer_code_id = FUSION_REGION_VERSION
    device = next(detector.parameters()).device
    config = {
        "version": FUSION_REGION_VERSION, "producer_code_id": producer_code_id,
        "source_id": data["provenance"]["source_id"], "split_id": data["provenance"]["split_id"],
        "annotations_sha256": data["provenance"]["annotations_sha256"],
        "weights_sha256": weight_hash, "detector_contract": region_feature_contract(),
        "planned_train_ids_sha256": _fusion_hash(train_order),
        "planned_val_ids_sha256": _fusion_hash(data["split_ids"]["val"]),
        "batch_size": batch_size, "minimum_train_images": min_train_images,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "selection": "Validation complete first; training keeps a completed prefix of the supplied predeclared random order. No content-based selection.",
    }
    key = _fusion_hash(config)
    planned = {"val2017": list(data["split_ids"]["val"]), "train2017": train_order}
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "cache_plan.json"
    plan = {"config": config, "cache_key": key, "planned_ids": planned}
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding="utf-8")) == plan, "Incompatible cached code/device/batch/weights/ID order; use the matching configuration or a new directory."
    else:
        _fusion_json(plan_path, plan)
    # Keep capacity arrays plus exact-size final arrays; reserve enough for both.
    required = 2 * sum(sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in _fusion_specs(len(ids)).values()) for ids in planned.values())
    existing = sum(path.stat().st_size for path in directory.glob("*.npy"))
    needed = max(0, required - existing) + 2 * 1024**3
    if shutil.disk_usage(directory).free < needed:
        raise OSError(f"Need approximately {needed/1024**3:.2f} GiB additional free space, including recovery arrays and reserve.")
    print(f"Fusion ROI plan: validation={len(planned['val2017']):,}, maximum train={len(train_order):,}, deadline UTC={deadline.isoformat()}, batch={batch_size}", flush=True)
    for source in ["val2017", "train2017"]:
        _fusion_extract_source(data, detector, directory, raw_root, source, planned[source], deadline,
                               batch_size, min_train_images, config)
    files = load_fusion_region_cache(data, directory, key, producer_code_id)
    selected = sorted(files["train2017"]["image_ids"])
    _fusion_json(directory / "selected_train_ids.json", selected)
    _fusion_json(directory / "cache_report.json", {
        "complete": True, "cache_key": key, "config": config,
        "train_images": len(selected), "validation_images": len(files["val2017"]["image_ids"]),
        "train_is_partial": len(selected) < len(data["split_ids"]["train"]),
        "selected_train_ids_path": str(directory / "selected_train_ids.json"),
        "total_active_seconds": sum(item["meta"]["elapsed_seconds"] for item in files.values()),
        "sources": {source: item["meta"] for source, item in files.items()},
        "no_test_images_extracted": True,
    })
    return files
