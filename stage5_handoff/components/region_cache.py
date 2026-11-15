from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import shutil
import time

import numpy as np
import torch
import torchvision

from region_features import (
    REGION_CACHE_VERSION, REGION_DIM, REGION_WEIGHT_NAME,
    region_feature_contract, load_region_image, extract_region_batch,
)


REGION_SOURCES = ("train2017", "val2017")


def _region_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def _region_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _region_file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _region_shapes(count, max_regions):
    return {
        "features": ((count, max_regions, REGION_DIM), np.dtype("float16")),
        "mask": ((count, max_regions), np.dtype("bool")),
        "num_regions": ((count,), np.dtype("uint8")),
        "boxes": ((count, max_regions, 4), np.dtype("float32")),
        "scores": ((count, max_regions), np.dtype("float32")),
        "labels": ((count, max_regions), np.dtype("int64")),
        "fallback": ((count,), np.dtype("bool")),
    }


def _region_file_bytes(shape, dtype):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": np.lib.format.dtype_to_descr(dtype),
                                               "fortran_order": False, "shape": shape})
    return len(header.getvalue()) + int(np.prod(shape)) * dtype.itemsize


def _region_paths(cache_dir, source, shapes):
    directory = Path(cache_dir)
    return {
        "final": {name: directory / f"{source}_{name}.npy" for name in shapes},
        "partial": {name: directory / f"{source}_{name}.partial.npy" for name in shapes},
        "ids": directory / f"{source}_image_ids.json",
        "progress": directory / f"{source}_progress.json",
        "meta": directory / f"{source}_meta.json",
    }


def _region_check_array(path, shape, dtype):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        assert array.shape == shape and array.dtype == dtype, f"Wrong region array shape/dtype: {path}"
        assert path.stat().st_size == array.offset + int(np.prod(shape)) * dtype.itemsize, f"Incomplete array: {path}"
    finally:
        array._mmap.close()


def _region_source_state(data, cache_dir, source, config):
    ids = data["source_image_ids"][source]
    shapes = _region_shapes(len(ids), config["max_regions"])
    paths = _region_paths(cache_dir, source, shapes)
    key = _region_hash(config)
    if paths["ids"].exists():
        assert json.loads(paths["ids"].read_text(encoding="utf-8")) == ids, "Region image row order differs."
    if paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if meta["cache_key"] != key or meta["config"] != config:
            raise ValueError("Incompatible completed region cache. Match its extraction settings or choose a new cache_dir.")
        assert meta["complete"]
        assert meta["source"] == source and meta["processed"] == len(ids) and paths["ids"].is_file()
        for name, (shape, dtype) in shapes.items():
            assert paths["final"][name].is_file() and not paths["partial"][name].exists()
            _region_check_array(paths["final"][name], shape, dtype)
            assert paths["final"][name].stat().st_size == meta["arrays"][name]["file_bytes"]
        return {"status": "complete", "processed": len(ids), "meta": meta, "paths": paths, "shapes": shapes}
    if paths["progress"].exists():
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if progress["config"] != config or progress["cache_key"] != key:
            raise ValueError("Incompatible region checkpoint. Resume requires the same code, weights, settings, extraction batch and device.")
        assert progress["source"] == source
        assert paths["ids"].is_file() and progress["total"] == len(ids)
        offset = progress["processed"]
        assert isinstance(offset, int) and not isinstance(offset, bool) and 0 <= offset <= len(ids)
        assert np.isfinite(progress["elapsed_seconds"]) and progress["elapsed_seconds"] >= 0
        status = progress["status"]
        assert status in ["initializing", "running", "paused", "ready"]
        if status == "initializing":
            assert offset == 0
        if status == "ready":
            assert offset == len(ids)
        for name, (shape, dtype) in shapes.items():
            final, partial = paths["final"][name], paths["partial"][name]
            assert not (final.exists() and partial.exists()), f"Ambiguous final/partial files: {name}"
            if final.exists():
                assert status == "ready", "A final array without complete metadata requires a ready checkpoint."
                _region_check_array(final, shape, dtype)
            elif partial.exists():
                _region_check_array(partial, shape, dtype)
            elif status != "initializing":
                raise FileNotFoundError(f"Region checkpoint has no matching {name} array.")
        return {"status": status, "processed": offset, "progress": progress, "paths": paths, "shapes": shapes}
    if any(path.exists() for group in [paths["final"], paths["partial"]] for path in group.values()):
        raise ValueError("Region arrays exist without a compatible progress/metadata file. Review them or choose a new cache directory.")
    return {"status": "new", "processed": 0, "paths": paths, "shapes": shapes}


def _region_cache_config(data, detector, max_regions, score_threshold, nms_iou, batch_size, producer_code_id):
    if not isinstance(max_regions, int) or isinstance(max_regions, bool) or not 1 <= max_regions <= 255:
        raise ValueError("max_regions must be in 1..255 for uint8 counts.")
    if not 0 <= score_threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Score threshold and NMS IoU must be in [0,1].")
    identity = getattr(detector, "region_feature_metadata", {})
    digest = identity.get("weights_sha256", "")
    assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    assert identity.get("weights") == REGION_WEIGHT_NAME and digest.startswith("258fb6c6")
    assert not detector.training and not any(parameter.requires_grad for parameter in detector.parameters())
    if not isinstance(producer_code_id, str) or len(producer_code_id) != 64 or any(character not in "0123456789abcdef" for character in producer_code_id):
        raise ValueError("Pass the protocol's 64-character producer_code_id fingerprint. Recompute it whenever the extraction source changes.")
    device = next(detector.parameters()).device
    device_info = {"device": str(device), "type": device.type,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                   "cuda_runtime": torch.version.cuda if device.type == "cuda" else None}
    assert data["feature_sources"] == {"train": "train2017", "val": "val2017", "test": "train2017"}
    for source in REGION_SOURCES:
        ids = data["source_image_ids"][source]
        assert ids and ids == sorted(set(ids))
    # Use the ACTUAL call settings, not the builder's default metadata settings.
    return {
        **region_feature_contract(max_regions, score_threshold, nms_iou),
        "source_id": data["provenance"]["source_id"],
        "annotation_sha256": {source: data["provenance"]["annotations_sha256"][source] for source in REGION_SOURCES},
        "source_image_ids_sha256": {source: _region_hash(data["source_image_ids"][source]) for source in REGION_SOURCES},
        "weights_sha256": digest,
        "producer_torch": torch.__version__, "producer_torchvision": torchvision.__version__,
        "producer_code_id": producer_code_id, "extraction_batch_size": batch_size, "extraction_device": device_info,
        "array_dtypes": {name: str(dtype) for name, (_, dtype) in _region_shapes(1, max_regions).items()},
    }


def _region_additional_space(states):
    additional = 0
    for state in states.values():
        if state["status"] in ["complete", "ready"]:
            continue
        for name, (shape, dtype) in state["shapes"].items():
            expected = _region_file_bytes(shape, dtype)
            partial = state["paths"]["partial"][name]
            if not partial.exists():
                additional += expected
                continue
            info = partial.stat()
            if hasattr(info, "st_blocks"):
                additional += max(0, expected - info.st_blocks * 512)
            elif os.name == "nt" and hasattr(info, "st_file_attributes") and not info.st_file_attributes & (0x200 | 0x800):
                additional += max(0, expected - info.st_size)
            else:
                remaining_rows = shape[0] - state["processed"]
                additional += remaining_rows * int(np.prod(shape[1:])) * dtype.itemsize
    return additional


def _region_image_records(data, source, raw_image_root):
    annotation_path = Path(data["data_root"]) / "annotations" / f"captions_{source}.json"
    payload = annotation_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == data["provenance"]["annotations_sha256"][source]
    annotation = json.loads(payload)
    images = {row["id"]: row for row in annotation["images"]}
    ids = data["source_image_ids"][source]
    assert sorted(images) == ids
    records = [images[image_id] for image_id in ids]
    directory = Path(raw_image_root) / source
    if not directory.is_dir():
        raise FileNotFoundError(f"Original photos missing: {directory}")
    with os.scandir(directory) as entries:
        available = {entry.name for entry in entries if entry.is_file()}
    for row in records:
        assert Path(row["file_name"]).name == row["file_name"]
    missing = [row["file_name"] for row in records if row["file_name"] not in available]
    if missing:
        raise FileNotFoundError(f"{source}: {len(missing)} photos missing; first {missing[:3]}")
    return records


def _region_pack_records(records, max_regions):
    assert records
    arrays = {}
    for name, (_, dtype) in _region_shapes(len(records), max_regions).items():
        values = [row[name] for row in records]
        if isinstance(values[0], torch.Tensor):
            values = torch.stack(values).detach().cpu().numpy()
        arrays[name] = np.asarray(values, dtype=dtype)
        assert arrays[name].shape == _region_shapes(len(records), max_regions)[name][0]
    _region_validate_chunk(arrays)
    for index, row in enumerate(records):
        count = row["num_regions"]
        height, width = row["original_size"]
        boxes = arrays["boxes"][index, :count]
        assert np.all(boxes[:, 0::2] <= width) and np.all(boxes[:, 1::2] <= height)
    return arrays


def _region_validate_chunk(arrays, score_threshold=None):
    features, masks, counts = arrays["features"], arrays["mask"], arrays["num_regions"]
    positions = np.arange(masks.shape[1])[None, :]
    expected_mask = positions < counts[:, None]
    assert np.all((counts >= 1) & (counts <= masks.shape[1]))
    assert np.array_equal(masks, expected_mask), "Mask/count or valid-region order mismatch."
    assert np.isfinite(features).all() and np.isfinite(arrays["boxes"]).all() and np.isfinite(arrays["scores"]).all()
    assert np.all(features[~masks] == 0) and np.all(arrays["boxes"][~masks] == 0)
    assert np.all(arrays["scores"][~masks] == 0) and np.all(arrays["labels"][~masks] == -1)
    boxes = arrays["boxes"][masks]
    assert np.all(boxes >= 0) and np.all(boxes[:, 2:] > boxes[:, :2])
    scores = arrays["scores"][masks]
    assert np.all((scores >= 0) & (scores <= 1))
    fallback = arrays["fallback"]
    assert np.all(counts[fallback] == 1)
    assert np.all(arrays["labels"][fallback, 0] == -1) and np.all(arrays["scores"][fallback, 0] == 0)
    normal_mask = masks & ~fallback[:, None]
    assert np.all(arrays["labels"][normal_mask] > 0)
    if score_threshold is not None:
        assert np.all(arrays["scores"][normal_mask] >= score_threshold)


def _region_validate_arrays(arrays, score_threshold, chunk_size=256):
    total = len(arrays["num_regions"])
    for start in range(0, total, chunk_size):
        _region_validate_chunk({name: array[start:start + chunk_size] for name, array in arrays.items()}, score_threshold)
    return {"all_finite": True, "counts_match_prefix_masks": True, "padding_valid": True,
            "images_checked": total, "valid_regions": int(arrays["num_regions"].sum(dtype=np.int64)),
            "count_min": int(arrays["num_regions"].min()), "count_mean": float(arrays["num_regions"].mean()),
            "count_max": int(arrays["num_regions"].max()),
            "fallback_images": int(arrays["fallback"].sum(dtype=np.int64)),
            "fallback_rate": float(arrays["fallback"].mean())}


def _region_finish_source(state, progress, config):
    paths = state["paths"]
    arrays = {name: np.load(paths["final"][name] if paths["final"][name].exists() else paths["partial"][name],
                            mmap_mode="r", allow_pickle=False) for name in state["shapes"]}
    validation_start = time.perf_counter()
    try:
        validation = _region_validate_arrays(arrays, config["score_threshold"])
    finally:
        for array in arrays.values():
            array._mmap.close()
    # A crash between renames is recoverable: the ready checkpoint permits a mix
    # of already-published final files and matching partial files, never both.
    for name in state["shapes"]:
        if paths["partial"][name].exists():
            paths["partial"][name].replace(paths["final"][name])
    array_metadata = {name: {"shape": list(shape), "dtype": str(dtype),
                             "file_bytes": paths["final"][name].stat().st_size,
                             "sha256": _region_file_sha256(paths["final"][name])}
                      for name, (shape, dtype) in state["shapes"].items()}
    elapsed = progress["elapsed_seconds"] + time.perf_counter() - validation_start
    meta = {**progress, "status": "complete", "complete": True, "elapsed_seconds": elapsed,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "validation": validation,
            "arrays": array_metadata}
    _region_json(paths["meta"], meta)
    return meta


def _extract_region_source(data, detector, cache_dir, source, records, raw_image_root, config, batch_size):
    state = _region_source_state(data, cache_dir, source, config)
    if state["status"] == "complete":
        return state["meta"]
    if state["status"] == "ready":
        return _region_finish_source(state, state["progress"], config)
    paths = state["paths"]
    ids = data["source_image_ids"][source]
    offset = state["processed"]
    previous = state.get("progress", {})
    previous_seconds = float(previous.get("elapsed_seconds", 0))
    device = next(detector.parameters()).device
    common = {
        "schema_version": 1, "source": source, "total": len(ids), "config": config, "cache_key": _region_hash(config),
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": batch_size, "resumed_from_row": offset,
        "timing": "Active allocation, image reading, detector/ROI computation, cache writes, full-array validation and SHA256 hashing; pauses and detector loading excluded.",
        "raw_image_validation": "EXIF-safe height/width pair checked against annotation dimensions; raw JPEG content bytes are not re-hashed.",
    }
    _region_json(paths["ids"], ids)
    if state["status"] == "new":
        _region_json(paths["progress"], {**common, "status": "initializing", "complete": False,
                                       "processed": 0, "elapsed_seconds": 0})
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    arrays = {}
    try:
        for name, (shape, dtype) in state["shapes"].items():
            path = paths["partial"][name]
            arrays[name] = np.load(path, mmap_mode="r+", allow_pickle=False) if path.exists() else np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape)

        def save_progress(status):
            for array in arrays.values():
                array.flush()
            progress = {**common, "status": status, "complete": False, "processed": offset,
                        "elapsed_seconds": previous_seconds + time.perf_counter() - start,
                        "updated_utc": datetime.now(timezone.utc).isoformat()}
            _region_json(paths["progress"], progress)
            return progress

        try:
            save_progress("running")
            last_save = time.perf_counter()
            for first in range(offset, len(ids), batch_size):
                last = min(first + batch_size, len(ids))
                images = []
                for row in range(first, last):
                    item = records[row]
                    image = load_region_image(Path(raw_image_root) / source / item["file_name"])
                    if sorted(image.shape[-2:]) != sorted([item["height"], item["width"]]):
                        raise ValueError(f"Image {item['id']} dimensions {tuple(image.shape[-2:])} differ from COCO metadata. Use full-size original photos, not resized previews.")
                    images.append(image)
                results = extract_region_batch(detector, images, max_regions=config["max_regions"],
                                               score_threshold=config["score_threshold"], nms_iou=config["class_agnostic_nms_iou"])
                assert len(results) == last - first
                batch = _region_pack_records(results, config["max_regions"])
                for name in arrays:
                    arrays[name][first:last] = batch[name]
                offset = last  # Advance only after every array has received the whole batch.
                if time.perf_counter() - last_save >= 20 or offset == len(ids):
                    progress = save_progress("running")
                    print(f"Regions {source}: {offset:,}/{len(ids):,}, active {progress['elapsed_seconds']/60:.2f} min", flush=True)
                    last_save = time.perf_counter()
            assert offset == len(ids)
            progress = save_progress("ready")
        except BaseException:
            save_progress("paused")
            raise
    finally:
        for array in arrays.values():
            array._mmap.close()
    return _region_finish_source(state, progress, config)


def load_region_cache(data, cache_dir, expected_cache_key=None, expected_contract=None, producer_code_id=None):
    """Read compatible completed metadata/headers without loading all vectors."""
    first_meta = Path(cache_dir) / "train2017_meta.json"
    if not first_meta.is_file():
        raise FileNotFoundError("Region cache is not complete. Run its explicit extraction step first.")
    config = json.loads(first_meta.read_text(encoding="utf-8"))["config"]
    expected_contract = region_feature_contract() if expected_contract is None else expected_contract
    if any(config.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Completed region cache does not match the requested selection/transform contract.")
    if producer_code_id is not None and config.get("producer_code_id") != producer_code_id:
        raise ValueError("Completed region cache was created by a different extractor code fingerprint.")
    assert config["version"] == REGION_CACHE_VERSION and config["weights"] == REGION_WEIGHT_NAME
    assert config["source_id"] == data["provenance"]["source_id"]
    assert config["annotation_sha256"] == {source: data["provenance"]["annotations_sha256"][source] for source in REGION_SOURCES}
    assert config["source_image_ids_sha256"] == {source: _region_hash(data["source_image_ids"][source]) for source in REGION_SOURCES}
    if expected_cache_key is not None:
        assert _region_hash(config) == expected_cache_key, "Unexpected region-feature contract."
    files = {}
    for source in REGION_SOURCES:
        state = _region_source_state(data, cache_dir, source, config)
        if state["status"] != "complete":
            raise FileNotFoundError(f"Region {source} cache is {state['status']}; resume extraction first.")
        files[source] = {**{f"{name}_path": path for name, path in state["paths"]["final"].items()},
                         "image_ids_path": state["paths"]["ids"], "image_ids": data["source_image_ids"][source],
                         "cache_key": _region_hash(config), "shape": state["meta"]["arrays"]["features"]["shape"],
                         "meta": state["meta"]}
    return files


def extract_region_cache(data, detector, cache_dir, raw_image_root, batch_size=1,
                         max_regions=36, score_threshold=0.2, nms_iou=0.5, resume=True, reserve_gib=2,
                         producer_code_id=None):
    """Explicit cache step only. This does not train a decoder or evaluate caption metrics."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer; begin with 1.")
    if not np.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be nonnegative.")
    cache_dir = Path(cache_dir)
    config = _region_cache_config(data, detector, max_regions, score_threshold, nms_iou, batch_size, producer_code_id)
    contract = region_feature_contract(max_regions, score_threshold, nms_iou)
    states = {source: _region_source_state(data, cache_dir, source, config) for source in REGION_SOURCES}
    pending = [source for source, state in states.items() if state["status"] != "complete"]
    if not resume and any(states[source]["status"] != "new" for source in pending):
        raise ValueError("A matching incomplete cache exists. Set resume=True; it will not be replaced.")
    if not pending:
        return load_region_cache(data, cache_dir, _region_hash(config), contract, producer_code_id)
    active = [source for source in pending if states[source]["status"] != "ready"]
    records = {source: _region_image_records(data, source, raw_image_root) for source in active}
    existing_parent = cache_dir
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    additional = _region_additional_space(states)
    required = additional + int(reserve_gib * 1024 ** 3) if active else 0
    free = shutil.disk_usage(existing_parent).free
    print(f"Region cache: {additional/1024**3:.2f} GiB additional allocation; {required/1024**3:.2f} GiB with reserve; {free/1024**3:.2f} GiB free.", flush=True)
    if free < required:
        raise OSError("Not enough free space for the region cache and reserve.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in pending:
        _extract_region_source(data, detector, cache_dir, source, records.get(source), raw_image_root, config, batch_size)
    files = load_region_cache(data, cache_dir, _region_hash(config), contract, producer_code_id)
    report = {"complete": True, "config": config, "cache_key": _region_hash(config),
              "total_images": sum(len(item["image_ids"]) for item in files.values()),
              "total_active_seconds": sum(item["meta"]["elapsed_seconds"] for item in files.values()),
              "sources": {source: item["meta"] for source, item in files.items()}}
    _region_json(cache_dir / "region_cache_report.json", report)
    return files
