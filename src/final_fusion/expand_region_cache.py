"""Optional migration of a COMPLETED ROI cache into a larger, new namespace.

This copies existing features only. It never executes the detector, trains a model,
changes the source cache, or weakens the frozen extractor's identity checks.
"""
import copy
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from region_cache import (
    load_fusion_region_cache, _fusion_specs, _fusion_paths, _fusion_hash,
    _fusion_file_sha, _fusion_json, _fusion_validate_chunk, _fusion_close,
)


def _expansion_clock():
    return datetime.now(timezone.utc).isoformat()


def _verify_completed_source(data, source_dir):
    files = load_fusion_region_cache(data, source_dir)
    records = {}
    for source, entry in files.items():
        metadata = entry["meta"]
        file_hashes = {}
        for name, description in metadata["arrays"].items():
            path = Path(entry[f"{name}_path"])
            actual = _fusion_file_sha(path)
            if actual != description["sha256"]:
                raise ValueError(f"Source cache checksum mismatch: {path}")
            file_hashes[name] = actual
        records[source] = {
            "image_ids": entry["image_ids"], "arrays_sha256": file_hashes,
            "metadata_sha256": _fusion_file_sha(Path(source_dir) / f"{source}_meta.json"),
            "image_ids_sha256": _fusion_file_sha(entry["image_ids_path"]),
            "historical_active_seconds": float(metadata["elapsed_seconds"]),
        }
    return files, records


def migrate_completed_fusion_cache(data, source_dir, destination_dir, expanded_train_order,
                                   minimum_train_images=None):
    """Copy a complete shorter cache; return the provenance report and new key.

    Afterwards resume the ORIGINAL extract_fusion_region_cache with the expanded
    train order/capacity and the same producer ID, batch and device. If supplied,
    minimum_train_images changes ONLY the new cache's required minimum; pass the
    same value as min_train_images when resuming the extractor. Set it to the full
    planned count to pause at a deadline instead of publishing a partial train.
    The deadline may differ because the frozen extractor excludes it from identity.
    """
    source_dir = Path(source_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    if source_dir == destination_dir or source_dir in destination_dir.parents or destination_dir in source_dir.parents:
        raise ValueError("Use separate sibling source/destination cache directories.")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    expanded_train_order = list(expanded_train_order)
    if not expanded_train_order or any(type(value) is not int for value in expanded_train_order):
        raise ValueError("Expanded train IDs must be nonempty integers.")
    if len(expanded_train_order) != len(set(expanded_train_order)):
        raise ValueError("Repeated expanded training IDs.")
    if minimum_train_images is not None:
        if type(minimum_train_images) is not int or not 1 <= minimum_train_images <= len(expanded_train_order):
            raise ValueError("minimum_train_images must be an integer from 1 to the expanded train count.")
    if not set(expanded_train_order) <= set(data["split_ids"]["train"]):
        raise ValueError("Expansion may only use the saved training split.")
    if set(expanded_train_order) & (set(data["split_ids"]["val"]) | set(data["split_ids"]["test"])):
        raise ValueError("Validation/test IDs may not enter the expanded training cache.")

    # Verify all old arrays before publishing anything in the destination.
    start = time.perf_counter()
    source_files, source_record = _verify_completed_source(data, source_dir)
    old_plan_path = source_dir / "cache_plan.json"
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    old_plan_sha = _fusion_file_sha(old_plan_path)
    previous_ids = source_files["train2017"]["image_ids"]
    if len(expanded_train_order) <= len(previous_ids):
        raise ValueError("The new cache must have a larger planned training capacity.")
    if expanded_train_order[:len(previous_ids)] != previous_ids:
        raise ValueError("Existing train rows are not the exact prefix of the expanded order.")
    if old_plan["planned_ids"]["train2017"] != expanded_train_order[:len(old_plan["planned_ids"]["train2017"])]:
        raise ValueError("The original predeclared order differs from the expanded order.")

    config = copy.deepcopy(old_plan["config"])
    config["planned_train_ids_sha256"] = _fusion_hash(expanded_train_order)
    if minimum_train_images is not None:
        config["minimum_train_images"] = minimum_train_images
    key = _fusion_hash(config)
    plan = {"config": config, "cache_key": key,
            "planned_ids": {"val2017": list(data["split_ids"]["val"]), "train2017": expanded_train_order}}
    migration_identity = {
        "source_dir": str(source_dir), "source_cache_key": old_plan["cache_key"],
        "source_plan_sha256": old_plan_sha, "source_arrays": source_record,
        "destination_dir": str(destination_dir), "destination_cache_key": key,
        "previous_train_rows": len(previous_ids), "planned_train_rows": len(expanded_train_order),
        "minimum_train_images": config["minimum_train_images"],
        "producer_code_id": config["producer_code_id"],
    }
    report_path = destination_dir / "migration.json"
    if destination_dir.exists():
        if not report_path.is_file():
            raise ValueError("Destination already exists without this migration record. Use a new namespace.")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["identity"] != migration_identity:
            raise ValueError("Destination belongs to a different migration.")
        if previous.get("complete"):
            # Do not overwrite progress if the resumed extractor has already appended rows.
            assert json.loads((destination_dir / "cache_plan.json").read_text(encoding="utf-8")) == plan
            return previous
        progress = destination_dir / "train2017_progress.json"
        if progress.exists():
            state = json.loads(progress.read_text(encoding="utf-8"))
            if state.get("stop_reason") != "migration_ready" or state.get("processed") != len(previous_ids):
                raise ValueError("Destination extraction has started; do not repeat the migration over live progress.")
    else:
        destination_dir.mkdir(parents=True, exist_ok=False)
    required = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in _fusion_specs(len(expanded_train_order)).values())
    required += sum(item["bytes"] for item in source_files["val2017"]["meta"]["arrays"].values())
    already_allocated = sum(path.stat().st_size for path in destination_dir.glob("*.npy"))
    if shutil.disk_usage(destination_dir).free < max(0, required - already_allocated) + 2 * 1024**3:
        raise OSError("Not enough free space for expanded partial arrays and copied validation plus reserve.")
    record = {"version": 1, "identity": migration_identity, "complete": False,
              "started_utc": _expansion_clock(), "detector_computation_performed": False,
              "source_preserved": True, "next_action": "Resume the unchanged frozen region extractor in this destination namespace."}
    _fusion_json(report_path, record)
    _fusion_json(destination_dir / "cache_plan.json", plan)

    # Validation is already complete. Copy actual files, not links to mutable sources.
    val_start = time.perf_counter()
    val_paths = _fusion_paths(destination_dir, "val2017")
    for name, target in val_paths["final"].items():
        shutil.copyfile(source_files["val2017"][f"{name}_path"], target)
        if _fusion_file_sha(target) != source_record["val2017"]["arrays_sha256"][name]:
            raise ValueError(f"Copied validation array failed checksum: {name}")
    _fusion_json(val_paths["ids"], source_files["val2017"]["image_ids"])
    val_migration_seconds = time.perf_counter() - val_start
    val_meta = copy.deepcopy(source_files["val2017"]["meta"])
    val_meta.update(config=config, cache_key=key, source="val2017", status="complete", complete=True,
                    materialized_by="verified_copy_from_completed_cache", materialized_utc=_expansion_clock(),
                    historical_active_seconds=source_record["val2017"]["historical_active_seconds"],
                    migration_copy_and_hash_seconds=val_migration_seconds,
                    elapsed_seconds=source_record["val2017"]["historical_active_seconds"] + val_migration_seconds,
                    migration_record="migration.json")
    _fusion_json(val_paths["meta"], val_meta)

    # Train final files are deliberately absent until the original extractor finishes.
    train_start = time.perf_counter()
    train_paths = _fusion_paths(destination_dir, "train2017")
    if any(path.exists() for path in train_paths["final"].values()) or train_paths["meta"].exists():
        raise ValueError("Destination contains published train results; refuse to overwrite them.")
    old_arrays, new_arrays = {}, {}
    try:
        for name, (shape, dtype) in _fusion_specs(len(expanded_train_order)).items():
            old_arrays[name] = np.load(source_files["train2017"][f"{name}_path"], mmap_mode="r", allow_pickle=False)
            new_arrays[name] = np.lib.format.open_memmap(train_paths["partial"][name], mode="w+", shape=shape, dtype=dtype)
        for first in range(0, len(previous_ids), 256):
            last = min(first + 256, len(previous_ids))
            values = {name: array[first:last] for name, array in old_arrays.items()}
            _fusion_validate_chunk(values)
            for name in new_arrays:
                new_arrays[name][first:last] = values[name]
        for array in new_arrays.values():
            array.flush()
        # Validate the exact committed prefix after copying, without touching the tail.
        for first in range(0, len(previous_ids), 256):
            last = min(first + 256, len(previous_ids))
            _fusion_validate_chunk({name: array[first:last] for name, array in new_arrays.items()})
            for name in new_arrays:
                assert np.array_equal(new_arrays[name][first:last], old_arrays[name][first:last])
    finally:
        _fusion_close(new_arrays)
        _fusion_close(old_arrays)
    train_migration_seconds = time.perf_counter() - train_start
    progress = {
        "source": "train2017", "cache_key": key, "config": config,
        "planned_count": len(expanded_train_order), "planned_ids_sha256": _fusion_hash(expanded_train_order),
        "status": "paused", "complete": False, "processed": len(previous_ids),
        "resume_from_row": len(previous_ids), "batch_size": config["batch_size"],
        "elapsed_seconds": source_record["train2017"]["historical_active_seconds"] + train_migration_seconds,
        "historical_active_seconds": source_record["train2017"]["historical_active_seconds"],
        "migration_copy_seconds": train_migration_seconds, "migration_record": "migration.json",
        "stop_reason": "migration_ready", "updated_utc": _expansion_clock(),
        "timing": "Previous cache active time plus migration copy; later extractor resumes cumulative time. See migration.json for separate historical and copy costs.",
    }
    _fusion_json(train_paths["progress"], progress)

    # A source prefix has not been recomputed. Recheck immutable source metadata/IDs;
    # array checksums were verified before and are never opened writable here.
    assert _fusion_file_sha(old_plan_path) == old_plan_sha
    for source in ["val2017", "train2017"]:
        assert _fusion_file_sha(source_dir / f"{source}_meta.json") == source_record[source]["metadata_sha256"]
        assert _fusion_file_sha(source_dir / f"{source}_image_ids.json") == source_record[source]["image_ids_sha256"]
    record.update(complete=True, finished_utc=_expansion_clock(), copied_train_rows=len(previous_ids),
                  copied_validation_rows=len(source_files["val2017"]["image_ids"]),
                  val_migration_seconds=val_migration_seconds, train_migration_seconds=train_migration_seconds,
                  migration_total_seconds=time.perf_counter() - start,
                  final_train_arrays_exist=False,
                  timing_note="This is a cache-copy operation, not new detector inference. Active extraction metadata carries historical work plus per-source migration copy costs; source verification overhead is recorded in migration_total_seconds.")
    _fusion_json(report_path, record)
    return record
