"""Read-only full integrity check of a COMPLETED fusion ROI cache (CPU only)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import ast
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.resolve()
sys.path.append(str(BASE.parent / "stage5_handoff"))
from prepared_data import load_region_prepared_data
from region_cache import (
    load_fusion_region_cache, _fusion_file_sha, _fusion_hash, _fusion_json,
    _fusion_validate_chunk, _fusion_close, _fusion_specs,
)

PROJECT = Path(r"C:\Users\askat\Study\3.1\AML Applied Machine Learning\Final")
RAW = Path(r"C:\Users\askat\Downloads\MSCOCO\coco2017")


def frozen_producer_identity():
    paths = [BASE.parent / "stage5_handoff/region_features.py", BASE / "region_cache.py"]
    return hashlib.sha256("\n".join(ast.dump(ast.parse(path.read_text(encoding="utf-8")))
                                  for path in paths).encode()).hexdigest()


def verify_complete_cache(data, cache_dir, report_path, producer_code_id=None,
                          expected_validation_images=5000, chunk_size=256):
    """Check final .npy arrays only; publish passed=True after every check succeeds."""
    started = time.perf_counter()
    directory = Path(cache_dir).resolve()
    report_path = Path(report_path).resolve()
    if directory == report_path or directory in report_path.parents:
        raise ValueError("Put the audit report outside the immutable cache directory.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    producer_code_id = producer_code_id or frozen_producer_identity()
    report = {"passed": False, "status": "running", "cache_dir": str(directory),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "producer_code_id": producer_code_id, "cpu_only": True, "gpu_used": False,
              "final_arrays_only": True, "cache_modified": False,
              "verifier_sha256": _fusion_file_sha(Path(__file__)),
              "region_cache_source_sha256": _fusion_file_sha(BASE / "region_cache.py")}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _fusion_json(report_path, report)
    try:
        split_sets = {name: set(ids) for name, ids in data["split_ids"].items()}
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert split_sets[left].isdisjoint(split_sets[right]), f"Prepared split overlap: {left}/{right}"
        assert len(data["split_ids"]["val"]) == expected_validation_images
        files = load_fusion_region_cache(data, directory, producer_code_id=producer_code_id)
        plan_path, cache_report_path = directory / "cache_plan.json", directory / "cache_report.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
        assert cache_report["complete"] is True and cache_report["cache_key"] == plan["cache_key"]
        assert cache_report["config"] == plan["config"]
        selected_path = directory / "selected_train_ids.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        assert selected == sorted(files["train2017"]["image_ids"])
        assert len(selected) == len(set(selected)) and set(selected) <= split_sets["train"]
        assert set(selected).isdisjoint(split_sets["val"] | split_sets["test"])
        assert files["val2017"]["image_ids"] == data["split_ids"]["val"]
        assert cache_report["train_images"] == len(selected)
        assert cache_report["validation_images"] == expected_validation_images
        assert cache_report["train_is_partial"] == (len(selected) < len(data["split_ids"]["train"]))
        assert cache_report["no_test_images_extracted"] is True
        source_results = {}
        for source, entry in files.items():
            metadata = entry["meta"]
            assert metadata["config"] == plan["config"]
            assert metadata["status"] == "complete" and metadata["complete"] is True
            assert metadata["shape"] == [len(entry["image_ids"]), 36, 1024]
            assert metadata["processed"] == len(entry["image_ids"])
            planned_ids = plan["planned_ids"][source]
            assert metadata["planned_count"] == len(planned_ids)
            assert metadata["planned_ids_sha256"] == _fusion_hash(planned_ids)
            arrays, hashes = {}, {}
            count_min, count_max, count_sum, fallbacks, rows = 36, 0, 0, 0, 0
            feature_min, feature_max = float("inf"), float("-inf")
            zero_valid_vectors = 0
            try:
                for name in _fusion_specs(1):
                    path = Path(entry[f"{name}_path"])
                    assert ".partial." not in path.name, "Never audit unpublished capacity arrays."
                    hashes[name] = _fusion_file_sha(path)
                    assert hashes[name] == metadata["arrays"][name]["sha256"], f"Checksum mismatch: {path}"
                    arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                for first in range(0, len(entry["image_ids"]), chunk_size):
                    last = min(first + chunk_size, len(entry["image_ids"]))
                    chunk = {name: array[first:last] for name, array in arrays.items()}
                    _fusion_validate_chunk(chunk)
                    count = chunk["num_regions"]
                    count_min, count_max = min(count_min, int(count.min())), max(count_max, int(count.max()))
                    count_sum += int(count.sum(dtype=np.int64))
                    fallbacks += int(chunk["fallback"].sum(dtype=np.int64))
                    rows += last - first
                    features = chunk["features"][chunk["mask"]]
                    feature_min, feature_max = min(feature_min, float(features.min())), max(feature_max, float(features.max()))
                    zero_valid_vectors += int(np.all(features == 0, axis=1).sum())
            finally:
                _fusion_close(arrays)
            computed = {"images": rows, "regions_min": count_min, "regions_max": count_max,
                        "regions_mean": count_sum / rows, "valid_regions": count_sum,
                        "fallback_images": fallbacks, "fallback_rate": fallbacks / rows,
                        "all_rows_finite_and_masks_valid": True}
            for name, value in computed.items():
                recorded = metadata["statistics"][name]
                assert np.isclose(recorded, value, rtol=1e-12, atol=1e-12), f"Wrong cached statistic: {source}/{name}"
            source_results[source] = {
                **computed, "shape": [rows, 36, 1024], "features_min": feature_min,
                "features_max": feature_max, "zero_valid_feature_vectors": zero_valid_vectors,
                "all_array_sha256_verified": True, "arrays_sha256": hashes,
                "image_ids_sha256": _fusion_file_sha(entry["image_ids_path"]),
                "metadata_sha256": _fusion_file_sha(directory / f"{source}_meta.json"),
            }
            print(f"Verified {source}: {rows:,} images, {count_sum:,} regions, {fallbacks:,} fallbacks; all 8 hashes and every row checked.", flush=True)
        report.update(passed=True, status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started, cache_key=plan["cache_key"],
                      source_id=data["provenance"]["source_id"], split_id=data["provenance"]["split_id"],
                      cache_plan_sha256=_fusion_file_sha(plan_path), cache_report_sha256=_fusion_file_sha(cache_report_path),
                      selected_train_ids_sha256=_fusion_file_sha(selected_path),
                      train_images=len(selected), validation_images=expected_validation_images,
                      total_images=sum(item["images"] for item in source_results.values()),
                      total_regions=sum(item["valid_regions"] for item in source_results.values()),
                      train_is_subset=len(selected) < len(data["split_ids"]["train"]),
                      all_prepared_splits_disjoint=True, no_test_images=True,
                      all_array_sha256_verified=True, all_rows_validated=True, sources=source_results)
        _fusion_json(report_path, report)
        return report
    except BaseException as error:
        report.update(passed=False, status="failed", elapsed_seconds=time.perf_counter() - started,
                      error_type=type(error).__name__, error=str(error))
        _fusion_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Output JSON outside the cache; passed=true only after full verification")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--producer-code-id", help="Expected producer fingerprint; default recomputes the frozen source AST identity")
    args = parser.parse_args()
    data = load_region_prepared_data(args.project, args.raw)
    report = verify_complete_cache(data, args.cache_dir, args.report, args.producer_code_id)
    print(json.dumps({name: report[name] for name in ["passed", "cache_key", "train_images", "validation_images", "total_regions", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
