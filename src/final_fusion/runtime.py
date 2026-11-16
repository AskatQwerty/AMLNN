"""Cached fusion data and one resumable, time-bounded caption training run."""
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


def fusion_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fusion_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def fusion_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_fusion_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def fusion_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fusion_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def fusion_deadline(value):
    if value is None:
        return float("inf")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadlines must include an explicit UTC offset.")
    return parsed.timestamp()


def _fusion_ids(entry, prefix=""):
    if prefix + "image_ids" in entry:
        values = entry[prefix + "image_ids"]
    else:
        values = json.loads(Path(entry[prefix + "image_ids_path"]).read_text(encoding="utf-8"))
    if not values or any(type(value) is not int for value in values) or len(values) != len(set(values)):
        raise ValueError("Feature row IDs must be nonempty unique integers.")
    return values


class FusionCaptionDataset(Dataset):
    """The selected train IDs or complete validation, reading the two frozen caches."""
    def __init__(self, data, grid_files, region_files, split, image_ids=None, seed=42):
        if split not in ["train", "val"]:
            raise ValueError("Fusion training only uses train and validation; test is excluded.")
        self.data, self.split, self.seed = data, split, seed
        self.image_ids = sorted(data["split_ids"][split] if image_ids is None else image_ids)
        assert self.image_ids and len(self.image_ids) == len(set(self.image_ids))
        assert set(self.image_ids) <= set(data["split_ids"][split])
        self.arrays = {}
        source = data["feature_sources"][split]
        grid_entry, region_entry = grid_files[source], region_files[source]
        grid_ids, region_ids = _fusion_ids(grid_entry), _fusion_ids(region_entry)
        self.grid_rows = {image_id: row for row, image_id in enumerate(grid_ids)}
        self.region_rows = {image_id: row for row, image_id in enumerate(region_ids)}
        assert all(image_id in self.grid_rows and image_id in self.region_rows for image_id in self.image_ids)
        grid_path = grid_entry.get("path", grid_entry.get("features_path"))
        assert grid_path is not None
        boxes_path = region_entry.get("normalized_boxes_path", region_entry.get("boxes_normalized_path", region_entry.get("boxes_path")))
        assert boxes_path is not None, "Region boxes are required for position embeddings."
        self.boxes_normalized = bool(region_entry.get("boxes_normalized", False) or
                                     "normalized_boxes_path" in region_entry or "boxes_normalized_path" in region_entry)
        self.sizes = {}
        if not self.boxes_normalized:
            annotation_path = Path(data["data_root"]) / "annotations" / f"captions_{source}.json"
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            self.sizes = {row["id"]: (row["width"], row["height"]) for row in annotation["images"]}
            del annotation
        paths = {"grid": grid_path, "regions": region_entry["features_path"],
                 "region_mask": region_entry["mask_path"], "region_boxes": boxes_path}
        self.global_rows = None
        if "global_path" in grid_entry:
            paths["global_features"] = grid_entry["global_path"]
            global_ids = _fusion_ids(grid_entry, "global_") if ("global_image_ids" in grid_entry or "global_image_ids_path" in grid_entry) else grid_ids
            self.global_rows = {image_id: row for row, image_id in enumerate(global_ids)}
            assert all(image_id in self.global_rows for image_id in self.image_ids)
        try:
            for name, path in paths.items():
                self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            assert self.arrays["grid"].shape == (len(grid_ids), 49, 2048)
            assert self.arrays["regions"].shape == (len(region_ids), 36, 1024)
            assert self.arrays["region_mask"].shape == (len(region_ids), 36)
            assert self.arrays["region_mask"].dtype == np.bool_
            assert self.arrays["region_boxes"].shape == (len(region_ids), 36, 4)
            if self.global_rows is not None:
                assert self.arrays["global_features"].shape == (len(self.global_rows), 2048)
            selected_rows = [self.region_rows[image_id] for image_id in self.image_ids]
            assert self.arrays["region_mask"][selected_rows].any(axis=1).all(), "Every selected image needs a valid region."
            self.set_epoch(0)
        except BaseException:
            self.close()
            raise
        self.provenance = {"source": source, "image_ids_sha256": fusion_hash(self.image_ids),
            "grid_ids_sha256": fusion_hash(grid_ids), "region_ids_sha256": fusion_hash(region_ids),
            "grid_cache_key": grid_entry.get("cache_key"), "region_cache_key": region_entry.get("cache_key"),
            "paths": {name: str(Path(path).resolve()) for name, path in paths.items()},
            "shapes": {name: list(array.shape) for name, array in self.arrays.items()},
            "boxes_normalized": self.boxes_normalized}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        self.caption_choices = [rng.randrange(len(self.data["tokens"][self.split][image_id]))
                                if self.split == "train" else 0 for image_id in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        grid_row, region_row = self.grid_rows[image_id], self.region_rows[image_id]
        tokens = self.data["tokens"][self.split][image_id][self.caption_choices[index]]
        special = self.data["ids"]
        caption = [special["start"]] + [self.data["token_to_id"].get(word, special["unk"])
                   for word in tokens[:self.data["max_content_tokens"]]] + [special["end"]]
        boxes = self.arrays["region_boxes"][region_row].copy().astype(np.float32)
        if not self.boxes_normalized:
            width, height = self.sizes[image_id]
            boxes /= np.asarray([width, height, width, height], dtype=np.float32)
        item = {"grid": torch.from_numpy(self.arrays["grid"][grid_row].copy()).float(),
                "regions": torch.from_numpy(self.arrays["regions"][region_row].copy()).float(),
                "region_mask": torch.from_numpy(self.arrays["region_mask"][region_row].copy()),
                "region_boxes": torch.from_numpy(boxes),
                "caption": torch.tensor(caption, dtype=torch.long),
                "image_id": image_id, "references": self.data["references"][self.split][image_id]}
        if self.global_rows is not None:
            item["global_features"] = torch.from_numpy(self.arrays["global_features"][self.global_rows[image_id]].copy()).float()
        return item

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays = {}


def fusion_collate(items, pad_id):
    result = {name: torch.stack([item[name] for item in items])
              for name in ["grid", "regions", "region_mask", "region_boxes"]}
    if "global_features" in items[0]:
        result["global_features"] = torch.stack([item["global_features"] for item in items])
    result.update(captions=pad_sequence([item["caption"] for item in items], batch_first=True, padding_value=pad_id),
                  image_ids=[item["image_id"] for item in items],
                  references=[item["references"] for item in items])
    return result


def fusion_loader(dataset, batch_size, indices=None, seed=42, pin_memory=False):
    chosen = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(chosen, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=partial(fusion_collate, pad_id=dataset.data["ids"]["pad"]),
                      generator=torch.Generator().manual_seed(seed), pin_memory=pin_memory)


def fusion_device_batch(batch, device):
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()}


def fusion_logits(model, batch):
    extra = {"global_features": batch["global_features"]} if "global_features" in batch else {}
    return model(batch["grid"], batch["regions"], batch["captions"][:, :-1],
                 batch["region_mask"], batch["region_boxes"], **extra)


def fusion_greedy(model, batch, ids, max_steps):
    extra = {"global_features": batch["global_features"]} if "global_features" in batch else {}
    return model.greedy(batch["grid"], batch["regions"], ids["start"], ids["end"],
                        max_steps=max_steps, region_mask=batch["region_mask"],
                        region_boxes=batch["region_boxes"], **extra)


def fusion_optimizer_step(model, optimizer, scaler, clip=5.0, amp=True,
                          diagnostics=None, max_consecutive_overflows=8):
    """Unscale, clip and step; AMP overflows skip the update and reduce the scale."""
    if not math.isfinite(clip) or clip <= 0 or max_consecutive_overflows < 1:
        raise ValueError("Use a positive finite clip norm and overflow limit.")
    if diagnostics is None:
        # Standalone batch callers also retain a consecutive-overflow guard.
        if not hasattr(scaler, "fusion_diagnostics"):
            scaler.fusion_diagnostics = {}
        diagnostics = scaler.fusion_diagnostics
    diagnostics.setdefault("gradient_diagnostics_scope", "Counters start when overflow-aware stepping is first used; earlier checkpoint steps are not inferred.")
    scale_before = float(scaler.get_scale())
    if not math.isfinite(scale_before) or scale_before <= 0:
        raise FloatingPointError("AMP scale must stay finite and positive; inspect the saved checkpoint.")
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("No gradients are available for the optimizer step.")
    skipped = False
    try:
        norm = nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        gradients = [parameter.grad for parameter in parameters]
        nonfinite = any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if nonfinite:
            if not amp or not scaler.is_enabled():
                raise FloatingPointError("Nonfinite unscaled gradients without AMP; no optimizer step was taken.") from error
            optimizer_has_nonfinite = any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for group in optimizer.param_groups for parameter in group["params"])
            if not optimizer_has_nonfinite:
                raise FloatingPointError("Nonfinite gradients are outside the AMP optimizer; refusing an unchecked update.") from error
            # unscale_ recorded the overflow. GradScaler.step skips Adam safely.
            skipped = True
            norm = None
        else:
            # A float32 norm reduction can overflow even if every gradient is finite.
            # GradScaler would not skip that case, so clip using a float64 norm.
            norms = [torch.linalg.vector_norm(gradient.detach().double()) for gradient in gradients]
            norm = torch.linalg.vector_norm(torch.stack(norms))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("The gradient norm is still nonfinite in float64; no optimizer step was taken.") from error
            coefficient = min(1.0, clip / (float(norm) + 1e-6))
            with torch.no_grad():
                for gradient in gradients:
                    gradient.mul_(coefficient)
            diagnostics["gradient_norm_fallback_steps"] = diagnostics.get("gradient_norm_fallback_steps", 0) + 1

    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if not math.isfinite(scale_after) or scale_after <= 0:
        raise FloatingPointError("AMP scale became nonfinite or nonpositive; inspect the last safe checkpoint.")
    diagnostics.update(last_optimizer_step_skipped=skipped, last_amp_scale_before=scale_before,
                       last_amp_scale_after=scale_after, last_gradient_norm=None if norm is None else float(norm))
    if skipped:
        diagnostics["amp_skipped_steps"] = diagnostics.get("amp_skipped_steps", 0) + 1
        diagnostics["epoch_amp_skipped_steps"] = diagnostics.get("epoch_amp_skipped_steps", 0) + 1
        diagnostics["amp_consecutive_skips"] = diagnostics.get("amp_consecutive_skips", 0) + 1
        event = {"epoch":diagnostics.get("epoch"), "next_index_before_batch":diagnostics.get("next_index"),
                 "scale_before":scale_before, "scale_after":scale_after,
                 "consecutive_skips":diagnostics["amp_consecutive_skips"]}
        diagnostics.setdefault("amp_overflow_events", []).append(event)
        print(f"AMP overflow: skipped optimizer step; scale {scale_before:g} -> {scale_after:g} "
              f"(consecutive {diagnostics['amp_consecutive_skips']}/{max_consecutive_overflows}).", flush=True)
        if scale_after >= scale_before:
            raise FloatingPointError("AMP overflow did not reduce its scale; stop before further training.")
        if diagnostics["amp_consecutive_skips"] >= max_consecutive_overflows:
            raise FloatingPointError(f"{max_consecutive_overflows} consecutive AMP overflows; stop and inspect the model/data.")
    else:
        diagnostics["amp_consecutive_skips"] = 0
        diagnostics["successful_optimizer_steps"] = diagnostics.get("successful_optimizer_steps", 0) + 1
        diagnostics["epoch_successful_optimizer_steps"] = diagnostics.get("epoch_successful_optimizer_steps", 0) + 1
    return skipped


def fusion_train_batch(model, batch, optimizer, scaler, device, pad_id, amp=True, clip=5.0,
                       diagnostics=None, max_consecutive_overflows=8):
    model.train()
    batch = fusion_device_batch(batch, device)
    targets = batch["captions"][:, 1:]
    token_count = int(targets.ne(pad_id).sum())
    assert token_count > 0
    optimizer.zero_grad(set_to_none=True)
    enabled = amp and torch.device(device).type == "cuda"
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=enabled):
        logits = fusion_logits(model, batch)
        loss_sum = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                               ignore_index=pad_id, reduction="sum")
        loss = loss_sum / token_count
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite caption loss; preserve the last committed checkpoint.")
    scale = float(scaler.get_scale())
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError("AMP scale must stay finite and positive before backward.")
    scaler.scale(loss).backward()
    fusion_optimizer_step(model, optimizer, scaler, clip, amp, diagnostics, max_consecutive_overflows)
    # Loss describes the evaluated tokens, including a batch whose AMP update was skipped.
    return float(loss_sum.detach()), token_count


class FusionTimeLimit(RuntimeError):
    pass


@torch.no_grad()
def evaluate_fusion(model, dataset, data, config, device, evaluator, hard_deadline=None):
    if dataset.split != "val":
        raise ValueError("This run only evaluates validation; test remains unused.")
    if not config.get("diagnostic", False):
        assert dataset.image_ids == data["split_ids"]["val"] and len(dataset) == 5000
    model.eval()
    records, total_loss, total_tokens = [], 0.0, 0
    enabled = config.get("amp", True) and torch.device(device).type == "cuda"
    deadline = fusion_deadline(hard_deadline)
    for number, batch in enumerate(fusion_loader(dataset, config.get("validation_batch_size", config["batch_size"]),
                                                seed=config["seed"], pin_memory=enabled), 1):
        if time.time() >= deadline:
            raise FusionTimeLimit("Validation reached the hard deadline. Partial-corpus metrics are not calculated.")
        moved = fusion_device_batch(batch, device)
        targets = moved["captions"][:, 1:]
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=enabled):
            logits = fusion_logits(model, moved)
            loss_sum = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                   ignore_index=data["ids"]["pad"], reduction="sum")
            predicted = fusion_greedy(model, moved, data["ids"], config.get("max_generation_steps", 33))
        total_loss += float(loss_sum)
        total_tokens += int(targets.ne(data["ids"]["pad"]).sum())
        for image_id, tokens, refs in zip(batch["image_ids"], predicted, batch["references"]):
            records.append({"image_id": image_id, "prediction": " ".join(data["id_to_token"][token] for token in tokens),
                            "token_ids": tokens, "references": refs})
        if number % 20 == 0:
            print(f"  Validation generation {len(records):,}/{len(dataset):,}", flush=True)
    assert len(records) == len(dataset) and total_tokens > 0
    scores = evaluator.score(records, expected_ids=dataset.image_ids)
    return {"loss": total_loss / total_tokens, **scores}, records


def _fusion_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_fusion_checkpoint(path, model, optimizer, scaler, state, identity):
    payload = {"format_version": 1, "identity": identity, "state": copy.deepcopy(state),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "rng": fusion_rng()}
    _fusion_torch_save(path, payload)
    _fusion_torch_save(Path(path).with_name(Path(path).stem + "_weights.pt"), model.state_dict())


def load_fusion_checkpoint(path, model, optimizer, scaler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity:
        raise ValueError("Checkpoint config, model, data, subset, feature cache or scorer differs.")
    assert payload["format_version"] == 1
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    restore_fusion_rng(payload["rng"])
    state = payload["state"]
    timing = Path(path).parent / "timing.json"
    if timing.exists():
        latest = json.loads(timing.read_text(encoding="utf-8"))
        if latest.get("checkpoint_serial") == state.get("checkpoint_serial"):
            state["training_seconds"] = latest["training_seconds"]
            state["epoch_seconds"] = latest["epoch_seconds"]
    return state


def fusion_summary(folder, config, state, parameter_count, train_ids):
    best = state["best_record"] or {}
    summary = {"encoder": "resnet50_spatial+fasterrcnn_resnet50_fpn", "decoder": "gated_cross_attention_transformer",
               "run_name": config["run_name"], "train_images": len(train_ids), "validation_images": state["validation_images"],
               "trainable_parameters": parameter_count, "best_epoch": state["best_epoch"],
               "best_is_initial_baseline": state["best_epoch"] == 0, "best_cider_d": state["best_score"],
               "val_loss": best.get("val_loss"), "bleu1": best.get("bleu1"), "bleu4": best.get("bleu4"),
               "cider_d": best.get("cider_d"), "bad_epochs": state["bad_epochs"],
               "epoch": state["epoch"], "phase": state["phase"], "stop_reason": state.get("stop_reason"),
               "training_seconds": state["training_seconds"], "training_time": fusion_time(state["training_seconds"]),
               "initial_validation_seconds": state.get("initial_validation_seconds", 0), "devices": state["devices"],
               "amp_skipped_steps": state.get("amp_skipped_steps", 0),
               "successful_optimizer_steps": state.get("successful_optimizer_steps", 0),
               "amp_consecutive_skips": state.get("amp_consecutive_skips", 0),
               "last_amp_scale": state.get("last_amp_scale_after"),
               "gradient_norm_fallback_steps": state.get("gradient_norm_fallback_steps", 0),
               "gradient_diagnostics_scope": state.get("gradient_diagnostics_scope", "No overflow-aware optimizer steps recorded yet."),
               "loss_note": "Train loss counts every evaluated non-PAD target token; AMP-skipped batches do not update weights and are counted separately.",
               "data_note": f"Fusion stage uses {len(train_ids):,} images from its saved train ID list. The warm-start caption model previously trained on the full original project train split.",
               "timing_note": "Active run includes initial validation, training, validation and checkpoint I/O; excludes loading, extraction, preflight and paused downtime.",
               "test_evaluated": False}
    fusion_json(folder / "summary.json", summary)
    fusion_json(folder / "history.json", state["history"])
    if state["history"]:
        temporary = folder / "history.csv.tmp"
        columns = ["epoch", "train_loss", "val_loss", "bleu1", "bleu4", "cider_d", "train_images_seen",
                   "partial_epoch", "validation_seconds", "epoch_seconds", "amp_skipped_steps",
                   "epoch_amp_skipped_steps", "successful_optimizer_steps", "epoch_successful_optimizer_steps",
                   "amp_scale"]
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in columns} for row in state["history"]])
        temporary.replace(folder / "history.csv")
    return summary


def train_fusion(data, config, evaluator, device, output_root, model_factory, train_dataset,
                 val_dataset, optimizer_factory=None, resume=True, after_checkpoint=None):
    """One initialized architecture, with all-data validation and safe time-limit stopping."""
    device = torch.device(device)
    config = dict(config)
    assert config["seed"] == 42
    assert train_dataset.split == "train" and val_dataset.split == "val"
    assert train_dataset.image_ids == sorted(train_dataset.image_ids)
    if not config.get("diagnostic", False):
        assert val_dataset.image_ids == data["split_ids"]["val"] and len(val_dataset) == 5000
        assert config.get("evaluate_initial", True), "Save the complete initialized baseline first."
    name = config["run_name"]
    if Path(name).name != name or name in [".", ".."]:
        raise ValueError("run_name must be one directory name.")
    folder = Path(output_root) / name
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "data": data["provenance"], "metrics": evaluator.metadata,
                "train_ids": train_dataset.image_ids, "validation_ids": val_dataset.image_ids,
                "grid_and_region_data": {"train": train_dataset.provenance, "val": val_dataset.provenance}}
    if (folder / "config.json").exists():
        if json.loads((folder / "config.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing run differs from the agreed config or data. Use a new name.")
        if not resume:
            raise ValueError("Existing run needs resume=True; it will not be replaced.")
    elif any(folder.iterdir()):
        raise ValueError("Output directory is not empty and has no matching configuration.")
    fusion_json(folder / "config.json", identity)
    fusion_json(folder / "train_image_ids.json", train_dataset.image_ids)
    fusion_json(folder / "validation_image_ids.json", val_dataset.image_ids)
    fusion_json(folder / "vocabulary.json", {"id_to_token": data["id_to_token"], "token_to_id": data["token_to_id"],
                                             "source_sha256": data["provenance"].get("vocab_sha256")})
    fusion_seed(config["seed"])
    model = model_factory().to(device)
    optimizer = optimizer_factory(model) if optimizer_factory else torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    enabled = config.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if shutil.disk_usage(folder).free < count * 4 * 10 + 128 * 1024**2:
        raise OSError("Insufficient space for full and plain best/last checkpoints plus atomic temporary writes.")
    details = {"device": str(device), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    state = {"epoch": 0 if config.get("evaluate_initial", True) else 1,
             "phase": "initial_evaluate" if config.get("evaluate_initial", True) else "train",
             "next_index": 0, "train_loss_sum": 0.0, "train_token_count": 0, "best_score": None,
             "best_epoch": None, "best_record": None, "bad_epochs": 0, "history": [], "devices": [details],
             "training_seconds": 0.0, "epoch_seconds": 0.0, "validation_images": len(val_dataset),
             "checkpoint_serial": 0, "stop_reason": None, "partial_epoch": False}
    if (folder / "last.pt").exists():
        state = load_fusion_checkpoint(folder / "last.pt", model, optimizer, scaler, identity)
        if details not in state["devices"]:
            state["devices"].append(details)
        print(f"Resume epoch {state['epoch']}, phase {state['phase']}, trained images {state['next_index']:,}", flush=True)
    if state["phase"] == "finished":
        summary = fusion_summary(folder, config, state, count, train_dataset.image_ids)
        train_dataset.close()
        val_dataset.close()
        return summary

    hard_stop = fusion_deadline(config.get("hard_deadline_utc"))
    admission_stop = fusion_deadline(config.get("epoch_admission_deadline_utc", config.get("train_deadline_utc")))
    started = time.perf_counter()
    previous_seconds = state["training_seconds"]
    epoch_started = started
    previous_epoch_seconds = state["epoch_seconds"]

    def commit(best=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        state["training_seconds"] = previous_seconds + time.perf_counter() - started
        state["epoch_seconds"] = previous_epoch_seconds + time.perf_counter() - epoch_started
        state["checkpoint_serial"] += 1
        if best:
            write_fusion_checkpoint(folder / "best.pt", model, optimizer, scaler, state, identity)
        write_fusion_checkpoint(folder / "last.pt", model, optimizer, scaler, state, identity)
        state["training_seconds"] = previous_seconds + time.perf_counter() - started
        state["epoch_seconds"] = previous_epoch_seconds + time.perf_counter() - epoch_started
        fusion_json(folder / "timing.json", {"checkpoint_serial": state["checkpoint_serial"],
                    "training_seconds": state["training_seconds"], "epoch_seconds": state["epoch_seconds"]})
        fusion_summary(folder, config, state, count, train_dataset.image_ids)
        fusion_json(folder / "status.json", {"phase": state["phase"], "epoch": state["epoch"],
                    "next_index": state["next_index"], "train_images": len(train_dataset),
                    "best_epoch": state["best_epoch"], "best_cider_d": state["best_score"],
                    "bad_epochs": state["bad_epochs"], "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "amp_skipped_steps": state.get("amp_skipped_steps", 0),
                    "amp_consecutive_skips": state.get("amp_consecutive_skips", 0),
                    "amp_scale": state.get("last_amp_scale_after"),
                    "stop_reason": state.get("stop_reason"), "active_seconds": state["training_seconds"]})
        if after_checkpoint is not None:
            after_checkpoint(copy.deepcopy(state))

    if not (folder / "last.pt").exists():
        commit()
    try:
        while state["phase"] != "finished":
            if state["phase"] == "epoch_done":
                reserve = max(config.get("validation_reserve_seconds", 240),
                              max((row["validation_seconds"] for row in state["history"]), default=0) * 1.35 + 60)
                latest_train_seconds = state["history"][-1].get("train_seconds", 0)
                if (time.time() >= admission_stop or time.time() + reserve + latest_train_seconds * 1.1 >= hard_stop):
                    state.update(phase="finished", stop_reason="time_budget_before_new_epoch")
                    commit()
                    break
                state.update(epoch=state["epoch"] + 1, phase="train", next_index=0,
                             train_loss_sum=0.0, train_token_count=0, epoch_seconds=0.0, partial_epoch=False,
                             epoch_amp_skipped_steps=0, epoch_successful_optimizer_steps=0)
                epoch_started = time.perf_counter()
                previous_epoch_seconds = 0.0

            if state["phase"] == "train":
                epoch = state["epoch"]
                train_dataset.set_epoch(epoch)
                order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
                completed_at_start = state["next_index"]
                segment_start = time.perf_counter()
                print(f"Epoch {epoch}/{config.get('max_epochs',100)} | {len(train_dataset):,} train images | active {fusion_time(state['training_seconds'])}", flush=True)
                for batch in fusion_loader(train_dataset, config["batch_size"], order[state["next_index"]:],
                                            seed=config["seed"] + epoch, pin_memory=enabled):
                    reserve = max(config.get("validation_reserve_seconds", 240),
                                  max((row["validation_seconds"] for row in state["history"]), default=0) * 1.35 + 60)
                    if time.time() + reserve >= hard_stop:
                        state.update(partial_epoch=True, stop_reason="time_budget_partial_epoch")
                        break
                    loss_sum, tokens = fusion_train_batch(model, batch, optimizer, scaler, device, data["ids"]["pad"],
                                                          amp=config.get("amp", True), clip=config.get("gradient_clip", 5.0),
                                                          diagnostics=state)
                    state["train_loss_sum"] += loss_sum
                    state["train_token_count"] += tokens
                    state["next_index"] += len(batch["image_ids"])
                    number = (state["next_index"] + config["batch_size"] - 1) // config["batch_size"]
                    if number % config.get("print_every_batches", 100) == 0:
                        elapsed = time.perf_counter() - segment_start
                        throughput = (state["next_index"] - completed_at_start) / max(elapsed, 1e-6)
                        eta = (len(train_dataset) - state["next_index"]) / max(throughput, 1e-6)
                        print(f"  {state['next_index']:,}/{len(train_dataset):,} | loss {state['train_loss_sum']/state['train_token_count']:.4f} | {throughput:.1f} img/s | train ETA {fusion_time(eta)}", flush=True)
                        fusion_json(folder / "status.json", {"phase":"train", "epoch":epoch, "next_index":state["next_index"],
                            "train_images":len(train_dataset), "train_loss":state["train_loss_sum"]/state["train_token_count"],
                            "images_per_second":throughput, "train_eta_seconds":eta,
                            "updated_utc":datetime.now(timezone.utc).isoformat(), "best_epoch":state["best_epoch"],
                            "best_cider_d":state["best_score"], "bad_epochs":state["bad_epochs"],
                            "amp_skipped_steps":state.get("amp_skipped_steps",0),
                            "amp_consecutive_skips":state.get("amp_consecutive_skips",0),
                            "amp_scale":state.get("last_amp_scale_after")})
                    if state.get("last_optimizer_step_skipped",False) or number % config.get("checkpoint_every_batches", 200) == 0:
                        commit()
                state["phase"] = "evaluate"
                state["train_seconds"] = previous_epoch_seconds + time.perf_counter() - epoch_started
                commit()

            if state["phase"] in ["initial_evaluate", "evaluate"]:
                initial = state["phase"] == "initial_evaluate"
                print(f"Full validation for {'initialized baseline (epoch 0)' if initial else 'epoch '+str(state['epoch'])}: {len(val_dataset):,} images", flush=True)
                validation_start = time.perf_counter()
                scores, records = evaluate_fusion(model, val_dataset, data, config, device, evaluator, config.get("hard_deadline_utc"))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_seconds = time.perf_counter() - validation_start
                if initial:
                    state["initial_validation_seconds"] = validation_seconds
                row = {"epoch":state["epoch"],
                       "train_loss":state["train_loss_sum"]/state["train_token_count"] if state["train_token_count"] else None,
                       "val_loss":scores["loss"], "bleu1":scores["bleu1"], "bleu4":scores["bleu4"], "cider_d":scores["cider_d"],
                       "train_images_seen":state["next_index"], "partial_epoch":state["partial_epoch"],
                       "train_seconds":state.get("train_seconds",0), "validation_seconds":validation_seconds,
                       "epoch_seconds":previous_epoch_seconds + time.perf_counter() - epoch_started,
                       "amp_skipped_steps":state.get("amp_skipped_steps",0),
                       "epoch_amp_skipped_steps":state.get("epoch_amp_skipped_steps",0),
                       "successful_optimizer_steps":state.get("successful_optimizer_steps",0),
                       "epoch_successful_optimizer_steps":state.get("epoch_successful_optimizer_steps",0),
                       "amp_scale":float(scaler.get_scale())}
                state["history"].append(row)
                fusion_json(folder / "predictions" / f"val_epoch_{state['epoch']:03d}.json", records)
                fusion_json(folder / "validation_last.json", records)
                improved = state["best_score"] is None or scores["cider_d"] > state["best_score"]
                if improved:
                    state.update(best_score=scores["cider_d"], best_epoch=state["epoch"], best_record=row, bad_epochs=0)
                    fusion_json(folder / "validation_best.json", records)
                elif not initial:
                    state["bad_epochs"] += 1
                done = state["partial_epoch"] or state["bad_epochs"] >= config.get("patience",3) or state["epoch"] >= config.get("max_epochs",100)
                if done and state["stop_reason"] is None:
                    state["stop_reason"] = "early_stopping" if state["bad_epochs"] >= config.get("patience",3) else "max_epochs"
                state["phase"] = "finished" if done else "epoch_done"
                commit(best=improved)
                print(f"Epoch {state['epoch']} | val loss {scores['loss']:.4f} | BLEU-4 {scores['bleu4']:.4f} | CIDEr-D {scores['cider_d']:.4f} | patience {state['bad_epochs']}/{config.get('patience',3)} | active {fusion_time(state['training_seconds'])}", flush=True)
        return fusion_summary(folder, config, state, count, train_dataset.image_ids)
    except FusionTimeLimit as error:
        state.update(phase="finished", stop_reason="hard_deadline_during_validation")
        commit()
        print(str(error), flush=True)
        return fusion_summary(folder, config, state, count, train_dataset.image_ids)
    except BaseException:
        # Do not serialize an optimizer that may have been interrupted inside step().
        # last.pt records the last fully committed batch; later work can repeat safely.
        fusion_json(folder / "status.json", {"phase":"interrupted", "resume_from":"last.pt",
                    "updated_utc":datetime.now(timezone.utc).isoformat(), "test_evaluated":False,
                    "amp_skipped_steps":state.get("amp_skipped_steps",0),
                    "amp_consecutive_skips":state.get("amp_consecutive_skips",0),
                    "amp_scale":state.get("last_amp_scale_after")})
        raise
    finally:
        train_dataset.close()
        val_dataset.close()
