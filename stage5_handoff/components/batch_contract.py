import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


class RegionCaptionDataset(Dataset):
    def __init__(self, data, cache_files, split, seed=42):
        if split not in ["train", "val"]:
            raise ValueError("Use train or val. Test evaluation is outside this training stage.")
        self.data, self.split, self.seed = data, split, seed
        self.image_ids = data["split_ids"][split]
        self.features = self.visual_masks = self.num_regions = None
        source = data["feature_sources"][split]
        assert source in ["train2017", "val2017"]
        saved = cache_files[source]
        source_ids = json.loads(Path(saved["image_ids_path"]).read_text(encoding="utf-8"))
        assert source_ids == data["source_image_ids"][source], "Different cache row order."
        assert len(source_ids) == len(set(source_ids)), "Duplicate cache image IDs."
        self.feature_rows = {image_id: row for row, image_id in enumerate(source_ids)}
        assert all(image_id in self.feature_rows for image_id in self.image_ids)
        assert data["max_content_tokens"] == 32, "Keep the shared 32-content-token limit."
        try:
            self.features = np.load(saved["features_path"], mmap_mode="r", allow_pickle=False)
            self.visual_masks = np.load(saved["mask_path"], mmap_mode="r", allow_pickle=False)
            self.num_regions = np.load(saved["num_regions_path"], mmap_mode="r", allow_pickle=False)
            n = len(source_ids)
            assert self.features.shape == (n, 36, 1024) and self.features.dtype == np.float16
            assert self.visual_masks.shape == (n, 36) and self.visual_masks.dtype == np.bool_
            assert self.num_regions.shape == (n,) and self.num_regions.dtype == np.uint8
            assert np.all((self.num_regions >= 1) & (self.num_regions <= 36))
            assert np.array_equal(self.visual_masks.sum(axis=1), self.num_regions)
            self.set_epoch(0)
        except Exception:
            self.close()
            raise

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        self.caption_choices = [
            rng.randrange(len(self.data["tokens"][self.split][image_id]))
            if self.split == "train" else 0
            for image_id in self.image_ids
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        row = self.feature_rows[image_id]
        tokens = self.data["tokens"][self.split][image_id][self.caption_choices[index]]
        ids = self.data["ids"]
        caption = [ids["start"]] + [
            self.data["token_to_id"].get(token, ids["unk"]) for token in tokens[:32]
        ] + [ids["end"]]
        return {
            "features": torch.from_numpy(self.features[row].copy()).float(),
            "visual_mask": torch.from_numpy(self.visual_masks[row].copy()),
            "caption": torch.tensor(caption, dtype=torch.long),
            "image_id": image_id,
            "references": self.data["references"][self.split][image_id],
        }

    def close(self):
        for name in ["features", "visual_masks", "num_regions"]:
            array = getattr(self, name, None)
            if array is not None:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
                elif hasattr(array, "close"):
                    array.close()
                setattr(self, name, None)


def region_caption_collate(batch, pad_id):
    captions = [item["caption"] for item in batch]
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "visual_mask": torch.stack([item["visual_mask"] for item in batch]),
        "captions": pad_sequence(captions, batch_first=True, padding_value=pad_id),
        "lengths": torch.tensor([len(caption) for caption in captions], dtype=torch.long),
        "image_ids": [item["image_id"] for item in batch],
        "references": [item["references"] for item in batch],
    }


def make_region_loader(dataset, batch_size=64, indices=None, seed=42, pin_memory=False):
    # Pass the shared shuffled index list for training; validation keeps source split order.
    selected = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        selected, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(region_caption_collate, pad_id=dataset.data["ids"]["pad"]),
        pin_memory=pin_memory, generator=torch.Generator().manual_seed(seed),
    )


def train_region_batch(model, batch, optimizer, scaler, device, pad_id, clip=5, amp=True):
    device = torch.device(device)
    use_amp = amp and device.type == "cuda"
    if use_amp and (scaler is None or not scaler.is_enabled()):
        raise ValueError("CUDA AMP requires an enabled GradScaler.")
    model.train()
    features = batch["features"].to(device)
    visual_mask = batch["visual_mask"].to(device)
    captions = batch["captions"].to(device)
    inputs, targets = captions[:, :-1], captions[:, 1:]
    token_count = int(targets.ne(pad_id).sum())
    if token_count == 0:
        raise ValueError("The batch has no non-padding target tokens.")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(features, inputs, visual_mask=visual_mask)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
        loss_sum = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss = loss_sum / token_count
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    # Epoch loss = total_loss_sum / total_token_count, not an average of batch averages.
    return float(loss_sum.detach()), token_count


@torch.no_grad()
def generate_region_batch(model, batch, device, ids, max_steps=33):
    return model.greedy(
        batch["features"].to(device), ids["start"], ids["end"], max_steps=max_steps,
        visual_mask=batch["visual_mask"].to(device),
    )
