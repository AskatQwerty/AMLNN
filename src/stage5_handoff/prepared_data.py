import json
import hashlib
import pickle
from pathlib import Path
from collections import Counter, defaultdict


def load_region_prepared_data(project_dir, data_root=None):
    project_dir = Path(project_dir)
    data_root = Path(data_root) if data_root is not None else project_dir / "data" / "coco2017"
    for source in ["train2017", "val2017"]:
        annotation_path = data_root / "annotations" / f"captions_{source}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing {annotation_path}. Extract the shared training-data bundle beside "
                "Stage5_Starter.ipynb, or set DATA_ROOT to the bundled annotation folder. "
                "Raw train2017/val2017 JPEGs are required for region extraction."
            )
    cache = project_dir / "cache"
    required = [
        "splits.json",
        "prepared.pkl",
        "text_report.json",
        "vocab.json",
        "image_audit.json",
        "test_captions.json",
    ]
    missing = [name for name in required if not (cache / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared files: {missing}. Extract the shared training-data bundle into PROJECT_DIR. "
            "Do not recreate the split or vocabulary for this comparison."
        )

    # Load saved preparation results, then verify their versions agree.
    split_bytes = (cache / "splits.json").read_bytes()
    vocab_bytes = (cache / "vocab.json").read_bytes()
    prepared_bytes = (cache / "prepared.pkl").read_bytes()
    split_file = json.loads(split_bytes)
    vocab = json.loads(vocab_bytes)
    report = json.loads((cache / "text_report.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "image_audit.json").read_text(encoding="utf-8"))

    prepared_sha = hashlib.sha256(prepared_bytes).hexdigest()
    assert (
        prepared_sha == report["cache_sha256"]
    ), "Prepared caption cache changed. Rerun preparation."
    prepared = pickle.loads(
        prepared_bytes
    )  # This is our own cache, checked against its saved hash.
    del prepared_bytes

    split_ids = split_file["ids"]
    sources = split_file["feature_sources"]
    assert set(split_ids) == {"train", "val", "test"}
    assert sources == {
        "train": "train2017",
        "val": "val2017",
        "test": "train2017",
    }, "Use the saved labeled split, not official test2017."
    payload = {"config": split_file["config"], "ids": split_ids, "feature_sources": sources}
    split_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert split_id == split_file["split_id"] == prepared["config"]["split_id"] == vocab["split_id"]

    source_id = split_file["config"]["source_id"]
    assert source_id == prepared["config"]["source_id"] == vocab["source_id"] == audit["source_id"]
    assert prepared["config"] == report["config"]
    assert audit["complete"] and not audit["image_errors"] and not audit["metadata_errors"]
    assert (
        prepared["token_to_id"] == vocab["token_to_id"]
        and prepared["id_to_token"] == vocab["id_to_token"]
    )
    assert vocab["token_to_id"] == {word: index for index, word in enumerate(vocab["id_to_token"])}

    ids = {
        name: vocab["token_to_id"][token]
        for name, token in [
            ("pad", "<pad>"),
            ("unk", "<unk>"),
            ("start", "<start>"),
            ("end", "<end>"),
        ]
    }
    assert len(set(ids.values())) == 4

    # One photograph and its exact duplicates must stay in one labeled split.
    sets = {name: set(values) for name, values in split_ids.items()}
    for name, values in split_ids.items():
        assert values == sorted(sets[name]) and values, f"Invalid or repeated IDs in {name}."
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not sets[left] & sets[right], f"Image overlap: {left}/{right}."
    assert len(sets["test"]) == split_file["config"]["holdout_size"]

    owners = {
        (sources[name], image_id): name for name, values in sets.items() for image_id in values
    }
    for group in audit["pixel_duplicates"] + audit["file_duplicates"]:
        assigned = {owners[tuple(item)] for item in group if tuple(item) in owners}
        assert len(assigned) <= 1, "Exact duplicate photos cross the labeled splits."

    # Keep the original human references for evaluation.
    raw_by_source = {}
    source_image_ids = {}
    annotation_hashes = {}

    for source in ["train2017", "val2017"]:
        annotation_bytes = (data_root / "annotations" / f"captions_{source}.json").read_bytes()
        annotation_hashes[source] = hashlib.sha256(annotation_bytes).hexdigest()
        annotation = json.loads(annotation_bytes)
        source_image_ids[source] = sorted(row["id"] for row in annotation["images"])
        assert len(source_image_ids[source]) == len(set(source_image_ids[source]))

        raw_by_source[source] = defaultdict(list)
        for row in annotation["annotations"]:
            assert (
                isinstance(row["caption"], str) and row["caption"].strip()
            ), "Empty human reference."
            raw_by_source[source][row["image_id"]].append(row["caption"])
        assert set(raw_by_source[source]) == set(
            source_image_ids[source]
        ), "Missing human descriptions."

        if source == "train2017":
            saved_test = json.loads((cache / "test_captions.json").read_text(encoding="utf-8"))
            assert saved_test["split_id"] == split_id
            expected_test = [
                row for row in annotation["annotations"] if row["image_id"] in sets["test"]
            ]
            assert (
                saved_test["annotations"] == expected_test
            ), "Saved test references differ from the original human captions."
        del annotation_bytes, annotation

    assert sets["train"] | sets["test"] == set(source_image_ids["train2017"])
    assert sets["val"] == set(source_image_ids["val2017"])

    references = {
        name: {image_id: raw_by_source[sources[name]][image_id] for image_id in values}
        for name, values in split_ids.items()
    }
    tokens = {name: prepared[f"{name}_tokens"] for name in split_ids}
    for name, values in tokens.items():
        assert sorted(values) == split_ids[name]
        assert all(
            rows and all(row for row in rows) for rows in values.values()
        ), "Empty tokenized caption."
        if name != "train":
            assert all(
                len(rows) == len(references[name][image_id]) for image_id, rows in values.items()
            ), "Evaluation references were dropped."

    # Test and validation captions must not contribute to the vocabulary.
    train_counts = Counter(
        token for rows in tokens["train"].values() for row in rows for token in row
    )
    assert train_counts == prepared["train_counts"]
    settings = prepared["config"]["settings"]
    regular = [
        token
        for token, count in sorted(train_counts.items(), key=lambda row: (-row[1], row[0]))
        if count >= settings["min_frequency"]
    ][: settings["max_vocab_tokens"]]
    assert (
        vocab["id_to_token"] == settings["special_tokens"] + regular
    ), "Vocabulary must be fitted on training captions only."

    data = {
        "project_dir": project_dir,
        "data_root": data_root,
        "split_ids": split_ids,
        "feature_sources": sources,
        "tokens": tokens,
        "references": references,
        "source_image_ids": source_image_ids,
        "token_to_id": vocab["token_to_id"],
        "id_to_token": vocab["id_to_token"],
        "ids": ids,
        "max_content_tokens": prepared["config"]["max_content_tokens"],
        "text_settings": settings,
        "provenance": {
            "source_id": source_id,
            "split_id": split_id,
            "prepared_sha256": prepared_sha,
            "vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
            "splits_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "annotations_sha256": annotation_hashes,
        },
    }

    print(
        "Verified disjoint labeled splits:",
        {name: len(values) for name, values in split_ids.items()},
    )
    print("Vocabulary:", len(data["id_to_token"]), "| special IDs:", ids)
    return data
