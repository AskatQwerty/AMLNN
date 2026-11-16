"""Review cached fusion data, or explicitly start/resume the single final run."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
import gc
import hashlib
import json
import random
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.resolve()
sys.path.append(str(BASE.parent / "stage5_handoff"))
from prepared_data import load_region_prepared_data
from metrics import MetricEvaluator
from region_features import build_frozen_region_detector
from region_cache import load_fusion_region_cache, extract_fusion_region_cache
from models import FusionCaptionTransformer, warm_start_from_prefix, fusion_parameter_groups
from runtime import FusionCaptionDataset, train_fusion, fusion_json


PROJECT_DIR = Path(r"C:\Users\askat\Study\3.1\AML Applied Machine Learning\Final")
RAW_IMAGE_ROOT = Path(r"C:\Users\askat\Downloads\MSCOCO\coco2017")
RUN_NAME = "resnet50_fasterrcnn_transformer_fusion_depth3_seed42"
WARM_START_RUN = "resnet50_transformer_depth_3_seed42"
WARM_START_SHA256 = "bff8c8c6d4a8c54205c3b8502c0ef03938d5dfc2a786379f8157eb9151f40b73"
EXTRACTION_DEADLINE_UTC = "2026-11-15T15:00:00+00:00"
EPOCH_ADMISSION_DEADLINE_UTC = None
HARD_DEADLINE_UTC = None


def final_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_ast_sha256(source, remove_imports=False):
    tree = ast.parse(source)
    if remove_imports:
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def final_code_identity(source_dir=None, notebook_path=None, expected=None):
    """Notebook code is checked against its saved tagged cells before training."""
    if notebook_path is not None:
        if expected is None:
            raise ValueError("The notebook needs its generated source fingerprints.")
        notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
        cells = {cell.get("metadata", {}).get("final_fusion_section"): "".join(cell["source"])
                 for cell in notebook["cells"] if cell["cell_type"] == "code"}
        for name in ["models", "runtime", "region_features", "region_cache"]:
            if final_ast_sha256(cells[name]) != expected[name + "_cell_ast_sha256"]:
                raise ValueError(f"The saved notebook {name} code changed; run and save new matching checks first.")
        return dict(expected)
    source_dir = Path(source_dir)
    paths = {"models":source_dir/"models.py", "runtime":source_dir/"runtime.py",
             "region_features":source_dir.parent/"stage5_handoff/region_features.py",
             "region_cache":source_dir/"region_cache.py"}
    identity = {}
    for name,path in paths.items():
        source = path.read_text(encoding="utf-8")
        identity[name + "_sha256"] = final_file_sha256(path)
        identity[name + "_cell_ast_sha256"] = final_ast_sha256(source, remove_imports=True)
    identity["model_runtime_implementation_id"] = hashlib.sha256("\n".join(
        ast.dump(ast.parse(paths[name].read_text(encoding="utf-8"))) for name in ["models","runtime"]
    ).encode()).hexdigest()
    identity["region_producer_code_id"] = hashlib.sha256("\n".join(
        ast.dump(ast.parse(paths[name].read_text(encoding="utf-8"))) for name in ["region_features","region_cache"]
    ).encode()).hexdigest()
    return identity


def load_final_grid_features(data, project_dir):
    root = Path(project_dir)/"cache/features"
    grid_root = root/"resnet50_spatial_v1"
    files = {}
    for source in ["train2017","val2017"]:
        grid_path, global_path = grid_root/f"{source}.npy", root/f"{source}.npy"
        grid_meta = json.loads((grid_root/f"{source}_meta.json").read_text(encoding="utf-8"))
        global_meta = json.loads((root/f"{source}_meta.json").read_text(encoding="utf-8"))
        grid_ids = json.loads((grid_root/f"{source}_image_ids.json").read_text(encoding="utf-8"))
        global_ids = json.loads((root/f"{source}_image_ids.json").read_text(encoding="utf-8"))
        assert grid_ids == global_ids == data["source_image_ids"][source]
        assert grid_meta["complete"] and global_meta["complete"]
        for meta in [grid_meta,global_meta]:
            assert meta["config"]["source_id"] == data["provenance"]["source_id"]
            assert meta["config"]["weights"] == "ResNet50_Weights.IMAGENET1K_V2"
            key = hashlib.sha256(json.dumps(meta["config"],sort_keys=True).encode()).hexdigest()
            assert key == meta["cache_key"], "Cached ResNet50 configuration fingerprint changed."
        assert grid_meta["config"]["global_cache_keys"][source] == global_meta["cache_key"]
        assert grid_path.stat().st_size == grid_meta["feature_file_bytes"]
        assert global_path.stat().st_size == global_meta["feature_file_bytes"]
        for path,shape,dtype in [(grid_path,(len(grid_ids),49,2048),np.float16),
                                 (global_path,(len(grid_ids),2048),np.float32)]:
            array = np.load(path,mmap_mode="r",allow_pickle=False)
            try:
                assert array.shape == shape and array.dtype == dtype
            finally:
                array._mmap.close()
        files[source] = {"path":str(grid_path),"image_ids":grid_ids,"cache_key":grid_meta["cache_key"],
            "global_path":str(global_path),"global_image_ids":global_ids,"global_cache_key":global_meta["cache_key"],
            "grid_meta_sha256":final_file_sha256(grid_root/f"{source}_meta.json"),
            "global_meta_sha256":final_file_sha256(root/f"{source}_meta.json")}
    return files


def validate_final_control(data, project_dir, evaluator):
    control = Path(project_dir)/"experiments/all_encoders_v5_100epochs"/WARM_START_RUN
    settings = json.loads((control/"config.json").read_text(encoding="utf-8"))
    assert settings["metrics"] == evaluator.metadata, "Keep the exact existing caption scorer and tokenizer."
    for key in ["source_id","split_id","prepared_sha256","vocab_sha256","splits_sha256","annotations_sha256"]:
        assert settings["data"][key] == data["provenance"][key], f"Warm-start data differs: {key}"
    assert final_file_sha256(control/"best_weights.pt") == WARM_START_SHA256
    return control


def require_full_fusion_train(data, region_files):
    """The final experiment requires every saved train image; a pilot cache is insufficient."""
    expected_ids = sorted(data["split_ids"]["train"])
    assert len(expected_ids) == 113287 and len(set(expected_ids)) == 113287
    train_ids = sorted(region_files["train2017"]["image_ids"])
    if train_ids != expected_ids:
        missing = len(set(expected_ids)-set(train_ids))
        raise ValueError(f"Full train is required: cache has {len(train_ids):,} rows, needs exactly 113,287 saved train IDs ({missing:,} missing). Finish expanded extraction before training.")
    assert not set(train_ids) & (set(data["split_ids"]["val"]) | set(data["split_ids"]["test"]))
    assert region_files["val2017"]["image_ids"] == data["split_ids"]["val"]
    assert len(data["split_ids"]["val"]) == 5000
    return train_ids


def make_final_config(data, region_files, code_identity):
    train_ids = require_full_fusion_train(data,region_files)
    return {"run_name":RUN_NAME,"seed":42,"batch_size":64,"validation_batch_size":64,
        "learning_rate":0.0001,"core_learning_rate":0.0001,"fusion_learning_rate":0.0005,
        "optimizer":"Adam","weight_decay":0.0,"amp":True,"gradient_clip":5.0,
        "max_epochs":100,"patience":3,"max_generation_steps":33,
        "print_every_batches":100,"checkpoint_every_batches":200,"evaluate_initial":True,
        "epoch_admission_deadline_utc":EPOCH_ADMISSION_DEADLINE_UTC,
        "hard_deadline_utc":HARD_DEADLINE_UTC,"validation_reserve_seconds":300,
        "d_model":512,"num_layers":3,"nhead":8,"dim_feedforward":2048,"dropout":0.3,
        "grid_size":7,"grid_feature_dim":2048,"region_feature_dim":1024,"max_regions":36,
        "fusion":"Separate learned projections+LayerNorm, grid row/column and region xyxy positions, source embeddings; concatenated visual memory",
        "cross_attention":"Three trainable tanh-gated cross-attention residuals, initialized at zero",
        "prefix":"Exact existing float32 global ResNet50 features preserve the pretrained image prefix",
        "cnn_trainable":False,"trainable":"Caption decoder, visual projections, fusion positions/norms/gates/cross-attention",
        "warm_start_run":WARM_START_RUN,"warm_start_sha256":WARM_START_SHA256,
        "warm_start_caption_train_images":113287,"new_fusion_train_images":len(train_ids),
        "new_fusion_train_is_subset":False,"full_train_required":True,
        "train_policy":"All 113,287 saved train image IDs are required; sorted IDs for caption selection. No partial-cache training.",
        "region_cache_namespace":"fasterrcnn_fusion_expanded_v1",
        "extraction_policy":"The 40k pilot is reused through an audited migration; the expanded cache must contain all 113,287 train and 5,000 validation images.",
        "region_cache_key":region_files["train2017"]["cache_key"],
        "region_cache_producer_code_id":region_files["train2017"]["meta"]["config"]["producer_code_id"],
        "implementation_id":code_identity["model_runtime_implementation_id"],"code":code_identity,
        "test_policy":"No test inference or metrics. COCO detector pretraining overlaps the source of the project holdout.",
        "baseline_policy":"Evaluate initialized model on all5000 validation images at epoch0; retain it if no trained epoch improves CIDEr-D."}


def require_final_preflight(path, code_identity, data):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report.get("passed") is True, "A successful real GPU preflight is required."
    model_hash = report.get("model_source_sha256",report.get("model_sha256"))
    runtime_hash = report.get("runtime_source_sha256",report.get("runtime_sha256"))
    assert model_hash == code_identity["models_sha256"], "Model code changed after preflight."
    assert runtime_hash == code_identity["runtime_sha256"], "Runtime code changed after preflight."
    if "split_id" in report:
        assert report["split_id"] == data["provenance"]["split_id"]
    return report


@contextmanager
def final_training_lock(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    path = output_root/"fusion_training.lock"
    token = uuid.uuid4().hex
    identity = {"token":token,"pid":os.getpid(),"host":socket.gethostname(),
                "started_utc":datetime.now(timezone.utc).isoformat(),"run_name":RUN_NAME}
    try:
        with path.open("x",encoding="utf-8") as stream:
            json.dump(identity,stream,indent=2)
    except FileExistsError as error:
        raise RuntimeError(f"An active/stale fusion lock exists: {path}. Check its PID before starting another process; never remove a live lock.") from error
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("token") == token:
            path.unlink()


def start_final_training(data, project_dir, code_identity, preflight_path=None, device=None):
    project_dir = Path(project_dir)
    output_root = project_dir/"experiments/final_fusion_v1"
    checks = output_root/"checks"
    if preflight_path is None:
        preflight_path = checks/"gpu_preflight_verification.json"
    preflight = require_final_preflight(preflight_path,code_identity,data)
    evaluator = MetricEvaluator(data["text_settings"])
    control = validate_final_control(data,project_dir,evaluator)
    grid_files = load_final_grid_features(data,project_dir)
    regions = load_fusion_region_cache(data,project_dir/"cache/features/fasterrcnn_fusion_expanded_v1",
                                      producer_code_id=code_identity["region_producer_code_id"])
    train_ids = require_full_fusion_train(data,regions)
    selected_path = project_dir/"cache/features/fasterrcnn_fusion_expanded_v1/selected_train_ids.json"
    assert json.loads(selected_path.read_text(encoding="utf-8")) == train_ids
    config = make_final_config(data,regions,code_identity)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The scheduled final run requires the checked CUDA GPU.")
    with final_training_lock(output_root):
        checks.mkdir(parents=True,exist_ok=True)
        fusion_json(checks/"gpu_preflight_verification.json",preflight)
        train_data = FusionCaptionDataset(data,grid_files,regions,"train",image_ids=train_ids,seed=42)
        val_data = FusionCaptionDataset(data,grid_files,regions,"val",seed=42)
        def factory():
            model = FusionCaptionTransformer(len(data["id_to_token"]),data["ids"]["pad"])
            warm = warm_start_from_prefix(model,control/"best_weights.pt",WARM_START_SHA256)
            fusion_json(checks/"warm_start_report.json",warm)
            return model
        def optimizer_factory(model):
            groups, report = fusion_parameter_groups(model,core_lr=config["core_learning_rate"],fusion_lr=config["fusion_learning_rate"])
            fusion_json(checks/"optimizer_groups.json",report)
            return torch.optim.Adam(groups,weight_decay=0.0)
        try:
            print("Final fusion uses",len(train_ids),"fixed train images and all5000 validation images.",flush=True)
            result = train_fusion(data,config,evaluator,device,output_root,factory,train_data,val_data,
                                  optimizer_factory=optimizer_factory,resume=True)
        finally:
            train_data.close()
            val_data.close()
            gc.collect()
            torch.cuda.empty_cache()
    return result


def extract_final_regions(data, project_dir, raw_root, code_identity, allow_download=False):
    """Extract every saved train image; an unfinished deadline pauses instead of accepting a subset."""
    directory = Path(project_dir)/"cache/features/fasterrcnn_fusion_expanded_v1"
    try:
        regions = load_fusion_region_cache(data,directory,producer_code_id=code_identity["region_producer_code_id"])
        require_full_fusion_train(data,regions)
        return regions
    except FileNotFoundError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("This timeboxed extraction requires CUDA.")
    order = random.Random(42).sample(sorted(data["split_ids"]["train"]),len(data["split_ids"]["train"]))
    assert len(order) == 113287
    detector = build_frozen_region_detector("cuda",allow_download=allow_download)
    try:
        regions = extract_fusion_region_cache(data,detector,directory,raw_root,order,
            deadline_utc=EXTRACTION_DEADLINE_UTC,batch_size=1,max_train_images=len(order),
            min_train_images=len(order),producer_code_id=code_identity["region_producer_code_id"])
        require_full_fusion_train(data,regions)
        return regions
    finally:
        del detector
        gc.collect()
        torch.cuda.empty_cache()


def show_final_status(project_dir, plot=False):
    project_dir = Path(project_dir)
    region_dir = project_dir/"cache/features/fasterrcnn_fusion_expanded_v1"
    for source in ["val2017","train2017"]:
        path = region_dir/f"{source}_meta.json"
        if not path.exists():
            path = region_dir/f"{source}_progress.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            print("Region cache",source,"| status",item.get("status"),"| rows",item.get("processed",item.get("actual_count")),
                  "| active seconds",round(item.get("elapsed_seconds",0),1))
    folder = project_dir/"experiments/final_fusion_v1"/RUN_NAME
    summary_path = folder/"summary.json"
    if not summary_path.exists():
        print("No completed validation checkpoint is recorded for the final fusion run yet.")
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if summary.get("best_is_initial_baseline"):
        print("The best checkpoint is still the initialized pretrained baseline, not an achieved fusion improvement.")
    if summary.get("phase") == "finished" and (folder/"best_weights.pt").is_file():
        weights = torch.load(folder/"best_weights.pt",map_location="cpu",weights_only=True)
        gates = [float(torch.tanh(weights[f"layers.{i}.cross_gate"])) for i in range(3)]
        print("Best checkpoint cross-attention gates:",gates)
        del weights
    if plot and (folder/"history.json").exists():
        history = json.loads((folder/"history.json").read_text(encoding="utf-8"))
        if history:
            figure,axes = plt.subplots(1,2,figsize=(11,4))
            epochs = [row["epoch"] for row in history]
            axes[0].plot(epochs,[row["cider_d"] for row in history],marker="o")
            axes[0].set(title="Full validation CIDEr-D",xlabel="Epoch (0 = initialized baseline)",ylabel="Native CIDEr-D")
            axes[1].plot(epochs,[row["val_loss"] for row in history],marker="o",label="Validation")
            learned = [row for row in history if row.get("train_loss") is not None]
            axes[1].plot([row["epoch"] for row in learned],[row["train_loss"] for row in learned],marker="o",label="Train")
            axes[1].set(title="Token-weighted caption loss",xlabel="Epoch",ylabel="Cross entropy")
            axes[1].legend()
            figure.tight_layout()
            plt.show()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true",help="Explicitly start/resume the final training run after GPU preflight")
    parser.add_argument("--project",type=Path,default=PROJECT_DIR)
    parser.add_argument("--raw",type=Path,default=RAW_IMAGE_ROOT)
    parser.add_argument("--preflight",type=Path,help="Successful GPU preflight report; copied into project checks on start")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.execute:
        print("Review mode. Pass --execute only after complete caches and successful GPU preflight.")
        show_final_status(args.project)
        return
    identity = final_code_identity(source_dir=BASE)
    data = load_region_prepared_data(args.project,args.raw)
    result = start_final_training(data,args.project,identity,preflight_path=args.preflight)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
