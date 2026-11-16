"""Grouped validation search: greedy and three beam widths with five length penalties."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

BASE = Path(__file__).parent.resolve()
FINAL_SOURCE = BASE.parent/"final_fusion"
sys.path.insert(0,str(FINAL_SOURCE))
from train_run import (PROJECT_DIR,RAW_IMAGE_ROOT,final_code_identity,load_final_grid_features,
                       load_region_prepared_data,load_fusion_region_cache,MetricEvaluator)
from models import FusionCaptionTransformer
from runtime import FusionCaptionDataset,fusion_loader,fusion_device_batch,fusion_seed
from beam_search import fusion_beam_search
from beam_search_many import fusion_beam_search_many

# Reuse unchanged, previously checked snapshot/JSON/record utilities by explicit path.
# Their definitions already exist in the self-contained FinalFusion notebook.
_shared_path = BASE.parent/"fusion_beam_followup_20261115/evaluate_decoding.py"
_shared_spec = importlib.util.spec_from_file_location("fusion_search_shared_evaluation",_shared_path)
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)
beam_eval_hash = _shared.beam_eval_hash
beam_eval_file_sha = _shared.beam_eval_file_sha
beam_eval_json = _shared.beam_eval_json
beam_eval_function_id = _shared.beam_eval_function_id
beam_evaluation_lock = _shared.beam_evaluation_lock
beam_validate_records = _shared.beam_validate_records
load_beam_snapshot = _shared.load_beam_snapshot


def search_mode_name(width,alpha):
    return f"beam{width}_alpha{alpha:g}".replace(".","p")


def search_group_modes(group,config):
    if group == "greedy":
        return ["greedy"]
    if group not in [f"beam{width}" for width in config["beam_sizes"]]:
        raise ValueError("The generation group is not configured.")
    return [search_mode_name(int(group[4:]),alpha) for alpha in config["length_penalties"]]


def search_validate_chunk(chunk,identity_hash,group,first,last,dataset,data,config):
    if (chunk.get("identity_sha256") != identity_hash or chunk.get("group") != group or
            chunk.get("first_index") != first or chunk.get("last_index") != last):
        raise ValueError("Grouped prediction chunk belongs to a different snapshot/configuration/range.")
    seconds = chunk.get("generation_seconds")
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid grouped generation timer.")
    if set(chunk.get("records_by_mode",{})) != set(search_group_modes(group,config)):
        raise ValueError("A grouped chunk must contain every configured alpha atomically.")
    for records in chunk["records_by_mode"].values():
        beam_validate_records(records,dataset.image_ids[first:last],data,config["max_steps"])
    if first == 0 and group == "beam3" and config["verify_reference"]:
        reference = chunk.get("reference_check",{})
        if reference.get("passed") is not True or reference.get("images") != last-first:
            raise ValueError("The first width3 chunk lacks its original-helper equivalence check.")


def search_load_chunks(folder,identity_hash,group,dataset,data,config):
    chunk_dir = Path(folder)/"chunks"
    size = config["batch_size"]
    names = {f"batch_{first:05d}_{min(first+size,len(dataset)):05d}.json" for first in range(0,len(dataset),size)}
    if any(path.name not in names for path in chunk_dir.glob("*.json")):
        raise ValueError("Unexpected grouped chunk name.")
    modes = search_group_modes(group,config)
    records = {mode:[] for mode in modes}
    seconds,missing,reference = 0.0,False,None
    for first in range(0,len(dataset),size):
        last = min(first+size,len(dataset))
        path = chunk_dir/f"batch_{first:05d}_{last:05d}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise ValueError("A grouped chunk follows a missing earlier chunk.")
        chunk = json.loads(path.read_text(encoding="utf-8"))
        search_validate_chunk(chunk,identity_hash,group,first,last,dataset,data,config)
        for mode in modes:
            records[mode].extend(chunk["records_by_mode"][mode])
        seconds += chunk["generation_seconds"]
        if first == 0: reference = chunk.get("reference_check")
    return records,seconds,reference


def search_score_variant(mode,group,records,shared_seconds,output_dir,identity_hash,dataset,data,config,evaluator):
    beam_validate_records(records,dataset.image_ids,data,config["max_steps"])
    folder = Path(output_dir)/"modes"/mode
    folder.mkdir(parents=True,exist_ok=True)
    prediction_hash = beam_eval_hash(records)
    prediction_path = folder/"validation_predictions.json"
    if prediction_path.exists():
        if beam_eval_hash(json.loads(prediction_path.read_text(encoding="utf-8"))) != prediction_hash:
            raise ValueError("Final predictions differ from validated grouped chunks.")
    else:
        beam_eval_json(prediction_path,records)
    metrics_path = folder/"metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (saved.get("identity_sha256") != identity_hash or saved.get("predictions_sha256") != prediction_hash or
                saved.get("status") != "complete" or saved.get("mode") != mode or saved.get("generation_group") != group or
                saved.get("images") != len(dataset) or set(saved.get("scores",{})) != {"bleu1","bleu4","cider_d"} or
                not all(math.isfinite(float(score)) for score in saved["scores"].values())):
            raise ValueError("Saved metrics belong to a different full prediction corpus.")
        return saved
    started = time.perf_counter()
    scores = evaluator.score(records,expected_ids=dataset.image_ids)
    scoring_seconds = time.perf_counter()-started
    if set(scores) != {"bleu1","bleu4","cider_d"} or not all(math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("Unexpected or nonfinite corpus scores.")
    result = {"status":"complete","identity_sha256":identity_hash,"mode":mode,"images":len(records),
              "generation_group":group,"shared_group_generation_seconds":shared_seconds,
              "individual_generation_seconds":None,"scoring_seconds":scoring_seconds,"scores":scores,
              "predictions_sha256":prediction_hash,"predictions_file":str(prediction_path.resolve()),
              "average_words":sum(len(record["token_ids"]) for record in records)/len(records),
              "timing_note":"Generation belongs to the shared width pass; do not sum it across alpha rows."}
    beam_eval_json(metrics_path,result)
    return result


@torch.no_grad()
def evaluate_search_group(model,dataset,data,config,device,evaluator,group,output_dir,identity_hash,
                           max_new_batches=None,after_chunk=None):
    """One traversal per width; all alpha outputs for each batch are committed together."""
    if dataset.split != "val":
        raise ValueError("Only the labeled validation split is evaluated.")
    if max_new_batches is not None and (type(max_new_batches) is not int or max_new_batches < 1):
        raise ValueError("max_new_batches must be a positive optional work limit.")
    modes = search_group_modes(group,config)
    folder = Path(output_dir)/"groups"/group
    (folder/"chunks").mkdir(parents=True,exist_ok=True)
    records,seconds,reference = search_load_chunks(folder,identity_hash,group,dataset,data,config)
    completed = len(records[modes[0]])
    device = torch.device(device)
    enabled = bool(config["amp"] and device.type == "cuda")
    model.eval()
    fusion_seed(config["seed"])
    loader = iter(fusion_loader(dataset,config["batch_size"],indices=list(range(completed,len(dataset))),
                               seed=config["seed"],pin_memory=enabled))
    batches = 0
    try:
        while completed < len(dataset):
            if max_new_batches is not None and batches >= max_new_batches: break
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = next(loader)
            moved = fusion_device_batch(batch,device)
            extras = {"region_mask":moved["region_mask"],"region_boxes":moved["region_boxes"],
                      "global_features":moved["global_features"]}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):
                if group == "greedy":
                    predictions = {"greedy":model.greedy(moved["grid"],moved["regions"],data["ids"]["start"],data["ids"]["end"],
                                                       max_steps=config["max_steps"],**extras)}
                else:
                    width = int(group[4:])
                    outputs = fusion_beam_search_many(model,moved["grid"],moved["regions"],data["ids"]["start"],data["ids"]["end"],
                        max_steps=config["max_steps"],beam_size=width,length_penalties=config["length_penalties"],**extras)
                    if set(outputs) != set(config["length_penalties"]):
                        raise ValueError("Grouped beam search omitted or added a length penalty.")
                    predictions = {search_mode_name(width,alpha):outputs[alpha] for alpha in config["length_penalties"]}
            chunk_records = {}
            for mode,tokens in predictions.items():
                if len(tokens) != len(batch["image_ids"]): raise ValueError("Decoder caption count differs from image count.")
                chunk_records[mode] = [{"image_id":int(image_id),"token_ids":row,
                    "prediction":" ".join(data["id_to_token"][token] for token in row),"references":refs}
                    for image_id,row,refs in zip(batch["image_ids"],tokens,batch["references"])]
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-started
            first,last = completed,completed+len(batch["image_ids"])
            if first == 0 and group == "beam3" and config["verify_reference"]:
                check_started = time.perf_counter()
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):
                    original = fusion_beam_search(model,moved["grid"],moved["regions"],data["ids"]["start"],data["ids"]["end"],
                        max_steps=config["max_steps"],beam_size=3,length_penalty=0.6,**extras)
                if original != predictions[search_mode_name(3,0.6)]:
                    raise AssertionError("Grouped width3/alpha0.6 differs from the original beam helper on the same batch.")
                if device.type == "cuda": torch.cuda.synchronize(device)
                reference = {"passed":True,"images":last-first,"beam_size":3,"length_penalty":0.6,
                             "seconds":time.perf_counter()-check_started,"note":"Excluded from the shared generation timer; included in total wall time."}
                print("Real-batch grouped/original beam3 alpha0.6 equivalence passed.",flush=True)
            chunk = {"identity_sha256":identity_hash,"group":group,"first_index":first,"last_index":last,
                     "generation_seconds":elapsed,"records_by_mode":chunk_records,
                     "reference_check":reference if first == 0 else None}
            search_validate_chunk(chunk,identity_hash,group,first,last,dataset,data,config)
            beam_eval_json(folder/"chunks"/f"batch_{first:05d}_{last:05d}.json",chunk)
            for mode in modes: records[mode].extend(chunk_records[mode])
            completed,seconds,batches = last,seconds+elapsed,batches+1
            beam_eval_json(folder/"progress.json",{"status":"generating","identity_sha256":identity_hash,"group":group,
                "completed_images":completed,"required_images":len(dataset),"shared_generation_seconds":seconds,
                "modes":modes,"updated_utc":datetime.now(timezone.utc).isoformat()})
            if batches == 1 or batches % 10 == 0 or completed == len(dataset):
                print(f"{group} ({len(modes)} variants): {completed:,}/{len(dataset):,} | {completed/max(seconds,1e-9):.1f} img/s | shared {seconds:.1f} s",flush=True)
            if after_chunk is not None: after_chunk(chunk)
            del batch,moved,predictions,chunk_records
        if completed < len(dataset):
            partial = {"status":"partial","group":group,"completed_images":completed,"required_images":len(dataset),
                       "generation_seconds":seconds,"modes":modes,"metrics_computed":False,"reference_check":reference}
            beam_eval_json(folder/"progress.json",partial)
            return partial
        results = {}
        for mode in modes:
            results[mode] = search_score_variant(mode,group,records[mode],seconds,output_dir,identity_hash,dataset,data,config,evaluator)
            if after_chunk is not None: after_chunk(None)
        complete = {"status":"complete","group":group,"images":len(dataset),"generation_seconds":seconds,
                    "modes":modes,"results":results,"metrics_computed":True,"reference_check":reference}
        beam_eval_json(folder/"progress.json",complete)
        return complete
    except BaseException as error:
        beam_eval_json(folder/"progress.json",{"status":"interrupted","group":group,"completed_images":completed,
            "required_images":len(dataset),"shared_generation_seconds":seconds,"error_type":type(error).__name__,
            "error":str(error),"metrics_computed":False})
        raise


def run_fusion_decoding_search(project_dir,snapshot_dir,output_dir,raw_root=None,code_identity=None,
                               batch_size=64,beam_sizes=(3,5,7),length_penalties=(0.0,0.3,0.6,0.9,1.2),
                               amp=True,device=None,max_new_batches=None,verify_reference=True):
    if (type(batch_size) is not int or batch_size < 1 or not isinstance(beam_sizes,(tuple,list)) or not beam_sizes or
            any(type(width) is not int or width < 2 for width in beam_sizes) or len(set(beam_sizes)) != len(beam_sizes)):
        raise ValueError("Use a positive batch size and distinct integer beam widths >=2.")
    if (not isinstance(length_penalties,(tuple,list)) or not length_penalties or
            any(type(alpha) not in [int,float] or not math.isfinite(alpha) or alpha < 0 for alpha in length_penalties) or
            len(set(length_penalties)) != len(length_penalties)):
        raise ValueError("Use distinct finite nonnegative length penalties.")
    if verify_reference and (3 not in beam_sizes or 0.6 not in length_penalties):
        raise ValueError("The original-helper pilot requires width3 and alpha0.6 in the fixed search.")
    mode_names = [search_mode_name(width,alpha) for width in beam_sizes for alpha in length_penalties]
    if len(mode_names) != len(set(mode_names)):
        raise ValueError("Length penalties produce colliding output names; use clearly separated values.")
    code_identity = code_identity or globals().get("EXPECTED_CODE_IDENTITY")
    if not code_identity: raise ValueError("Pass the known model/cache code identity.")
    project_dir,snapshot_dir,output_dir = Path(project_dir),Path(snapshot_dir),Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir/"evaluation_identity.json").exists():
        raise ValueError("Use a new dedicated output directory or resume its saved identity.")
    output_dir.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    dataset = model = None
    identity_hash = None
    previous_wall = 0.0
    timing_ready = False
    def save_timing(chunk=None):
        if timing_ready:
            beam_eval_json(output_dir/"timing.json",{"identity_sha256":identity_hash,
                "active_wall_seconds":previous_wall+time.perf_counter()-started,
                "note":"Includes data/model loading, shared generation, reference check, scoring and artifact I/O; excludes pauses between attempts."})
    with beam_evaluation_lock(output_dir):
        try:
            data = load_region_prepared_data(project_dir,raw_root)
            evaluator = MetricEvaluator(data["text_settings"])
            grids = load_final_grid_features(data,project_dir)
            regions = load_fusion_region_cache(data,project_dir/"cache/features/fasterrcnn_fusion_expanded_v1",
                                              producer_code_id=code_identity["region_producer_code_id"])
            dataset = FusionCaptionDataset(data,grids,regions,"val",seed=42)
            payload,snapshot = load_beam_snapshot(snapshot_dir,data,evaluator,dataset,code_identity)
            device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
            details = {"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cpu_threads":torch.get_num_threads()}
            config = {"seed":42,"batch_size":batch_size,"amp":bool(amp),"amp_enabled":bool(amp and device.type == "cuda"),
                "max_steps":33,"beam_sizes":list(beam_sizes),"length_penalties":[float(alpha) for alpha in length_penalties],
                "verify_reference":bool(verify_reference),"groups":["greedy",*[f"beam{width}" for width in beam_sizes]],
                "length_penalty_formula":"((5+L)/6)^alpha; L excludes START and includes emitted EOS",
                "selection":"Fixed 16-variant validation decoding search; no model training or test evaluation"}
            identity = {"snapshot_full_checkpoint_sha256":snapshot["full_checkpoint_sha256"],"snapshot_epoch":snapshot["snapshot_epoch"],
                "model_source_sha256":code_identity["models_sha256"],"data":data["provenance"],"validation_ids":dataset.image_ids,
                "cache":dataset.provenance,"scorer":evaluator.metadata,"config":config,"device":details,
                "beam_many_implementation_id":beam_eval_function_id([fusion_beam_search_many]),
                "original_beam_implementation_id":beam_eval_function_id([fusion_beam_search]),
                "evaluation_implementation_id":beam_eval_function_id([search_mode_name,search_group_modes,search_validate_chunk,
                    search_load_chunks,search_score_variant,evaluate_search_group,beam_validate_records])}
            identity_path = output_dir/"evaluation_identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Snapshot/search settings/device/code changed; use a different evaluation directory.")
            beam_eval_json(identity_path,identity)
            identity_hash = beam_eval_hash(identity)
            timing_path = output_dir/"timing.json"
            if timing_path.exists():
                saved = json.loads(timing_path.read_text(encoding="utf-8"))
                if saved["identity_sha256"] != identity_hash: raise ValueError("Timing belongs to another evaluation.")
                previous_wall = float(saved["active_wall_seconds"])
                if not math.isfinite(previous_wall) or previous_wall < 0:
                    raise ValueError("Saved evaluation wall time must be finite and nonnegative.")
            timing_ready = True
            beam_eval_json(output_dir/"snapshot_metadata.json",snapshot)
            beam_eval_json(output_dir/"scorer.json",evaluator.metadata)
            beam_eval_json(output_dir/"validation_image_ids.json",dataset.image_ids)
            trained = payload["identity"]["config"]
            model = FusionCaptionTransformer(len(data["id_to_token"]),data["ids"]["pad"],
                grid_feature_dim=trained["grid_feature_dim"],region_feature_dim=trained["region_feature_dim"],
                d_model=trained["d_model"],num_layers=trained["num_layers"],nhead=trained["nhead"],
                dim_feedforward=trained["dim_feedforward"],dropout=trained["dropout"],grid_size=trained["grid_size"],
                max_regions=trained["max_regions"],max_positions=34)
            model.load_state_dict(payload["model"],strict=True)
            del payload
            model = model.to(device).eval().requires_grad_(False)
            groups = {group:evaluate_search_group(model,dataset,data,config,device,evaluator,group,output_dir,identity_hash,
                                                   max_new_batches=max_new_batches,after_chunk=save_timing)
                      for group in config["groups"]}
            complete = all(group["status"] == "complete" for group in groups.values())
            results = {mode:result for group in groups.values() for mode,result in group.get("results",{}).items()}
            comparison = {"status":"complete" if complete else "partial","identity_sha256":identity_hash,
                "snapshot_epoch":snapshot["snapshot_epoch"],"snapshot_full_checkpoint_sha256":snapshot["full_checkpoint_sha256"],
                "validation_images":5000,"configured_variants":1+len(beam_sizes)*len(length_penalties),
                "groups":groups,"results":results,"device":details,
                "unique_generation_seconds":sum(group["generation_seconds"] for group in groups.values()),
                "total_scoring_seconds":sum(result["scoring_seconds"] for result in results.values()),
                "active_wall_seconds":previous_wall+time.perf_counter()-started,
                "timing_note":"Each width generation pass is counted once. Alpha variants share that pass; their repeated CSV timer column must not be summed. Wall time includes setup, scoring, reference check and I/O.",
                "test_evaluated":False,"model_training_performed":False,"original_checkpoints_modified":False}
            if complete:
                comparison["selected_mode_by_validation_cider"] = max(results,key=lambda mode:results[mode]["scores"]["cider_d"])
                comparison["differences_from_greedy"] = {mode:{key:result["scores"][key]-results["greedy"]["scores"][key]
                    for key in ["cider_d","bleu1","bleu4"]} for mode,result in results.items() if mode != "greedy"}
                temporary = output_dir/"comparison.csv.tmp"
                columns = ["mode","generation_group","snapshot_epoch","images","bleu1","bleu4","cider_d",
                           "shared_group_generation_seconds","scoring_seconds","average_words","device"]
                with temporary.open("w",newline="",encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream,fieldnames=columns)
                    writer.writeheader()
                    for mode,result in results.items():
                        writer.writerow({"mode":mode,"generation_group":result["generation_group"],"snapshot_epoch":snapshot["snapshot_epoch"],
                            "images":result["images"],**result["scores"],"shared_group_generation_seconds":result["shared_group_generation_seconds"],
                            "scoring_seconds":result["scoring_seconds"],"average_words":result["average_words"],"device":details["gpu"] or details["device"]})
                temporary.replace(output_dir/"comparison.csv")
            beam_eval_json(output_dir/"comparison.json",comparison)
            save_timing()
            return comparison
        finally:
            if dataset is not None: dataset.close()
            model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            save_timing()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--project",type=Path,default=PROJECT_DIR)
    parser.add_argument("--raw",type=Path,default=RAW_IMAGE_ROOT)
    parser.add_argument("--snapshot",type=Path,default=BASE/"best_snapshot")
    parser.add_argument("--output",type=Path,default=BASE/"validation_search")
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--max-new-batches",type=int)
    args = parser.parse_args()
    if not args.execute:
        print("Decoding search is off. Add --execute; --max-new-batches 1 performs a resumable one-batch-per-width pilot.")
        return
    torch.set_num_threads(2)
    result = run_fusion_decoding_search(args.project,args.snapshot,args.output,raw_root=args.raw,
        code_identity=final_code_identity(source_dir=FINAL_SOURCE),batch_size=args.batch_size,max_new_batches=args.max_new_batches)
    print(json.dumps({"status":result["status"],"snapshot_epoch":result["snapshot_epoch"],
        "selected_mode_by_validation_cider":result.get("selected_mode_by_validation_cider"),
        "unique_generation_seconds":result["unique_generation_seconds"],"active_wall_seconds":result["active_wall_seconds"]},indent=2))


if __name__ == "__main__":
    main()
